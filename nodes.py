"""CutSwapWan (CSW) — scene cuts → face-swap prep → Wan wipe redraw → assemble (ComfyUI)."""

from __future__ import annotations

import json
from pathlib import Path

import folder_paths
import numpy as np
import torch


def _tensor_batch_to_uint8(images: torch.Tensor) -> np.ndarray:
    """Comfy IMAGE [B,H,W,C] float 0..1 → uint8 NHWC."""
    arr = images.detach().cpu().numpy()
    if arr.dtype != np.float32 and arr.dtype != np.float64:
        arr = arr.astype(np.float32)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def _normalize_rgb_u8(fr: np.ndarray) -> np.ndarray:
    if fr.ndim == 2:
        fr = np.stack([fr] * 3, axis=-1)
    if fr.shape[-1] == 4:
        fr = fr[..., :3]
    return fr


def _pad_frames_to_common(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Letterbox-pad to max H/W so preview batch can torch.stack (masters may differ)."""
    frames = [_normalize_rgb_u8(fr) for fr in frames]
    mh = max(fr.shape[0] for fr in frames)
    mw = max(fr.shape[1] for fr in frames)
    out = []
    for fr in frames:
        h, w = fr.shape[:2]
        if h == mh and w == mw:
            out.append(fr)
            continue
        canvas = np.zeros((mh, mw, 3), dtype=fr.dtype)
        y0 = (mh - h) // 2
        x0 = (mw - w) // 2
        canvas[y0 : y0 + h, x0 : x0 + w] = fr
        out.append(canvas)
    return out


def _uint8_to_tensor_batch(frames: list[np.ndarray]) -> torch.Tensor:
    frames = _pad_frames_to_common(frames)
    tensors = [torch.from_numpy(fr.astype(np.float32) / 255.0) for fr in frames]
    return torch.stack(tensors, dim=0)


def _hist_score(a: np.ndarray, b: np.ndarray) -> float:
    """1 = identical-ish, 0 = very different (BGR/RGB uint8)."""
    import cv2

    ha = cv2.calcHist([a], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hb = cv2.calcHist([b], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(ha, ha)
    cv2.normalize(hb, hb)
    return float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))


def _mask_feats(mask_u8: np.ndarray) -> tuple[float, float, float, float, int]:
    """area_frac, cx, cy, scale, visible."""
    g = mask_u8[..., 0] if mask_u8.ndim == 3 else mask_u8
    ys, xs = np.where(g > 127)
    h, w = g.shape[:2]
    if len(xs) < 32:
        return 0.0, float("nan"), float("nan"), 0.0, 0
    area = len(xs) / float(w * h)
    cx = float(xs.mean()) / w
    cy = float(ys.mean()) / h
    bw = (xs.max() - xs.min() + 1) / w
    bh = (ys.max() - ys.min() + 1) / h
    scale = float(np.hypot(bw, bh))
    return area, cx, cy, scale, 1


def detect_segment_starts(
    frames: np.ndarray,
    masks: np.ndarray | None,
    *,
    min_len: int,
    hist_corr_cut: float,
    jump_center: float,
    jump_scale: float,
    use_camera: bool,
    use_mask: bool,
) -> list[dict]:
    """Return list of {start, reason} cut points; always includes 0."""
    n = frames.shape[0]
    cut_at = [0]
    reasons = {0: "start"}

    mask_cx = mask_cy = mask_sc = mask_vis = mask_area = None
    if use_mask and masks is not None:
        feats = [_mask_feats(masks[i]) for i in range(min(n, masks.shape[0]))]
        while len(feats) < n:
            feats.append(feats[-1] if feats else (0.0, 0.5, 0.5, 0.0, 0))
        mask_area = [f[0] for f in feats]
        mask_cx = [f[1] for f in feats]
        mask_cy = [f[2] for f in feats]
        mask_sc = [f[3] for f in feats]
        mask_vis = [f[4] for f in feats]
        for i in range(n):
            if mask_vis[i] and (mask_cx[i] != mask_cx[i]):  # nan
                mask_cx[i] = mask_cx[i - 1] if i else 0.5
                mask_cy[i] = mask_cy[i - 1] if i else 0.5

    def _mask_jump_score(i: int) -> float:
        """How strong a mask geometry jump is at frame i vs i-1. 0 = none."""
        if mask_vis is None or i <= 0:
            return 0.0
        score = 0.0
        if mask_vis[i] != mask_vis[i - 1]:
            # Prefer mask APPEARING (0→1) over disappearing (1→0): real shot usually starts with matte on.
            score += 2.0 if mask_vis[i] and not mask_vis[i - 1] else 0.35
        if mask_vis[i] and mask_vis[i - 1]:
            dc = float(np.hypot(mask_cx[i] - mask_cx[i - 1], mask_cy[i] - mask_cy[i - 1]))
            if dc >= jump_center:
                score += dc / max(jump_center, 1e-6)
            s0, s1 = float(mask_sc[i - 1]), float(mask_sc[i])
            if s0 > 1e-4:
                rel = abs(s1 - s0) / s0
                if rel >= jump_scale:
                    score += rel / max(jump_scale, 1e-6)
            # Area explode after a collapse (empty-ish transition frame).
            a0, a1 = float(mask_area[i - 1]), float(mask_area[i])
            if a0 > 1e-6 and a1 / a0 >= 3.0:
                score += a1 / a0
        return score

    for i in range(1, n):
        why = []
        strong_mask = False
        corr = 1.0
        if use_camera:
            corr = _hist_score(frames[i - 1], frames[i])
            if corr < hist_corr_cut:
                why.append(f"cameraΔ corr={corr:.2f}")
        # Same continuous look (zoom/pan of one shot): hist still high → ignore pure
        # matte scale/center jumps. Those fire on zoom because the FaceHair bbox grows.
        continuous_shot = (not use_camera) or (corr >= hist_corr_cut)
        if use_mask and mask_vis is not None:
            # Disappear alone is weak; appear / geometry / area-explode are real shot changes.
            if (not mask_vis[i - 1]) and mask_vis[i]:
                why.append("mask_visibility")
                strong_mask = True
            elif mask_vis[i] and mask_vis[i - 1]:
                dc = float(
                    np.hypot(mask_cx[i] - mask_cx[i - 1], mask_cy[i] - mask_cy[i - 1])
                )
                s0, s1 = mask_sc[i - 1], mask_sc[i]
                rel_scale = (abs(s1 - s0) / s0) if s0 > 1e-4 else 0.0
                a0, a1 = float(mask_area[i - 1]), float(mask_area[i])
                if a0 > 1e-6 and a1 / a0 >= 3.0:
                    why.append(f"mask_area×{a1 / a0:.1f}")
                    strong_mask = True
                # Collapse (matte suddenly tiny) — keep as its own beat, e.g. 1-frame two-shot.
                if a0 > 1e-6 and a1 / a0 <= 0.45 and a1 < 0.05:
                    why.append(f"mask_collapse×{a1 / a0:.2f}")
                    strong_mask = True
                if continuous_shot:
                    # Zoom of one plan: only keep brutal reframes (half-frame jump / 2× scale).
                    if dc >= max(jump_center * 3.0, 0.22):
                        why.append(f"mask_centerΔ={dc:.2f}")
                        strong_mask = True
                    if rel_scale >= max(jump_scale * 4.0, 0.80):
                        why.append(f"mask_scaleΔ={rel_scale:.2f}")
                        strong_mask = True
                else:
                    if dc >= jump_center:
                        why.append(f"mask_centerΔ={dc:.2f}")
                    if rel_scale >= jump_scale:
                        why.append(f"mask_scaleΔ={rel_scale:.2f}")
        gap = i - cut_at[-1]
        # Strong matte explode/collapse may start a new SEG even inside min_len (need BOTH the
        # 1-frame transition AND the following OTS, not only one of them).
        gap_ok = gap >= min_len or (strong_mask and gap >= 1 and bool(why))
        if why and gap_ok:
            cut_at.append(i)
            reasons[i] = ",".join(why)

    # After a collapse cut, also ensure the following matte-explode frame is a cut (OTS etc.).
    if use_mask and mask_area is not None and len(cut_at) > 1:
        extra: list[int] = []
        for c in list(cut_at[1:]):
            a_c = float(mask_area[c])
            a_prev = float(mask_area[max(0, c - 1)])
            if a_c > max(1e-6, a_prev * 0.45):
                continue
            for j in range(c + 1, min(n, c + max(2, min_len))):
                a_j = float(mask_area[j])
                if a_j >= max(a_c * 3.0, 0.02) and a_j > a_c:
                    if j not in cut_at and j not in extra:
                        extra.append(j)
                        reasons[j] = f"mask_area×{a_j / max(a_c, 1e-6):.1f},after_collapse"
                    break
        if extra:
            cut_at = sorted(set(cut_at + extra))

    edges = cut_at + [n]
    segs = []
    for si, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        n_fr = int(b - a)
        reason = reasons.get(a, "cont")
        # Keep intentional short beats (1-frame two-shot etc.); only merge tiny noise without mask reason.
        keep_short = n_fr >= 1 and any(
            k in reason for k in ("mask_collapse", "mask_area", "after_collapse", "mask_scale", "mask_center")
        )
        if n_fr < min_len and segs and not keep_short:
            segs[-1]["end"] = b
            segs[-1]["n_frames"] = segs[-1]["end"] - segs[-1]["start"]
            segs[-1]["reason"] += "+merge_short"
            continue
        segs.append(
            {
                "id": f"{len(segs):02d}",
                "start": int(a),
                "end": int(b),
                "n_frames": n_fr,
                "reason": reason,
            }
        )
    for i, s in enumerate(segs):
        s["id"] = f"{i:02d}"
    return segs


def _label_frame(fr: np.ndarray, text: str, big_index: str | None = None) -> np.ndarray:
    import cv2

    out = _normalize_rgb_u8(fr).copy()
    h, w = out.shape[:2]
    # Big index on the image (visible in PreviewImage grid)
    tag = (big_index or "").strip()
    if tag:
        scale = max(1.2, min(h, w) / 180.0)
        thickness = max(2, int(scale * 2))
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        pad = 8
        x0, y0 = 10, 10
        cv2.rectangle(out, (x0, y0), (x0 + tw + pad * 2, y0 + th + pad * 2), (0, 0, 0), -1)
        cv2.putText(
            out,
            tag,
            (x0 + pad, y0 + th + pad - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    bar_h = max(36, min(72, h // 12))
    bar = np.zeros((bar_h, w, 3), np.uint8)
    font_scale = max(0.45, min(0.9, w / 900.0))
    cv2.putText(
        bar,
        text[:120],
        (8, int(bar_h * 0.7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([bar, out])


class CVFSDetectSceneCuts:
    """From a loaded video (IMAGE batch) + optional mask batch → start frames of each scene."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE",),
                "min_scene_frames": ("INT", {"default": 8, "min": 2, "max": 1000}),
                "camera_hist_cut": (
                    "FLOAT",
                    {
                        "default": 0.85,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "New camera/scene if hist correlation between frames drops below this",
                    },
                ),
                "use_camera_cuts": ("BOOLEAN", {"default": True}),
                "use_mask_cuts": ("BOOLEAN", {"default": True}),
                "mask_jump_center": (
                    "FLOAT",
                    {
                        "default": 0.12,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Cut on matte center jump. On continuous zoom this is ignored unless the jump is much larger.",
                    },
                ),
                "mask_jump_scale": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Cut on matte size jump. Smooth zoom no longer cuts while the picture stays continuous.",
                    },
                ),
                "draw_labels": ("BOOLEAN", {"default": False}),
                "save_cut_frames": ("BOOLEAN", {"default": True}),
                "preserve_masters_parts": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Keep masters/*.png and parts/*.mp4; re-detect only refreshes cuts/cut_frames",
                    },
                ),
                "job_name": ("STRING", {"default": "comfy_job"}),
            },
            "optional": {
                "mask_frames": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "INT")
    RETURN_NAMES = ("scene_start_frames", "contact_sheet", "cuts_json", "num_scenes")
    FUNCTION = "run"
    CATEGORY = "cut_video_face_swapper"

    def run(
        self,
        video_frames,
        min_scene_frames,
        camera_hist_cut,
        use_camera_cuts,
        use_mask_cuts,
        mask_jump_center,
        mask_jump_scale,
        draw_labels,
        save_cut_frames,
        preserve_masters_parts,
        job_name,
        mask_frames=None,
    ):
        import cv2

        frames = _tensor_batch_to_uint8(video_frames)
        masks = _tensor_batch_to_uint8(mask_frames) if mask_frames is not None else None
        if masks is not None and masks.shape[0] != frames.shape[0]:
            # pad/truncate mask to video length
            if masks.shape[0] < frames.shape[0]:
                pad = np.repeat(masks[-1:], frames.shape[0] - masks.shape[0], axis=0)
                masks = np.concatenate([masks, pad], axis=0)
            else:
                masks = masks[: frames.shape[0]]

        segs = detect_segment_starts(
            frames,
            masks,
            min_len=int(min_scene_frames),
            hist_corr_cut=float(camera_hist_cut),
            jump_center=float(mask_jump_center),
            jump_scale=float(mask_jump_scale),
            use_camera=bool(use_camera_cuts),
            use_mask=bool(use_mask_cuts) and masks is not None,
        )

        job = _job_root(job_name)
        old_by_id = {}
        cuts_path = job / "cuts.json"
        if cuts_path.exists():
            try:
                old_by_id = {
                    s["id"]: s
                    for s in json.loads(cuts_path.read_text(encoding="utf-8")).get("segments", [])
                }
            except Exception:
                old_by_id = {}

        starts = []
        thumbs = []
        out_dir = job / "cut_frames"
        if save_cut_frames:
            out_dir.mkdir(parents=True, exist_ok=True)
            for old in out_dir.glob("*"):
                if old.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    old.unlink()

        for s in segs:
            start_i = int(s["start"])
            end_i = int(s["end"])
            ref = start_i
            if preserve_masters_parts and s["id"] in old_by_id:
                prev_ref = old_by_id[s["id"]].get("ref_frame")
                if prev_ref is not None and start_i <= int(prev_ref) < end_i:
                    ref = int(prev_ref)
            s["ref_frame"] = ref
            fr = frames[ref].copy()
            label = (
                f"SEG {s['id']}  кадр {ref}  "
                f"сцена {start_i}-{end_i - 1}  ({s['n_frames']}f)  {s['reason']}"
            )
            shown = _label_frame(fr, label, big_index=f"SEG{s['id']}") if draw_labels else fr
            starts.append(shown if draw_labels else fr)
            plain = fr
            if save_cut_frames:
                cv2.imwrite(str(out_dir / f"{s['id']}_f{ref:05d}_START.png"), cv2.cvtColor(shown, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(out_dir / f"{s['id']}_f{ref:05d}_PLAIN.png"), cv2.cvtColor(plain, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(out_dir / f"{s['id']}_REF.png"), cv2.cvtColor(plain, cv2.COLOR_RGB2BGR))
                s["cut_frame"] = f"cut_frames/{s['id']}_f{ref:05d}_PLAIN.png"
                s["master"] = f"masters/{s['id']}.png"
                s["status"] = "detected"
                # Checkpoint: keep master/part metadata from previous detect if same SEG id
                if preserve_masters_parts and s["id"] in old_by_id:
                    prev = old_by_id[s["id"]]
                    for k in ("master_source", "master"):
                        if prev.get(k):
                            s[k] = prev[k]
                    prev_status = prev.get("status") or ""
                    master_path = job / "masters" / f"{s['id']}.png"
                    part_path = job / "parts" / f"{s['id']}.mp4"
                    if part_path.exists():
                        s["status"] = "wan_done"
                    elif master_path.exists() or prev_status in {"master_bound", "ready_to_run", "wan_done"}:
                        s["status"] = "master_bound" if master_path.exists() else prev_status
            th = cv2.resize(shown, (320, int(320 * shown.shape[0] / max(1, shown.shape[1]))))
            thumbs.append(th)

        # contact sheet
        if thumbs:
            cols = min(4, len(thumbs))
            rows = (len(thumbs) + cols - 1) // cols
            th_h, th_w = thumbs[0].shape[:2]
            canvas = np.zeros((rows * th_h, cols * th_w, 3), np.uint8)
            for i, th in enumerate(thumbs):
                r, c = divmod(i, cols)
                canvas[r * th_h : (r + 1) * th_h, c * th_w : (c + 1) * th_w] = th
        else:
            canvas = frames[0].copy()

        job.mkdir(parents=True, exist_ok=True)
        (job / "masters").mkdir(exist_ok=True)
        (job / "parts").mkdir(exist_ok=True)
        (job / "workflows").mkdir(exist_ok=True)
        if save_cut_frames:
            cv2.imwrite(str(out_dir / "_contact_sheet.jpg"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

        _bake_job_media(job_name, frames, masks, segments=segs)
        _write_scrub_jpegs(job, frames)

        cuts_json = json.dumps({"num_scenes": len(segs), "segments": segs}, ensure_ascii=False, indent=2)
        start_batch = _uint8_to_tensor_batch(starts)
        sheet = _uint8_to_tensor_batch([canvas])
        return (start_batch, sheet, cuts_json, len(segs))


def _job_root(job_name: str) -> Path:
    name = str(job_name or "").strip()
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError(f"Некорректное имя job: {job_name!r}")
    return Path(folder_paths.get_input_directory()) / "cut_video_face_swapper" / "jobs" / name


def _write_scrub_jpegs(job: Path, frames_rgb: np.ndarray) -> int:
    """Small jpegs, one per driver frame, so the master-frame slider can scrub."""
    import cv2

    scrub = job / "cut_frames" / "scrub"
    scrub.mkdir(parents=True, exist_ok=True)
    for old in scrub.glob("*.jpg"):
        old.unlink()
    n = int(frames_rgb.shape[0])
    for i in range(n):
        fr = frames_rgb[i]
        h, w = fr.shape[:2]
        tw = 480 if w >= 480 else w
        th = max(2, int(round(h * (tw / max(1, w)))))
        small = cv2.resize(fr, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.imwrite(
            str(scrub / f"{i:05d}.jpg"),
            cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 80],
        )
    return n


def _scrub_count(job: Path) -> int:
    scrub = job / "cut_frames" / "scrub"
    if not scrub.is_dir():
        return 0
    return sum(1 for p in scrub.glob("*.jpg"))


def ensure_scene_scrub(job_name: str) -> dict:
    """Build slider previews from driver.mp4 when Detect has not written them yet."""
    import subprocess

    job = _job_root(job_name)
    meta_path = job / "job.json"
    if not meta_path.exists():
        raise FileNotFoundError("Нет job.json — сначала Queue на Detect.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    n = int(meta.get("frames") or 0)
    driver = job / "driver.mp4"
    if not driver.exists():
        raise FileNotFoundError("Нет driver.mp4 — сначала Queue на Detect.")
    if n > 0 and _scrub_count(job) == n and (job / "cut_frames" / "scrub" / "00000.jpg").exists():
        return {"ready": True, "frames": n, "job": job_name}
    scrub = job / "cut_frames" / "scrub"
    scrub.mkdir(parents=True, exist_ok=True)
    for old in scrub.glob("*.jpg"):
        old.unlink()
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(driver),
            "-vf", "scale=480:-2",
            "-start_number", "0",
            "-q:v", "4",
            str(scrub / "%05d.jpg"),
        ],
        check=True,
    )
    got = _scrub_count(job)
    if n and got != n:
        raise RuntimeError(f"Превью кадров: {got}, в ролике {n}. Запусти Detect ещё раз.")
    return {"ready": True, "frames": got, "job": job_name}


def list_scene_ranges(job_name: str) -> dict:
    job = _job_root(job_name)
    cuts_path = job / "cuts.json"
    if not cuts_path.exists():
        raise FileNotFoundError("Нет cuts.json — сначала Queue на Detect.")
    cuts = json.loads(cuts_path.read_text(encoding="utf-8"))
    rows = []
    for i, s in enumerate(cuts.get("segments") or []):
        start = int(s["start"])
        end = int(s["end"])
        ref = int(s.get("ref_frame", start))
        if ref < start or ref >= end:
            ref = start
        rows.append(
            {
                "seg_index": i,
                "seg_id": s["id"],
                "start": start,
                "end": end,
                "n_frames": max(0, end - start),
                "ref_frame": ref,
                "ref_offset": ref - start,
                "saved": (job / "cut_frames" / f"{s['id']}_REF.png").exists(),
            }
        )
    return {"job": job_name, "segments": rows, "scrub_ready": _scrub_count(job) > 0}


def scene_scrub_file(job_name: str, seg_index: int, offset: int) -> Path:
    job = _job_root(job_name)
    cuts = json.loads((job / "cuts.json").read_text(encoding="utf-8"))
    seg = (cuts.get("segments") or [])[int(seg_index)]
    start = int(seg["start"])
    end = int(seg["end"])
    offset = max(0, min(int(offset), max(0, end - start - 1)))
    path = job / "cut_frames" / "scrub" / f"{start + offset:05d}.jpg"
    if not path.exists():
        raise FileNotFoundError(f"Нет превью кадра {start + offset}")
    return path


def _extract_frame_bgr(path: Path, index: int):
    import subprocess

    import cv2

    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error",
            "-i", str(path),
            "-vf", f"select=eq(n\\,{int(index)})",
            "-frames:v", "1",
            "-f", "image2pipe", "-vcodec", "png", "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        err = (proc.stderr or b"").decode("utf-8", "replace")[-400:]
        raise RuntimeError(f"Не прочитал кадр {index}: {err}")
    bgr = cv2.imdecode(np.frombuffer(proc.stdout, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Не прочитал кадр {index}")
    return bgr


def save_scene_ref_frame(job_name: str, seg_index: int, offset: int) -> dict:
    """Remember one frame inside a cut scene. Scene bounds stay put."""
    import cv2

    job = _job_root(job_name)
    cuts_path = job / "cuts.json"
    cuts = json.loads(cuts_path.read_text(encoding="utf-8"))
    segs = cuts.get("segments") or []
    seg = segs[int(seg_index)]
    start = int(seg["start"])
    end = int(seg["end"])
    n = max(1, end - start)
    offset = max(0, min(int(offset), n - 1))
    ref = start + offset
    bgr = _extract_frame_bgr(job / "driver.mp4", ref)
    out_dir = job / "cut_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_name = f"{seg['id']}_REF.png"
    plain = f"{seg['id']}_f{ref:05d}_PLAIN.png"
    cv2.imwrite(str(out_dir / ref_name), bgr)
    cv2.imwrite(str(out_dir / plain), bgr)
    seg["ref_frame"] = ref
    seg["cut_frame"] = f"cut_frames/{plain}"
    cuts_path.write_text(json.dumps(cuts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rel = f"input/cut_video_face_swapper/jobs/{job_name}/cut_frames/{ref_name}"
    return {
        "seg_id": seg["id"],
        "seg_index": int(seg_index),
        "ref_frame": ref,
        "offset": offset,
        "n_frames": n,
        "start": start,
        "end": end,
        "file": rel,
    }


def _bake_job_media(
    job_name: str,
    frames_rgb: np.ndarray,
    masks_rgb: np.ndarray | None = None,
    *,
    source_driver: str = "",
    source_mask: str = "",
    segments: list[dict] | None = None,
    fps: float | None = None,
) -> tuple[str, str]:
    """Save whatever frames came from Load Video (any filename) into the job folder."""
    import cv2

    job = _job_root(job_name)
    job.mkdir(parents=True, exist_ok=True)
    (job / "masters").mkdir(exist_ok=True)
    (job / "parts").mkdir(exist_ok=True)
    (job / "workflows").mkdir(exist_ok=True)
    prev_job = {}
    if (job / "job.json").exists():
        try:
            prev_job = json.loads((job / "job.json").read_text(encoding="utf-8"))
        except Exception:
            prev_job = {}
    fps = float(fps if fps is not None else prev_job.get("fps") or 24.0)
    h, w = int(frames_rgb.shape[1]), int(frames_rgb.shape[2])
    n = int(frames_rgb.shape[0])
    driver_rel = f"cut_video_face_swapper/jobs/{job_name}/driver.mp4"
    mask_rel = f"cut_video_face_swapper/jobs/{job_name}/mask.mp4"
    _write_bgr_mp4([cv2.cvtColor(fr, cv2.COLOR_RGB2BGR) for fr in frames_rgb], job / "driver.mp4", fps)
    if masks_rgb is not None:
        if masks_rgb.shape[0] != n:
            if masks_rgb.shape[0] < n:
                pad = np.repeat(masks_rgb[-1:], n - masks_rgb.shape[0], axis=0)
                masks_rgb = np.concatenate([masks_rgb, pad], axis=0)
            else:
                masks_rgb = masks_rgb[:n]
        mask_bgr = [cv2.cvtColor(fr, cv2.COLOR_RGB2BGR) for fr in masks_rgb]
    else:
        mask_bgr = [np.zeros((h, w, 3), np.uint8) for _ in range(n)]
    _write_bgr_mp4(mask_bgr, job / "mask.mp4", fps)

    cuts_path = job / "cuts.json"
    cuts = {}
    if cuts_path.exists():
        try:
            cuts = json.loads(cuts_path.read_text(encoding="utf-8"))
        except Exception:
            cuts = {}
    if segments is not None:
        cuts["segments"] = segments
    cuts.update(
        {
            "job": job_name,
            "howto": [
                "Load Video (любое имя) → Detect или Masters. Имена файлов руками не нужны.",
                "Внутри job лежат driver.mp4/mask.mp4 — копия того, что было на проводах.",
            ],
            "driver_filename": driver_rel,
            "mask_filename": mask_rel,
            "source_driver": source_driver,
            "source_mask": source_mask,
        }
    )
    if "segments" not in cuts:
        cuts["segments"] = []
    cuts_path.write_text(json.dumps(cuts, indent=2, ensure_ascii=False), encoding="utf-8")
    prev_job.update(
        {
            "name": job_name,
            "comfy_driver": driver_rel,
            "comfy_mask_rect": mask_rel,
            "source_driver": source_driver,
            "source_mask": source_mask,
            "frames": n,
            "width": w,
            "height": h,
            "fps": fps,
        }
    )
    (job / "job.json").write_text(json.dumps(prev_job, indent=2), encoding="utf-8")
    return driver_rel, mask_rel


def _ensure_job_ready_from_frames(
    job_name: str,
    video_frames,
    mask_frames=None,
    *,
    min_scene_frames: int = 8,
    camera_hist_cut: float = 0.85,
    mask_jump_center: float = 0.12,
    mask_jump_scale: float = 0.35,
) -> None:
    """Bake Load Video frames into the job; re-detect cuts if missing or length mismatch."""
    frames = _tensor_batch_to_uint8(video_frames)
    masks = _tensor_batch_to_uint8(mask_frames) if mask_frames is not None else None
    job = _job_root(job_name)
    need_detect = True
    cuts_path = job / "cuts.json"
    if cuts_path.exists():
        try:
            cuts = json.loads(cuts_path.read_text(encoding="utf-8"))
            segs = cuts.get("segments") or []
            end = max((int(s.get("end") or 0) for s in segs), default=0)
            if segs and end <= int(frames.shape[0]):
                need_detect = False
                _bake_job_media(job_name, frames, masks, segments=segs)
        except Exception:
            need_detect = True
    if need_detect:
        segs = detect_segment_starts(
            frames,
            masks,
            min_len=int(min_scene_frames),
            hist_corr_cut=float(camera_hist_cut),
            jump_center=float(mask_jump_center),
            jump_scale=float(mask_jump_scale),
            use_camera=True,
            use_mask=masks is not None,
        )
        old_by_id = {}
        if cuts_path.exists():
            try:
                old_by_id = {
                    s["id"]: s
                    for s in json.loads(cuts_path.read_text(encoding="utf-8")).get("segments", [])
                }
            except Exception:
                old_by_id = {}
        for s in segs:
            s["master"] = f"masters/{s['id']}.png"
            s["status"] = "detected"
            if s["id"] in old_by_id:
                prev = old_by_id[s["id"]]
                for k in ("master_source", "master"):
                    if prev.get(k):
                        s[k] = prev[k]
                if (job / "masters" / f"{s['id']}.png").exists():
                    s["status"] = "master_bound"
        _bake_job_media(job_name, frames, masks, segments=segs)


def _pack_dir(masters_folder: str) -> Path:
    root = Path(folder_paths.get_input_directory()).resolve()
    folder = Path(masters_folder)
    if not folder.is_absolute():
        folder = root / folder
    folder = folder.resolve()
    if folder != root and root not in folder.parents:
        raise ValueError("masters_folder must stay inside ComfyUI input/")
    return folder


_PACK_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def list_master_pack(masters_folder: str) -> dict:
    folder = _pack_dir(masters_folder)
    if not folder.is_dir():
        raise FileNotFoundError(str(folder))
    files = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in _PACK_EXTS],
        key=lambda p: p.name.lower(),
    )
    root = Path(folder_paths.get_input_directory()).resolve()
    sub = folder.relative_to(root).as_posix()
    return {
        "folder": sub,
        "files": [{"name": p.name, "seg": f"{i:02d}"} for i, p in enumerate(files)],
    }


def reorder_master_pack(masters_folder: str, names: list[str], job_name: str = "") -> dict:
    """Rename pack files to 000_, 001_, … and refresh job masters/XX.png (no labels)."""
    import re
    import cv2

    folder = _pack_dir(masters_folder)
    existing = {p.name: p for p in folder.iterdir() if p.suffix.lower() in _PACK_EXTS}
    if not names or set(names) != set(existing):
        raise ValueError("Передай каждый файл пачки ровно один раз")
    temps = []
    for i, name in enumerate(names):
        src = existing[name]
        tmp = folder / f".reorder_{i:03d}{src.suffix.lower()}"
        if tmp.exists():
            tmp.unlink()
        src.rename(tmp)
        temps.append((tmp, name))
    final_names = []
    for i, (tmp, orig) in enumerate(temps):
        rest = re.sub(r"^\d{3}_", "", orig)
        dest = folder / f"{i:03d}_{rest}"
        if dest.exists():
            dest.unlink()
        tmp.rename(dest)
        final_names.append(dest.name)
    copied = []
    if job_name.strip():
        masters = _job_root(job_name.strip()) / "masters"
        masters.mkdir(parents=True, exist_ok=True)
        for i, name in enumerate(final_names):
            im = cv2.imread(str(folder / name), cv2.IMREAD_COLOR)
            if im is None:
                continue
            out = masters / f"{i:02d}.png"
            cv2.imwrite(str(out), im)
            copied.append(out.name)
    return {"folder": masters_folder, "order": final_names, "masters_written": copied}


def _wan_wh(w: int, h: int) -> tuple[int, int]:
    """Wan Animate floors to a multiple of 16."""
    return max(64, int(w) // 16 * 16), max(64, int(h) // 16 * 16)


def _wan_nframes(n: int) -> int:
    """Wan Animate floors length to 4n+1."""
    n = max(1, int(n))
    return ((n - 1) // 4) * 4 + 1


def _clip_seg_for_test(seg: dict, fps: float, seconds: float) -> dict:
    """A few seconds around the picked master frame. Does not change the real scene."""
    fps = float(fps or 24)
    want = max(1, int(round(float(seconds) * fps)))
    # Sub-frame / tiny seconds → single frame quality check (Wan accepts 4n+1, and 1 is valid).
    if float(seconds) <= 0.05 or want <= 1:
        n = 1
    else:
        n = _vace_nframes(max(17, want))
    start = int(seg["start"])
    end = int(seg["end"])
    ref = int(seg.get("ref_frame", start))
    ref = min(max(ref, start), max(start, end - 1))
    win_start = ref - n // 2
    win_start = max(start, min(win_start, max(start, end - n)))
    win_end = min(end, win_start + n)
    out = dict(seg)
    out["start"] = int(win_start)
    out["end"] = int(win_end)
    out["n_frames"] = int(win_end - win_start)
    out["test_preview"] = True
    return out


def _snap8(v: float, mx: int) -> int:
    return max(0, min(mx, (int(round(v)) // 8) * 8))


# v1.4 working window — Wan runs here, not full-plate 1080p
WAN_CROP_W = 832
WAN_CROP_H = 480
# Side hair sits outside the face matte. 0.55 of the matte width, each side.
FACE_PAD_X_RATIO = 0.55
# A face column is tall. Below this width/height the 832 fit becomes a slit.
FACE_MIN_WINDOW_ASPECT = 0.62
# Face-mode defaults. Mimicry at 1 — eyelid/mouth follow the driver.
# Relight/colormatch still eased so skin stays nearer the master.
FACE_STRENGTH_DEFAULT = 1.0
FACE_RELIGHT_DEFAULT = 0.5
FACE_COLORMATCH_DEFAULT = "disabled"
# Option A test: ReActor-swap master identity onto driver face crops before Wan.
FACE_PREP_REACTOR_MODEL = "inswapper_128.onnx"
# Separate fast profile (Af): smaller window + lighter detector. Swap stays VRAM-safe.
FAST_WAN_MAX_SIDE = 640
FAST_REACTOR_DETECT = "retinaface_mobile0.25"
# RTX 5080 16GB + Wan2.2 Animate 14B fp8. 768²/swap20 OK; 640²/swap10 CUDA-crashed at sample.
FACE_SAFE_MAX_SIDE = 768
FACE_SAFE_MAX_PIXELS = 768 * 768
FACE_BLOCKS_MIN = 20
FACE_BLOCKS_MAX = 28
FACE_PREFETCH_SAFE = 1


def _is_reactor_prep(val: str | None) -> bool:
    return str(val or "").strip().lower() in {"reactor", "a", "swap", "reactor_fast", "af", "fast_a"}


def _is_fast_profile(seg: dict | None) -> bool:
    s = seg or {}
    if str(s.get("speed_profile") or "").strip().lower() == "fast":
        return True
    return str(s.get("face_images_prep") or "").strip().lower() in {"reactor_fast", "af", "fast_a"}


def _face_block_swap_for_size(width: int, height: int, n_frames: int = 49) -> tuple[int, int]:
    """blocks_to_swap + prefetch so 14B Animate stays inside ~16GB for any clip size.

    More pixels / longer clips → more blocks on CPU. Never go below FACE_BLOCKS_MIN:
    under-swapping was what blew Af (illegal memory access at Sampling start).
    """
    px = max(1, int(width) * int(height))
    blocks = FACE_BLOCKS_MIN
    if px > 520_000:
        blocks += 2
    if px > 650_000:
        blocks += 2
    if int(n_frames) >= 81:
        blocks += 2
    if int(n_frames) >= 161:
        blocks += 2
    blocks = max(FACE_BLOCKS_MIN, min(FACE_BLOCKS_MAX, blocks))
    return blocks, FACE_PREFETCH_SAFE


def _apply_face_vram_guard(wf: dict, width: int, height: int, n_frames: int, seg: dict | None = None) -> None:
    """Force BlockSwap onto a known-safe budget for face mode (any video)."""
    if "15" not in wf:
        return
    blocks, prefetch = _face_block_swap_for_size(width, height, n_frames)
    if seg is not None and seg.get("blocks_to_swap") is not None:
        blocks = max(blocks, int(seg["blocks_to_swap"]))
    if seg is not None and seg.get("prefetch_blocks") is not None:
        prefetch = int(seg["prefetch_blocks"])
    wf["15"]["inputs"]["blocks_to_swap"] = int(blocks)
    wf["15"]["inputs"]["prefetch_blocks"] = int(prefetch)
    wf["15"]["inputs"]["use_non_blocking"] = True
    if seg is not None:
        seg["blocks_to_swap"] = int(blocks)
        seg["prefetch_blocks"] = int(prefetch)


def _input_video(path_name: str) -> Path:
    p = Path(path_name)
    if p.is_absolute():
        return p
    return Path(folder_paths.get_input_directory()) / p


def _prefer_job_media(job_name: str, cuts: dict | None = None, cfg: dict | None = None) -> tuple[str, str]:
    """Only job/driver.mp4 + job/mask.mp4. Never trust leftover filename strings from an old video."""
    del cuts, cfg
    job = _job_root(job_name)
    local_driver = job / "driver.mp4"
    local_mask = job / "mask.mp4"
    if local_driver.exists() and local_mask.exists() and local_driver.stat().st_size > 0 and local_mask.stat().st_size > 0:
        return (
            f"cut_video_face_swapper/jobs/{job_name}/driver.mp4",
            f"cut_video_face_swapper/jobs/{job_name}/mask.mp4",
        )
    raise ValueError(
        f"В job «{job_name}» ещё нет копии видео.\n"
        "В Load Video выбери оригинал и маску (имя файла любое),\n"
        "проведи провода в Detect или в ноду Masters, и нажми Queue.\n"
        "Писать имена файлов руками не нужно."
    )


def _union_matte_bbox(
    matte_frames: list[np.ndarray], pad_x: int = 64, pad_y: int = 48, thr: int = 16
) -> tuple[int, int, int, int] | None:
    """Union AABB of matte pixels across frames → (x0,y0,x1,y1), or None if empty."""
    xs_all: list[int] = []
    ys_all: list[int] = []
    for fr in matte_frames:
        g = fr[..., 0] if fr.ndim == 3 else fr
        ys, xs = np.where(g > thr)
        if len(xs):
            xs_all.extend([int(xs.min()), int(xs.max())])
            ys_all.extend([int(ys.min()), int(ys.max())])
    if not xs_all:
        return None
    h, w = (matte_frames[0].shape[0], matte_frames[0].shape[1])
    x0 = max(0, min(xs_all) - pad_x)
    y0 = max(0, min(ys_all) - pad_y)
    x1 = min(w, max(xs_all) + pad_x + 1)
    y1 = min(h, max(ys_all) + pad_y + 1)
    return x0, y0, x1, y1


def _center_wan_window(plate_w: int, plate_h: int, tw: int = WAN_CROP_W, th: int = WAN_CROP_H) -> tuple[int, int, int, int]:
    tw = min(tw, (plate_w // 8) * 8)
    th = min(th, (plate_h // 8) * 8)
    x0 = max(0, (plate_w - tw) // 2)
    y0 = max(0, (plate_h - th) // 2)
    return x0, y0, x0 + tw, y0 + th


def _expand_bbox_percent(
    x0: int, y0: int, x1: int, y1: int, plate_w: int, plate_h: int, pad_percent: float
) -> tuple[int, int, int, int]:
    """Pad each side by pad_percent of the bbox size. 20 → +20% width left and right."""
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    px = bw * (float(pad_percent) / 100.0)
    py = bh * (float(pad_percent) / 100.0)
    nx0 = max(0, int(np.floor(x0 - px)))
    ny0 = max(0, int(np.floor(y0 - py)))
    nx1 = min(plate_w, int(np.ceil(x1 + px)))
    ny1 = min(plate_h, int(np.ceil(y1 + py)))
    if nx1 - nx0 < 16:
        nx1 = min(plate_w, nx0 + 16)
    if ny1 - ny0 < 16:
        ny1 = min(plate_h, ny0 + 16)
    return nx0, ny0, nx1, ny1


def _ensure_min_crop_width(
    x0: int, y0: int, x1: int, y1: int, plate_w: int, plate_h: int, min_aspect: float
) -> tuple[int, int, int, int]:
    """Grow a tall face window sideways so the 832 fit does not collapse into a slit.

    The matte union over a few seconds is a column: the head drops, it barely
    moves left/right. Fitting the long side to 832 then leaves no pixels for hair.
    """
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    need = int(np.ceil(bh * float(min_aspect)))
    need = min(need, max(16, plate_w))
    if bw >= need:
        return x0, y0, x1, y1
    cx = 0.5 * (x0 + x1)
    nx0 = int(np.floor(cx - need / 2.0))
    nx1 = nx0 + need
    if nx0 < 0:
        nx1 -= nx0
        nx0 = 0
    if nx1 > plate_w:
        nx0 -= nx1 - plate_w
        nx1 = plate_w
    nx0 = max(0, nx0)
    if nx1 - nx0 < 16:
        nx1 = min(plate_w, nx0 + 16)
    return nx0, y0, nx1, y1


def _fill_none_boxes(boxes: list) -> list:
    last = None
    out = list(boxes)
    for i, b in enumerate(out):
        if b is None and last is not None:
            out[i] = last
        elif b is not None:
            last = b
    for i in range(len(out)):
        if out[i] is None:
            for j in range(i + 1, len(out)):
                if out[j] is not None:
                    out[i] = out[j]
                    break
    return out


def _smooth_centers(centers: list[tuple[float, float]], rad: int = 4) -> list[tuple[float, float]]:
    n = len(centers)
    out = []
    for i in range(n):
        a = max(0, i - rad)
        b = min(n, i + rad + 1)
        xs = [centers[k][0] for k in range(a, b)]
        ys = [centers[k][1] for k in range(a, b)]
        out.append((sum(xs) / len(xs), sum(ys) / len(ys)))
    return out


def _head_track_windows(
    mattes: list[np.ndarray], plate_w: int, plate_h: int, ref_index: int
) -> list[list[int]] | None:
    """One fixed window that follows the head, so the face stays large in every frame."""
    raw = []
    for fr in mattes:
        g = fr[..., 0] if fr.ndim == 3 else fr
        ys, xs = np.where(g > 16)
        if len(xs) == 0:
            raw.append(None)
        else:
            raw.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    if not any(b is not None for b in raw):
        return None
    boxes = _fill_none_boxes(raw)
    expanded = []
    for x0, y0, x1, y1 in boxes:
        bw = max(1, x1 - x0)
        bh = max(1, y1 - y0)
        px = bw * FACE_PAD_X_RATIO
        py = bh * 0.12
        expanded.append(
            (
                max(0, int(np.floor(x0 - px))),
                max(0, int(np.floor(y0 - py))),
                min(plate_w, int(np.ceil(x1 + px))),
                min(plate_h, int(np.ceil(y1 + py))),
            )
        )
    widths = [b[2] - b[0] for b in expanded]
    heights = [b[3] - b[1] for b in expanded]
    ww = int(np.median(widths))
    wh = int(np.median(heights))
    ri = min(max(0, int(ref_index)), len(expanded) - 1)
    ww = max(ww, expanded[ri][2] - expanded[ri][0])
    wh = max(wh, expanded[ri][3] - expanded[ri][1])
    # Wide enough that the 512 ref, stretched to this window, still lands on the head.
    if ww < wh * 0.85:
        ww = int(np.ceil(wh * 0.85))
    ww = max(16, min(ww, plate_w))
    wh = max(16, min(wh, plate_h))
    centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes]
    centers = _smooth_centers(centers, rad=4)
    placed = []
    for cx, cy in centers:
        x0 = int(round(cx - ww / 2.0))
        y0 = int(round(cy - wh / 2.0))
        x0 = max(0, min(x0, plate_w - ww))
        y0 = max(0, min(y0, plate_h - wh))
        placed.append([x0, y0, x0 + ww, y0 + wh])
    return placed


def _wan_fit_size(cw: int, ch: int, max_side: int = 832, max_pixels: int | None = None) -> tuple[int, int]:
    """Scale the mask window so its long side is 832. Aspect kept, both axes ÷16.

    Optional max_pixels (e.g. 832*480) keeps tall OTS crops from exceeding the
    sequence length of the known-good landscape face path — portrait 576×832 was
    wedging the sampler at the first step on RTX 5080.
    """
    long_side = max(1, cw, ch)
    scale = max_side / float(long_side)
    tw = max(64, int(round(cw * scale / 16.0)) * 16)
    th = max(64, int(round(ch * scale / 16.0)) * 16)
    if max_pixels is not None and tw * th > int(max_pixels):
        shrink = (float(max_pixels) / float(tw * th)) ** 0.5
        tw = max(64, int(round(tw * shrink / 16.0)) * 16)
        th = max(64, int(round(th * shrink / 16.0)) * 16)
        while tw * th > int(max_pixels) and max(tw, th) > 64:
            if tw >= th:
                tw = max(64, tw - 16)
            else:
                th = max(64, th - 16)
    return tw, th


def _read_video_slice(path: Path, start: int, count: int, gray: bool = False) -> list[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))
    out = []
    for _ in range(count):
        ok, fr = cap.read()
        if not ok:
            break
        if gray:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr
            out.append(g)
        else:
            out.append(fr)
    cap.release()
    if not out:
        total = int(cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        raise ValueError(
            f"Нет кадров из {path} start={start} count={count} (в файле всего {total}). "
            "Обычно это старое видео в job — снова запусти Detect на актуальном ролике."
        )
    return out


def _master_align_shift(seg: dict | None) -> tuple[float, float]:
    """Plate-pixel shift of the still vs the driver. Negative moves the crop up-left."""
    s = seg or {}
    return float(s.get("master_align_dx") or 0), float(s.get("master_align_dy") or 0)


def _crop_master_to_plate_window(
    im: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    plate_w: int,
    plate_h: int,
    dx: float = 0.0,
    dy: float = 0.0,
) -> np.ndarray:
    """Map a plate window onto a master of any size. dx/dy are plate pixels."""
    import cv2

    mh, mw = im.shape[:2]
    sx0 = (float(x0) + dx) * mw / float(plate_w)
    sy0 = (float(y0) + dy) * mh / float(plate_h)
    sx1 = (float(x1) + dx) * mw / float(plate_w)
    sy1 = (float(y1) + dy) * mh / float(plate_h)
    tw = max(1.0, sx1 - sx0)
    th = max(1.0, sy1 - sy0)
    ix0 = int(np.floor(sx0))
    iy0 = int(np.floor(sy0))
    ix1 = int(np.ceil(sx0 + tw))
    iy1 = int(np.ceil(sy0 + th))
    pad_l, pad_t = max(0, -ix0), max(0, -iy0)
    pad_r, pad_b = max(0, ix1 - mw), max(0, iy1 - mh)
    crop = im[max(0, iy0) : min(mh, iy1), max(0, ix0) : min(mw, ix1)]
    if crop.size == 0:
        crop = np.zeros((max(1, iy1 - iy0), max(1, ix1 - ix0), im.shape[2]), np.uint8)
    if pad_l or pad_t or pad_r or pad_b:
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
    return crop


def _prepare_seg_wan_crop(
    job: Path,
    job_name: str,
    seg: dict,
    driver_name: str,
    mask_name: str,
    *,
    tw: int = WAN_CROP_W,
    th: int = WAN_CROP_H,
    fps: float = 24.0,
    pad_percent: float = 20.0,
    fallback_crop: list[int] | None = None,
) -> dict:
    """v1.4: crop driver+matte+master to Wan window around face/hair; write segs/XX/*.mp4."""
    import cv2

    sid = seg["id"]
    bundle = f"{sid}_test" if seg.get("test_preview") else sid
    start = int(seg["start"])
    raw_n = int(seg["end"] - seg["start"])
    n = _wan_nframes(raw_n)
    driver_p = _input_video(driver_name)
    mask_p = _input_video(mask_name)
    mattes = _read_video_slice(mask_p, start, n, gray=True)
    drivers = _read_video_slice(driver_p, start, n, gray=False)
    n = min(n, len(mattes), len(drivers))
    mattes, drivers = mattes[:n], drivers[:n]
    plate_h, plate_w = mattes[0].shape[:2]
    ref_abs = int(seg.get("ref_frame", start))
    ref_index = min(max(0, ref_abs - start), max(0, n - 1))
    tracks = _head_track_windows(mattes, plate_w, plate_h, ref_index)
    if tracks is None:
        # No face/hair on this cut — keep original frames at assemble. Do not borrow another shot's window.
        return {
            "seg_id": sid,
            "passthrough": True,
            "matte_empty": True,
            "crop": None,
            "crop_note": "passthrough_original",
            "plate": [plate_w, plate_h],
            "window_src": [0, 0],
            "width": tw,
            "height": th,
            "n_frames": n,
            "raw_n_frames": raw_n,
            "driver": driver_name,
            "mask": mask_name,
            "master_image": f"cut_video_face_swapper/jobs/{job_name}/masters/{sid}.png",
            "master_exists": (job / "masters" / f"{sid}.png").exists(),
            "skip_first_frames": start,
            "frame_load_cap": n,
        }
    # Representative box is the chosen frame. Every frame uses its own tracked window.
    x0, y0, x1, y1 = tracks[ref_index]
    cw, ch = x1 - x0, y1 - y0
    if _is_fast_profile(seg):
        # Fast test: smaller window only. Block-swap budget is applied later and stays safe.
        wan_w, wan_h = _wan_fit_size(
            cw, ch, max_side=FAST_WAN_MAX_SIDE, max_pixels=FAST_WAN_MAX_SIDE * FAST_WAN_MAX_SIDE
        )
    else:
        # Hard cap so tall head-tracks cannot inflate past what 16GB + swap20 survived.
        wan_w, wan_h = _wan_fit_size(
            cw, ch, max_side=FACE_SAFE_MAX_SIDE, max_pixels=FACE_SAFE_MAX_PIXELS
        )
    tracks = [_fit_paste_crop(box, wan_w, wan_h, plate_w, plate_h) for box in tracks]
    x0, y0, x1, y1 = tracks[ref_index]
    cw, ch = x1 - x0, y1 - y0
    crop_note = "head_track"

    out_dir = job / "segs" / bundle
    out_dir.mkdir(parents=True, exist_ok=True)
    driver_out = out_dir / "driver.mp4"
    mask_out = out_dir / "mask.mp4"
    master_out = out_dir / "master.png"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wr_d = cv2.VideoWriter(str(driver_out), fourcc, fps, (wan_w, wan_h), True)
    wr_m = cv2.VideoWriter(str(mask_out), fourcc, fps, (wan_w, wan_h), True)
    for fr, m, box in zip(drivers, mattes, tracks):
        bx0, by0, bx1, by1 = box
        dc = cv2.resize(fr[by0:by1, bx0:bx1], (wan_w, wan_h), interpolation=cv2.INTER_AREA)
        mc = cv2.resize(m[by0:by1, bx0:bx1], (wan_w, wan_h), interpolation=cv2.INTER_NEAREST)
        if mc.ndim == 2:
            mc3 = cv2.cvtColor(mc, cv2.COLOR_GRAY2BGR)
        else:
            mc3 = mc
        wr_d.write(dc)
        wr_m.write(mc3)
    wr_d.release()
    wr_m.release()

    master_src = job / "masters" / f"{sid}.png"
    if master_src.exists():
        im = cv2.imread(str(master_src), cv2.IMREAD_COLOR)
        if im is not None:
            dx, dy = _master_align_shift(seg)
            mcrop = _crop_master_to_plate_window(im, x0, y0, x1, y1, plate_w, plate_h, dx, dy)
            mcrop = cv2.resize(mcrop, (wan_w, wan_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(master_out), mcrop)
            if dx or dy:
                crop_note = f"{crop_note} master_shift={dx:.1f},{dy:.1f}"
    (out_dir / "track.json").write_text(json.dumps(tracks), encoding="utf-8")

    rel = f"cut_video_face_swapper/jobs/{job_name}/segs/{bundle}"
    return {
        "seg_id": sid,
        "crop": [x0, y0, x1, y1],
        "track": tracks,
        "crop_note": crop_note,
        "plate": [plate_w, plate_h],
        "window_src": [cw, ch],
        "width": wan_w,
        "height": wan_h,
        "pad_percent": float(pad_percent),
        "n_frames": n,
        "raw_n_frames": raw_n,
        "driver": f"{rel}/driver.mp4",
        "mask": f"{rel}/mask.mp4",
        "master_image": f"{rel}/master.png",
        "master_exists": master_out.exists(),
        "skip_first_frames": 0,
        "frame_load_cap": n,
        "matte_empty": False,
        "passthrough": False,
    }


def _resolve_wan_mode(seg: dict | None) -> str:
    """face = v1.4 Wan Animate; vace = no-face inpaint (seg02_full); hair_back/face_gate kept."""
    mode = str((seg or {}).get("wan_mode") or "face").strip().lower()
    if mode in {"vace", "inpaint", "noface", "no_face", "back_vace"}:
        return "vace"
    if mode in {"hair", "hair_back", "ots", "back", "occiput"}:
        return "hair_back"
    if mode in {"face_gate", "adaptive", "gate", "auto"}:
        return "face_gate"
    return "face"


def _vace_nframes(n: int) -> int:
    """Smallest 4n+1 that covers n. VACE drops a tail if the count is short."""
    n = max(1, int(n))
    if (n - 1) % 4 == 0:
        return n
    return ((n - 1) // 4 + 1) * 4 + 1


def _haar_leftish_face_rights(drivers: list, plate_w: int) -> list[int]:
    """Right edges of faces whose center is in the left ~72% (foreground OTS partner stays right)."""
    import cv2

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    rights: list[int] = []
    if cascade.empty():
        return rights
    for fr in drivers:
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr
        faces = cascade.detectMultiScale(gray, 1.08, 4, minSize=(60, 60))
        best = None
        for x, y, w, h in faces:
            if (x + w / 2.0) > plate_w * 0.72:
                continue
            area = int(w * h)
            if best is None or area > best[0]:
                best = (area, int(x + w))
        if best:
            rights.append(best[1])
    return rights


def _write_seg_crop_bundle(
    job: Path,
    job_name: str,
    sid: str,
    drivers: list,
    mattes: list,
    *,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    wan_w: int,
    wan_h: int,
    fps: float,
    crop_note: str,
    pad_percent: float,
    raw_n: int,
    master_dx: float = 0.0,
    master_dy: float = 0.0,
) -> dict:
    """Shared writer for segs/<sid>/{driver,mask,master} after a plate crop is chosen."""
    import cv2

    n = min(len(drivers), len(mattes))
    drivers, mattes = drivers[:n], mattes[:n]
    plate_h, plate_w = drivers[0].shape[:2]
    out_dir = job / "segs" / sid
    out_dir.mkdir(parents=True, exist_ok=True)
    driver_out = out_dir / "driver.mp4"
    mask_out = out_dir / "mask.mp4"
    master_out = out_dir / "master.png"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wr_d = cv2.VideoWriter(str(driver_out), fourcc, fps, (wan_w, wan_h), True)
    wr_m = cv2.VideoWriter(str(mask_out), fourcc, fps, (wan_w, wan_h), True)
    for fr, m in zip(drivers, mattes):
        dc = cv2.resize(fr[y0:y1, x0:x1], (wan_w, wan_h), interpolation=cv2.INTER_AREA)
        mc = cv2.resize(m[y0:y1, x0:x1], (wan_w, wan_h), interpolation=cv2.INTER_NEAREST)
        if mc.ndim == 2:
            mc3 = cv2.cvtColor(mc, cv2.COLOR_GRAY2BGR)
        else:
            mc3 = mc
        wr_d.write(dc)
        wr_m.write(mc3)
    wr_d.release()
    wr_m.release()

    master_src = job / "masters" / f"{sid}.png"
    if master_src.exists():
        im = cv2.imread(str(master_src), cv2.IMREAD_COLOR)
        if im is not None:
            mcrop = _crop_master_to_plate_window(
                im, x0, y0, x1, y1, plate_w, plate_h, master_dx, master_dy
            )
            cv2.imwrite(
                str(master_out),
                cv2.resize(mcrop, (wan_w, wan_h), interpolation=cv2.INTER_AREA),
            )

    rel = f"cut_video_face_swapper/jobs/{job_name}/segs/{sid}"
    return {
        "seg_id": sid,
        "crop": [int(x0), int(y0), int(x1), int(y1)],
        "crop_note": crop_note,
        "plate": [plate_w, plate_h],
        "window_src": [int(x1 - x0), int(y1 - y0)],
        "width": int(wan_w),
        "height": int(wan_h),
        "pad_percent": float(pad_percent),
        "n_frames": n,
        "raw_n_frames": raw_n,
        "driver": f"{rel}/driver.mp4",
        "mask": f"{rel}/mask.mp4",
        "master_image": f"{rel}/master.png",
        "master_exists": master_out.exists(),
        "skip_first_frames": 0,
        "frame_load_cap": n,
        "matte_empty": False,
        "passthrough": False,
    }


def _prepare_seg_wan_crop_hair(
    job: Path,
    job_name: str,
    seg: dict,
    driver_name: str,
    mask_name: str,
    *,
    tw: int = WAN_CROP_W,
    th: int = WAN_CROP_H,
    fps: float = 24.0,
    pad_percent: float = 20.0,
    face_gap: int = 40,
) -> dict:
    """Hair/OTS submodule: FaceHair window with left edge clamped past any leftish face."""
    import cv2

    sid = seg["id"]
    start = int(seg["start"])
    raw_n = int(seg["end"] - seg["start"])
    n = _wan_nframes(raw_n)
    mattes = _read_video_slice(_input_video(mask_name), start, n, gray=True)
    drivers = _read_video_slice(_input_video(driver_name), start, n, gray=False)
    n = min(n, len(mattes), len(drivers))
    mattes, drivers = mattes[:n], drivers[:n]
    plate_h, plate_w = mattes[0].shape[:2]

    rights = _haar_leftish_face_rights(drivers, plate_w)
    clamp_x = 0
    if rights:
        clamp_x = min(plate_w - 64, int(np.median(rights)) + int(face_gap))

    mattes_c = []
    for m in mattes:
        mm = m.copy()
        if clamp_x > 0:
            mm[:, :clamp_x] = 0
        mattes_c.append(mm)

    bbox = _union_matte_bbox(mattes_c, pad_x=0, pad_y=0)
    if bbox is None:
        return {
            "seg_id": sid,
            "passthrough": True,
            "matte_empty": True,
            "crop": None,
            "crop_note": "passthrough_original",
            "plate": [plate_w, plate_h],
            "window_src": [0, 0],
            "width": tw,
            "height": th,
            "n_frames": n,
            "raw_n_frames": raw_n,
            "driver": driver_name,
            "mask": mask_name,
            "master_image": f"cut_video_face_swapper/jobs/{job_name}/masters/{sid}.png",
            "master_exists": (job / "masters" / f"{sid}.png").exists(),
            "skip_first_frames": start,
            "frame_load_cap": n,
            "wan_mode": "hair_back",
            "clamp_x": clamp_x,
        }

    # Cap pixels to landscape face budget — tall right-edge OTS crops otherwise become
    # 576×832 and the Wan sampler wedges at "Sampling start" on this box.
    _hair_max_px = WAN_CROP_W * WAN_CROP_H
    x0, y0, x1, y1 = _expand_bbox_percent(*bbox, plate_w, plate_h, pad_percent)
    if clamp_x > 0:
        x0 = max(x0, clamp_x)
    wan_w, wan_h = _wan_fit_size(x1 - x0, y1 - y0, max_pixels=_hair_max_px)
    x0, y0, x1, y1 = _fit_paste_crop([x0, y0, x1, y1], wan_w, wan_h, plate_w, plate_h)
    if clamp_x > 0 and x0 < clamp_x:
        # Keep left edge fixed at clamp; rebuild aspect-locked box toward plate right.
        cy = 0.5 * (y0 + y1)
        x0 = clamp_x
        x1 = plate_w
        wan_w, wan_h = _wan_fit_size(x1 - x0, max(1, y1 - y0), max_pixels=_hair_max_px)
        ch = int(round((x1 - x0) * wan_h / float(wan_w)))
        y0 = int(round(cy - ch / 2.0))
        y0 = max(0, min(plate_h - ch, y0))
        y1 = min(plate_h, y0 + ch)
        wan_w, wan_h = _wan_fit_size(x1 - x0, y1 - y0, max_pixels=_hair_max_px)

    note = f"hair_back+{pad_percent:g}%"
    if clamp_x > 0:
        note += f" clamp_x={clamp_x}"
    out = _write_seg_crop_bundle(
        job,
        job_name,
        sid,
        drivers,
        mattes_c,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        wan_w=wan_w,
        wan_h=wan_h,
        fps=fps,
        crop_note=note,
        pad_percent=pad_percent,
        raw_n=raw_n,
        master_dx=_master_align_shift(seg)[0],
        master_dy=_master_align_shift(seg)[1],
    )
    out["wan_mode"] = "hair_back"
    out["clamp_x"] = int(clamp_x)
    return out


def _prepare_seg_vace(
    job: Path,
    job_name: str,
    seg: dict,
    driver_name: str,
    mask_name: str,
    *,
    fps: float = 24.0,
    pad_percent: float = 20.0,
    **_kwargs,
) -> dict:
    """No-face path that produced seg02_full.

    Wide rectangle past the partner's face, full height, left strip kept as
    context, the rest black. Master is the reference still, not a moving plate.
    """
    import cv2

    sid = seg["id"]
    start = int(seg["start"])
    raw_n = int(seg["end"] - seg["start"])
    n_read = raw_n
    mattes = _read_video_slice(_input_video(mask_name), start, n_read, gray=True)
    drivers = _read_video_slice(_input_video(driver_name), start, n_read, gray=False)
    n_read = min(n_read, len(mattes), len(drivers))
    mattes, drivers = mattes[:n_read], drivers[:n_read]
    if n_read < 1:
        raise RuntimeError(f"SEG{sid}: no frames for vace")
    plate_h, plate_w = drivers[0].shape[:2]
    rights = _haar_leftish_face_rights(drivers, plate_w)
    if rights:
        x0 = max(0, int(np.median(rights)) - 20)
    else:
        bbox = _union_matte_bbox(mattes, pad_x=0, pad_y=0)
        if bbox is None:
            return {
                "seg_id": sid,
                "passthrough": True,
                "matte_empty": True,
                "crop": None,
                "crop_note": "passthrough_original",
                "plate": [plate_w, plate_h],
                "window_src": [0, 0],
                "width": WAN_CROP_W,
                "height": WAN_CROP_H,
                "n_frames": n_read,
                "raw_n_frames": raw_n,
                "driver": driver_name,
                "mask": mask_name,
                "master_image": f"cut_video_face_swapper/jobs/{job_name}/masters/{sid}.png",
                "master_exists": (job / "masters" / f"{sid}.png").exists(),
                "skip_first_frames": start,
                "frame_load_cap": n_read,
                "wan_mode": "vace",
            }
        x0 = max(0, bbox[0] - int(plate_w * 0.08))
    y0, x1, y1 = 0, plate_w, plate_h
    if x1 - x0 < 64:
        x0 = max(0, plate_w - 64)
    wan_w, wan_h = _wan_fit_size(x1 - x0, y1 - y0, max_pixels=WAN_CROP_W * WAN_CROP_H)
    gen_n = _vace_nframes(n_read)
    out_dir = job / "segs" / sid
    out_dir.mkdir(parents=True, exist_ok=True)

    def _fit(fr):
        return cv2.resize(fr[y0:y1, x0:x1], (wan_w, wan_h), interpolation=cv2.INTER_AREA)

    fitted = [_fit(fr) for fr in drivers]
    while len(fitted) < gen_n:
        fitted.append(fitted[-1].copy())
    rx0 = int(wan_w * 0.06)
    mask = np.zeros((wan_h, wan_w), np.uint8)
    mask[:, rx0:] = 255
    mask = cv2.GaussianBlur(mask, (0, 0), 4)
    soft = (mask.astype(np.float32) / 255.0)[:, :, None]
    hole = []
    for fr in fitted:
        blank = fr.astype(np.float32) * (1.0 - soft)
        hole.append(np.clip(blank, 0, 255).astype(np.uint8))
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    _write_bgr_mp4(fitted, out_dir / "driver.mp4", fps)
    _write_bgr_mp4(hole, out_dir / "vace_hole.mp4", fps)
    _write_bgr_mp4([mask_bgr] * gen_n, out_dir / "vace_mask.mp4", fps)

    master_out = out_dir / "master.png"
    master_src = job / "masters" / f"{sid}.png"
    if master_src.exists():
        im = cv2.imread(str(master_src), cv2.IMREAD_COLOR)
        if im is not None:
            dx, dy = _master_align_shift(seg)
            mcrop = _crop_master_to_plate_window(im, x0, y0, x1, y1, plate_w, plate_h, dx, dy)
            cv2.imwrite(
                str(master_out),
                cv2.resize(mcrop, (wan_w, wan_h), interpolation=cv2.INTER_AREA),
            )
    rel = f"cut_video_face_swapper/jobs/{job_name}/segs/{sid}"
    return {
        "seg_id": sid,
        "crop": [int(x0), int(y0), int(x1), int(y1)],
        "crop_note": f"vace rect x0={x0} full_height context=6%",
        "plate": [plate_w, plate_h],
        "window_src": [int(x1 - x0), int(y1 - y0)],
        "width": int(wan_w),
        "height": int(wan_h),
        "pad_percent": float(pad_percent),
        "n_frames": gen_n,
        "raw_n_frames": raw_n,
        "driver": f"{rel}/vace_hole.mp4",
        "mask": f"{rel}/vace_mask.mp4",
        "master_image": f"{rel}/master.png",
        "master_exists": master_out.exists(),
        "skip_first_frames": 0,
        "frame_load_cap": gen_n,
        "matte_empty": False,
        "passthrough": False,
        "wan_mode": "vace",
        "clamp_x": int(x0),
    }


def _write_bgr_mp4(frames: list, path: Path, fps: float) -> None:
    import subprocess

    h, w = frames[0].shape[:2]
    raw = b"".join(np.ascontiguousarray(f).tobytes() for f in frames)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
            "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ],
        input=raw,
        check=True,
    )


def _vace_workflow(
    driver: str,
    mask: str,
    master: str,
    w: int,
    h: int,
    n: int,
    prefix: str,
    seed: int,
    job_name: str = "",
    seg_id: str = "",
) -> dict:
    """Same graph as the seg02_full run: black rect in, master ref, VACE 14B."""
    wf = {
        "1": {
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": driver,
                "force_rate": 24.0,
                "force_size": "Disabled",
                "custom_width": w,
                "custom_height": h,
                "frame_load_cap": n,
                "skip_first_frames": 0,
                "select_every_nth": 1,
            },
        },
        "2": {
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": mask,
                "force_rate": 24.0,
                "force_size": "Disabled",
                "custom_width": w,
                "custom_height": h,
                "frame_load_cap": n,
                "skip_first_frames": 0,
                "select_every_nth": 1,
            },
        },
        "3": {"class_type": "ImageToMask", "inputs": {"image": ["2", 0], "channel": "red"}},
        "4": {
            "class_type": "CVFSFeatherMask",
            "inputs": {"mask": ["3", 0], "expand": 0, "blur": 1},
        },
        "6": {"class_type": "LoadImage", "inputs": {"image": master}},
        "13": {
            "class_type": "WanVideoVAELoader",
            "inputs": {"model_name": "wan2.1_vae.safetensors", "precision": "bf16"},
        },
        "15": {
            "class_type": "WanVideoBlockSwap",
            "inputs": {
                "blocks_to_swap": 30,
                "offload_img_emb": False,
                "offload_txt_emb": False,
                "use_non_blocking": False,
                "prefetch_blocks": 0,
                "vace_blocks_to_swap": 8,
            },
        },
        "16": {
            "class_type": "WanVideoModelLoader",
            "inputs": {
                "model": "wan2.1_vace_14B_fp8_scaled.safetensors",
                "base_precision": "bf16",
                "quantization": "disabled",
                "load_device": "offload_device",
                "attention_mode": "sdpa",
                "block_swap_args": ["15", 0],
            },
        },
        "18": {
            "class_type": "WanVideoTextEncodeCached",
            "inputs": {
                "model_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                "precision": "bf16",
                "positive_prompt": (
                    "Photorealistic back of a head, warm brown glossy tight curls and natural hair volume "
                    "matching the reference still, over-the-shoulder, shallow depth of field, film grain. "
                    "Keep the original head silhouette. No face."
                ),
                "negative_prompt": (
                    "face, eyes, lips, mouth, teeth, front view, profile face, looking at camera, "
                    "talking, blurry, plastic, burned highlights, deformed"
                ),
                "quantization": "disabled",
                "use_disk_cache": True,
                "device": "cpu",
            },
        },
        "17": {
            "class_type": "WanVideoVACEEncode",
            "inputs": {
                "vae": ["13", 0],
                "width": w,
                "height": h,
                "num_frames": n,
                "strength": 1.0,
                "vace_start_percent": 0.0,
                "vace_end_percent": 1.0,
                "input_frames": ["1", 0],
                "ref_images": ["6", 0],
                "input_masks": ["4", 0],
                "tiled_vae": False,
            },
        },
        "19": {
            "class_type": "WanVideoSampler",
            "inputs": {
                "model": ["16", 0],
                "image_embeds": ["17", 0],
                "text_embeds": ["18", 0],
                "steps": 12,
                "cfg": 5.0,
                "shift": 5.0,
                "seed": int(seed),
                "force_offload": True,
                "scheduler": "unipc",
                "riflex_freq_index": 0,
            },
        },
        "20": {
            "class_type": "WanVideoDecode",
            "inputs": {
                "vae": ["13", 0],
                "samples": ["19", 0],
                "enable_vae_tiling": False,
                "tile_x": 272,
                "tile_y": 272,
                "tile_stride_x": 144,
                "tile_stride_y": 128,
            },
        },
        "21": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["20", 0],
                "frame_rate": 24.0,
                "loop_count": 0,
                "filename_prefix": prefix,
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 18,
                "save_metadata": True,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
            },
        },
    }
    if job_name and seg_id:
        wf["22"] = {
            "class_type": "CVFSPasteVacePlate",
            "inputs": {"images": ["20", 0], "job_name": job_name, "seg_id": seg_id},
        }
        wf["23"] = {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["22", 0],
                "frame_rate": 24.0,
                "loop_count": 0,
                "filename_prefix": f"cut_video_face_swapper/{job_name}/vace_full/{seg_id}",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 18,
                "save_metadata": True,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
            },
        }
    return wf


def _prepare_seg_wan_crop_face_gate(
    job: Path,
    job_name: str,
    seg: dict,
    driver_name: str,
    mask_name: str,
    *,
    tw: int = WAN_CROP_W,
    th: int = WAN_CROP_H,
    fps: float = 24.0,
    pad_percent: float = 20.0,
) -> dict:
    """TEST hybrid prepare: if matte region rarely contains a face → hair_back crop; else face crop."""
    import cv2

    start = int(seg["start"])
    raw_n = int(seg["end"] - seg["start"])
    n = _wan_nframes(raw_n)
    mattes = _read_video_slice(_input_video(mask_name), start, n, gray=True)
    drivers = _read_video_slice(_input_video(driver_name), start, n, gray=False)
    n = min(n, len(mattes), len(drivers))
    mattes, drivers = mattes[:n], drivers[:n]
    bbox = _union_matte_bbox(mattes, pad_x=0, pad_y=0)
    face_hits = 0
    checked = 0
    if bbox is not None:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        x0, y0, x1, y1 = bbox
        step = max(1, n // 8)
        for i in range(0, n, step):
            checked += 1
            fr = drivers[i][y0:y1, x0:x1]
            if fr.size == 0:
                continue
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr
            found = cascade.detectMultiScale(gray, 1.08, 4, minSize=(40, 40))
            if found is not None and len(found) > 0:
                face_hits += 1
    face_ratio = (face_hits / checked) if checked else 0.0
    use_hair = face_ratio < 0.25
    if use_hair:
        out = _prepare_seg_wan_crop_hair(
            job,
            job_name,
            seg,
            driver_name,
            mask_name,
            tw=tw,
            th=th,
            fps=fps,
            pad_percent=pad_percent,
        )
    else:
        out = _prepare_seg_wan_crop(
            job,
            job_name,
            seg,
            driver_name,
            mask_name,
            tw=tw,
            th=th,
            fps=fps,
            pad_percent=pad_percent,
        )
    out["wan_mode"] = "face_gate"
    out["face_probe_ratio"] = float(face_ratio)
    out["prepare_branch"] = "hair_back" if use_hair else "face"
    note = out.get("crop_note") or ""
    out["crop_note"] = f"face_gate/{out['prepare_branch']} probe={face_ratio:.2f} {note}".strip()
    return out


def _rect_from_matte_u8(
    frames: np.ndarray,
    pad_x: int = 32,
    pad_y: int = 24,
    envelope: int = 2,
    pad_x_ratio: float = 0.0,
) -> np.ndarray:
    """v1.4 padded AABB rect. frames uint8 [F,H,W] or [F,H,W,C], already the segment (no Shot-1 crop)."""
    if frames.ndim == 4:
        g = frames[..., 0]
    else:
        g = frames
    f, th, tw = g.shape[:3]
    boxes: list[list[int] | None] = []
    raw = []
    for i in range(f):
        m = g[i]
        ys, xs = np.where(m > 127)
        raw.append(m)
        if len(xs) == 0:
            boxes.append(None)
            continue
        span = int(xs.max()) - int(xs.min()) + 1
        px = max(int(pad_x), int(round(span * float(pad_x_ratio))))
        boxes.append(
            [
                max(0, int(xs.min()) - px),
                max(0, int(ys.min()) - pad_y),
                min(tw, int(xs.max()) + px + 1),
                min(th, int(ys.max()) + pad_y + 1),
            ]
        )
    if not any(b is not None for b in boxes):
        raise ValueError("Matte segment is empty — no face/hair pixels to build a rect from")
    last = None
    for i, b in enumerate(boxes):
        if b is None and last is not None:
            boxes[i] = last[:]
        elif b is not None:
            last = b[:]
    for i in range(len(boxes)):
        if boxes[i] is None:
            for j in range(i + 1, len(boxes)):
                if boxes[j] is not None:
                    boxes[i] = boxes[j][:]
                    break
    arr = np.array(boxes, dtype=np.float32)
    env = arr.copy()
    rad = max(0, int(envelope))
    for i in range(len(arr)):
        a = max(0, i - rad)
        b = min(len(arr), i + rad + 1)
        env[i, 0] = arr[a:b, 0].min()
        env[i, 1] = arr[a:b, 1].min()
        env[i, 2] = arr[a:b, 2].max()
        env[i, 3] = arr[a:b, 3].max()
    out = np.zeros((f, th, tw), np.uint8)
    for i in range(f):
        x0, y0, x1b, y1b = [_snap8(v, lim) for v, lim in zip(env[i], [tw, th, tw, th])]
        rx0, ry0, rx1, ry1 = boxes[i]
        x0 = min(x0, _snap8(rx0, tw))
        y0 = min(y0, _snap8(ry0, th))
        x1b = max(x1b, min(tw, ((int(rx1) + 7) // 8) * 8))
        y1b = max(y1b, min(th, ((int(ry1) + 7) // 8) * 8))
        if x1b <= x0:
            x1b = min(tw, x0 + 8)
        if y1b <= y0:
            y1b = min(th, y0 + 8)
        out[i, y0:y1b, x0:x1b] = 255
    return out


def _picked_matte_rect(
    job_name: str,
    seg_index: int,
    pad_x: int = 32,
    pad_y: int = 24,
    pad_x_ratio: float = FACE_PAD_X_RATIO,
) -> np.ndarray | None:
    """Matte rect at the frame chosen in «Редактировать», in the seg mask's pixel space."""
    name = str(job_name or "").strip()
    if not name:
        return None
    job = _job_root(name)
    cuts_path = job / "cuts.json"
    if not cuts_path.exists():
        return None
    segs = json.loads(cuts_path.read_text(encoding="utf-8")).get("segments") or []
    if seg_index < 0 or seg_index >= len(segs):
        return None
    seg = segs[int(seg_index)]
    start = int(seg["start"])
    end = int(seg["end"])
    ref = int(seg.get("ref_frame", start))
    if ref < start or ref >= end:
        ref = start
    offset = ref - start
    mask_path = job / "segs" / str(seg["id"]) / "mask.mp4"
    if not mask_path.exists():
        return None
    import cv2

    total = int(cv2.VideoCapture(str(mask_path)).get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        return None
    offset = max(0, min(offset, total - 1))
    bgr = _extract_frame_bgr(mask_path, offset)
    gray = bgr.max(axis=2)
    ys, xs = np.where(gray > 127)
    if len(xs) == 0:
        return None
    h, w = gray.shape[:2]
    span = int(xs.max()) - int(xs.min()) + 1
    px = max(int(pad_x), int(round(span * float(pad_x_ratio))))
    x0 = max(0, int(xs.min()) - px)
    y0 = max(0, int(ys.min()) - pad_y)
    x1 = min(w, int(xs.max()) + px + 1)
    y1 = min(h, int(ys.max()) + pad_y + 1)
    rect = np.zeros((h, w), np.uint8)
    rect[y0:y1, x0:x1] = 255
    return rect


def _largest_mask_frame(mask_u8: np.ndarray) -> np.ndarray:
    """Frame 0 is often a speck (a dark sleeve, a cat) while the head shows up later."""
    if mask_u8.ndim == 2:
        return mask_u8
    areas = (mask_u8 > 127).reshape(mask_u8.shape[0], -1).sum(axis=1)
    return mask_u8[int(np.argmax(areas))]


def _crop_ref_in_rect(image_u8: np.ndarray, rect_u8: np.ndarray, size: int = 512) -> np.ndarray:
    """v1.4 identity slot: crop master to the wipe rect, then cover-fill 512 (face/hair fill the square).

    Letterboxing the whole Wan window left ~20% black and shrank the face — CLIP then
    under-weighted identity and Wan kept driver hair inside the wipe. With a face+hair
    wipe the rect is large enough that cover-fill is a head, not an eye zoom.
    """
    import cv2

    if rect_u8.ndim == 3:
        rect_u8 = rect_u8[..., 0]
    ys, xs = np.where(rect_u8 > 127)
    img = _normalize_rgb_u8(image_u8)
    H, W = img.shape[:2]
    if len(xs) == 0:
        side = min(H, W)
        y0, x0 = 0, max(0, (W - side) // 2)
        crop = img[y0 : y0 + side, x0 : x0 + side]
    else:
        rh, rw = rect_u8.shape[:2]
        sx = W / max(1, rw)
        sy = H / max(1, rh)
        x0 = max(0, int(xs.min() * sx))
        y0 = max(0, int(ys.min() * sy))
        x1 = min(W, int(np.ceil((xs.max() + 1) * sx)))
        y1 = min(H, int(np.ceil((ys.max() + 1) * sy)))
        if x1 <= x0 or y1 <= y0:
            crop = img
        else:
            crop = img[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    scale = max(size / max(1, cw), size / max(1, ch))
    nw = max(size, int(round(cw * scale)))
    nh = max(size, int(round(ch * scale)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    x_off = max(0, (nw - size) // 2)
    y_off = 0  # top-aligned like ImageResizeKJv2 crop_position=top in v1.4
    out = resized[y_off : y_off + size, x_off : x_off + size]
    if out.shape[0] != size or out.shape[1] != size:
        canvas = np.zeros((size, size, 3), np.uint8)
        canvas[: out.shape[0], : out.shape[1]] = out
        out = canvas
    return out


def _build_vace_seg_workflow(
    job: Path,
    cfg: dict,
    seg: dict,
    seed: int = 210,
    crop_info: dict | None = None,
    pad_percent: float = 20.0,
) -> Path:
    """Write the seg02_full VACE graph. Face mode never enters here."""
    sid = seg["id"]
    driver = cfg.get("comfy_driver") or cfg.get("driver_filename") or ""
    mask = cfg.get("comfy_mask_rect") or cfg.get("mask_filename") or ""
    if crop_info is None:
        crop_info = _prepare_seg_vace(
            job,
            cfg["name"],
            seg,
            driver,
            mask,
            fps=float(cfg.get("fps") or 24.0),
            pad_percent=float(pad_percent),
        )
    out_dir = job / "workflows"
    out_dir.mkdir(parents=True, exist_ok=True)
    if crop_info.get("passthrough"):
        seg["passthrough"] = True
        seg["wan_crop"] = None
        seg["wan_crop_note"] = "passthrough_original"
        seg["wan_mode"] = "vace"
        note = out_dir / f"seg_{sid}_SKIP.txt"
        note.write_text("empty matte — original frames kept at assemble\n", encoding="utf-8")
        return note
    seg["wan_crop"] = list(crop_info["crop"])
    seg["wan_crop_note"] = crop_info.get("crop_note") or "vace"
    seg["wan_size"] = [int(crop_info["width"]), int(crop_info["height"])]
    seg["wan_mode"] = "vace"
    seg["passthrough"] = False
    prefix = f"cut_video_face_swapper/{cfg['name']}/parts/{sid}"
    wf = _vace_workflow(
        crop_info["driver"],
        crop_info["mask"],
        crop_info["master_image"],
        int(crop_info["width"]),
        int(crop_info["height"]),
        int(crop_info["n_frames"]),
        prefix,
        seed,
        job_name=cfg["name"],
        seg_id=sid,
    )
    path = out_dir / f"seg_{sid}_API.json"
    path.write_text(json.dumps(wf, indent=2), encoding="utf-8")
    return path


def _build_seg_workflow(
    job: Path,
    cfg: dict,
    seg: dict,
    seed: int = 210,
    crop_info: dict | None = None,
    pad_percent: float = 20.0,
) -> Path:
    mode = _resolve_wan_mode(seg)
    if mode == "vace":
        return _build_vace_seg_workflow(job, cfg, seg, seed=seed, crop_info=crop_info, pad_percent=pad_percent)
    templates = Path(folder_paths.get_input_directory()) / "cut_video_face_swapper" / "templates"
    if mode == "hair_back":
        tmpl = templates / "WanAnimate_hair_back_API.json"
        local_tmpl = job / "template_wan_hair_api.json"
    elif mode == "face_gate":
        tmpl = templates / "WanAnimate_face_gate_API.json"
        local_tmpl = job / "template_wan_face_gate_api.json"
        if not tmpl.exists():
            # Fall back to hair_back graph shape then inject gate (created below if missing).
            tmpl = templates / "WanAnimate_hair_back_API.json"
    else:
        tmpl = templates / "WanAnimate_segment_API.json"
        local_tmpl = job / "template_wan_api.json"
    if local_tmpl.exists():
        tmpl = local_tmpl
    if not tmpl.exists():
        raise FileNotFoundError(f"Wan template missing: {tmpl}")
    wf = json.loads(tmpl.read_text(encoding="utf-8"))
    sid = seg["id"]
    driver = cfg.get("comfy_driver") or cfg.get("driver_filename") or ""
    mask = cfg.get("comfy_mask_rect") or cfg.get("mask_filename") or ""
    if not driver or not mask:
        raise ValueError("job.json missing comfy_driver / comfy_mask_rect — re-run detect with filenames filled")
    if crop_info is None:
        if mode == "hair_back":
            prep = _prepare_seg_wan_crop_hair
        elif mode == "face_gate":
            # Hybrid: hair clamp when matte window has almost no face (OTS), else stock face crop.
            prep = _prepare_seg_wan_crop_face_gate
        else:
            prep = _prepare_seg_wan_crop
        crop_info = prep(
            job,
            cfg["name"],
            seg,
            driver,
            mask,
            tw=WAN_CROP_W,
            th=WAN_CROP_H,
            fps=float(cfg.get("fps") or 24.0),
            pad_percent=float(pad_percent),
        )
    if crop_info.get("passthrough"):
        seg["passthrough"] = True
        seg["wan_crop"] = None
        seg["wan_crop_note"] = "passthrough_original"
        seg["wan_mode"] = mode
        note = job / "workflows" / f"seg_{sid}_SKIP.txt"
        note.parent.mkdir(parents=True, exist_ok=True)
        note.write_text("empty matte — original frames kept at assemble\n", encoding="utf-8")
        return note
    seg["wan_crop"] = list(crop_info["crop"])
    seg["wan_crop_note"] = crop_info.get("crop_note") or "from_matte"
    if crop_info.get("track") and not seg.get("test_preview"):
        seg["wan_track"] = crop_info["track"]
    seg["wan_size"] = [int(crop_info["width"]), int(crop_info["height"])]
    seg["wan_mode"] = mode
    seg["passthrough"] = False
    w, h = int(crop_info["width"]), int(crop_info["height"])
    n = int(crop_info["n_frames"])
    driver = crop_info["driver"]
    mask = crop_info["mask"]
    master = crop_info["master_image"]

    wf["1"]["inputs"]["video"] = driver
    wf["1"]["inputs"]["skip_first_frames"] = 0
    wf["1"]["inputs"]["frame_load_cap"] = n
    wf["1"]["inputs"]["custom_width"] = w
    wf["1"]["inputs"]["custom_height"] = h
    wf["2"]["inputs"]["video"] = mask
    wf["2"]["inputs"]["skip_first_frames"] = 0
    wf["2"]["inputs"]["frame_load_cap"] = n
    wf["2"]["inputs"]["custom_width"] = w
    wf["2"]["inputs"]["custom_height"] = h
    wf["6"]["inputs"]["image"] = master
    # v1.4 wipe is a padded rect of THIS matte, not Blockify of a prebaked square.
    if wf.get("4", {}).get("class_type") == "BlockifyMask":
        wf["4"] = {
            "class_type": "CVFSRectFromMask",
            "inputs": {
                "mask": ["3", 0],
                "pad_x": 32,
                "pad_y": 24,
                "envelope": 2,
                "pad_x_ratio": FACE_PAD_X_RATIO,
            },
        }
    ref_abs = int(seg.get("ref_frame", seg.get("start", 0)))
    ref_index = min(max(0, ref_abs - int(seg["start"])), max(0, n - 1))
    if wf.get("7", {}).get("class_type") in {"ImageResizeKJv2", "CVFSCropRefInRect"}:
        wf["7"] = {
            "class_type": "CVFSCropRefInRect",
            "inputs": {
                "image": ["6", 0],
                "mask": ["4", 0],
                "size": 512,
                "job_name": cfg.get("name") or "",
                "seg_index": int(seg.get("id") or 0),
                # Identity crop from the same frame that was picked for the master.
                "ref_index": int(ref_index),
            },
        }
    if "17" in wf and "num_frames" in wf["17"].get("inputs", {}):
        wf["17"]["inputs"]["num_frames"] = n
        wf["17"]["inputs"]["width"] = w
        wf["17"]["inputs"]["height"] = h
        if mode == "face":
            # No window colormatch → skin stays closer to the master; assemble still matches the plate.
            wf["17"]["inputs"]["colormatch"] = str(seg.get("colormatch", FACE_COLORMATCH_DEFAULT))
            wf["17"]["inputs"]["face_strength"] = float(seg.get("face_strength", FACE_STRENGTH_DEFAULT))
            wf["17"]["inputs"]["pose_strength"] = float(seg.get("pose_strength", 1.0))
            if "14" in wf and "strength_0" in wf["14"].get("inputs", {}):
                wf["14"]["inputs"]["strength_0"] = float(seg.get("relight_strength", FACE_RELIGHT_DEFAULT))
        if mode == "hair_back":
            wf["17"]["inputs"]["face_strength"] = float(seg.get("face_strength", 0.0))
            wf["17"]["inputs"]["pose_strength"] = float(seg.get("pose_strength", 0.0))
            # VitPose face crops on occiput still build motion_vec even at strength 0.
            # Drop the wire so sampler injects clean zeros (masked path) instead.
            wf["17"]["inputs"].pop("face_images", None)
            # Pose skeleton on occiput still steers a front-face prior — disconnect entirely.
            wf["17"]["inputs"].pop("pose_images", None)
            # P0: no black wipe void — keep driver pixels; soft FaceHair mask only.
            wf["4"] = {
                "class_type": "CVFSFeatherMask",
                "inputs": {
                    "mask": ["3", 0],
                    "expand": int(seg.get("mask_expand", 4)),
                    "blur": int(seg.get("mask_blur", 7)),
                },
            }
            wf["17"]["inputs"]["bg_images"] = ["1", 0]
            wf["17"]["inputs"]["mask"] = ["4", 0]
            wf["17"]["inputs"]["colormatch"] = str(seg.get("colormatch", "disabled"))
            if "10" in wf:
                # Head oval / face sticks on occiput → invented front face.
                wf["10"]["inputs"]["draw_head"] = False
            if "14" in wf and "strength_0" in wf["14"].get("inputs", {}):
                # Relight @1 burns curls; hair needs a light touch.
                wf["14"]["inputs"]["strength_0"] = float(seg.get("relight_strength", 0.4))
            if "18" in wf:
                wf["18"]["inputs"]["positive_prompt"] = str(
                    seg.get(
                        "positive_prompt",
                        "Photographic rear / over-the-shoulder shot of the back of a head, "
                        "detailed curly wavy hair texture and natural volume, soft shallow depth of field, "
                        "cinematic 35mm film still, film grain. Keep the original head silhouette and camera angle. "
                        "No face toward camera.",
                    )
                )
                wf["18"]["inputs"]["negative_prompt"] = str(
                    seg.get(
                        "negative_prompt",
                        "face, eyes, lips, mouth, teeth, tongue, front view, three-quarter face, profile face, "
                        "looking at camera, talking, open mouth, invented face on back of head, "
                        "burned highlights, overexposed hair, plastic skin, cartoon, anime, blurry",
                    )
                )
            if "15" in wf:
                # non_blocking prefetch thrash: first sampler step never finishes on 5080.
                wf["15"]["inputs"]["use_non_blocking"] = False
                wf["15"]["inputs"]["prefetch_blocks"] = 0
                wf["15"]["inputs"]["blocks_to_swap"] = int(seg.get("blocks_to_swap", 30))
            # P2: inject master hair into V2V init under soft mask, then mild denoise.
            # Raising denoise alone (0.88) collapses to green face artifacts; blend+0.72 is safer.
            denoise = float(seg.get("denoise_strength", 0.55))
            blend = float(seg.get("init_blend", 0.85))
            wf["24"] = {
                "class_type": "CVFSBlendMasked",
                "inputs": {
                    "video": ["1", 0],
                    # segs/XX/master.png is already wan-sized occiput crop — not the 512² CLIP square
                    "ref": ["6", 0],
                    "mask": ["4", 0],
                    "strength": blend,
                },
            }
            wf["23"] = {
                "class_type": "WanVideoEncode",
                "inputs": {
                    "vae": ["13", 0],
                    "image": ["24", 0],
                    "enable_vae_tiling": False,
                    "tile_x": 272,
                    "tile_y": 272,
                    "tile_stride_x": 144,
                    "tile_stride_y": 128,
                    "mask": ["4", 0],
                },
            }
            if "19" in wf:
                wf["19"]["inputs"]["samples"] = ["23", 0]
                wf["19"]["inputs"]["denoise_strength"] = denoise
                wf["19"]["inputs"]["add_noise_to_samples"] = True
            if "12" in wf and "strength_1" in wf["12"].get("inputs", {}):
                wf["12"]["inputs"]["strength_1"] = float(seg.get("clip_strength", 1.2))
            if "14" in wf and "strength_0" in wf["14"].get("inputs", {}):
                wf["14"]["inputs"]["strength_0"] = float(seg.get("relight_strength", 0.35))
            if "18" in wf:
                wf["18"]["inputs"]["positive_prompt"] = str(
                    seg.get(
                        "positive_prompt",
                        "Match the reference hair exactly: warm brown glossy curls, tight wavy volume, "
                        "strand detail and sheen from the reference still. Photographic rear / over-the-shoulder "
                        "back of head, keep original skull silhouette and camera angle, cinematic 35mm film still, "
                        "film grain. No face toward camera.",
                    )
                )
        if mode == "face_gate":
            # A shot that is the back of the head the whole way has nothing for the face
            # branch to track. Leaving face_strength at 1 there stalls the sampler.
            # A turn (face then away) keeps face_strength 1; the gate fades the crop.
            back_only = (crop_info or {}).get("prepare_branch") == "hair_back"
            default_face = 0.0 if back_only else 1.0
            # pose_strength 1 on an all-back window stalls the sampler; 0.25 completes.
            default_pose = 0.25 if back_only else 1.0
            wf["17"]["inputs"]["face_strength"] = float(seg.get("face_strength", default_face))
            wf["17"]["inputs"]["pose_strength"] = float(seg.get("pose_strength", default_pose))
            wf["17"]["inputs"]["colormatch"] = str(seg.get("colormatch", "hm-mvgd-hm"))
            if back_only:
                wf["17"]["inputs"].pop("face_images", None)
                # Same P0 as hair_back: keep driver, soft matte, no head sticks, soft relight.
                wf["4"] = {
                    "class_type": "CVFSFeatherMask",
                    "inputs": {
                        "mask": ["3", 0],
                        "expand": int(seg.get("mask_expand", 4)),
                        "blur": int(seg.get("mask_blur", 7)),
                    },
                }
                wf["17"]["inputs"]["bg_images"] = ["1", 0]
                wf["17"]["inputs"]["mask"] = ["4", 0]
                if "10" in wf:
                    wf["10"]["inputs"]["draw_head"] = False
                if "14" in wf and "strength_0" in wf["14"].get("inputs", {}):
                    wf["14"]["inputs"]["strength_0"] = float(seg.get("relight_strength", 0.4))
                if "15" in wf:
                    wf["15"]["inputs"]["use_non_blocking"] = False
                    wf["15"]["inputs"]["prefetch_blocks"] = 0
                    wf["15"]["inputs"]["blocks_to_swap"] = int(seg.get("blocks_to_swap", 30))
    if "9" in wf:
        wf["9"]["inputs"]["width"] = w
        wf["9"]["inputs"]["height"] = h
    if "10" in wf:
        wf["10"]["inputs"]["width"] = w
        wf["10"]["inputs"]["height"] = h
    wf["9"]["inputs"].pop("retarget_image", None)
    wf["9"]["inputs"].pop("face_padding", None)
    if "retarget_padding" in wf.get("10", {}).get("inputs", {}):
        wf["10"]["inputs"]["retarget_padding"] = 0
    if mode == "face_gate":
        # Inject / refresh TEST face gate between PoseAndFaceDetection faces → AnimateEmbeds.
        wf["22"] = {
            "class_type": "CVFSFaceGate",
            "inputs": {
                "images": ["1", 0],
                "face_images": ["9", 1],
                "pose_data": ["9", 0],
                "min_face_conf": float(seg.get("min_face_conf", 0.35)),
                "min_kp_frac": float(seg.get("min_kp_frac", 0.25)),
            },
        }
        if "17" in wf:
            wf["17"]["inputs"]["face_images"] = ["22", 0]
        if "10" in wf:
            wf["10"]["inputs"]["pose_data"] = ["22", 1]
        if "14" in wf and "strength_0" in wf["14"].get("inputs", {}):
            wf["14"]["inputs"]["strength_0"] = float(seg.get("relight_strength", 1.0))
        if "18" in wf:
            wf["18"]["inputs"]["positive_prompt"] = str(
                seg.get(
                    "positive_prompt",
                    "Match the reference image. While the face is visible, keep that face. "
                    "As the head turns away, draw the back of the head and the hair from the reference, "
                    "following the pose. Closed mouth, photorealistic 35mm film still, natural hair texture, film grain.",
                )
            )
            wf["18"]["inputs"]["negative_prompt"] = str(
                seg.get(
                    "negative_prompt",
                    "invented front face on back of head, melted face, open mouth, teeth, talking, "
                    "laughing, overexposed, bleached colors, burned highlights, plastic skin, "
                    "airbrushed, cartoon, anime, blurry, deformed face",
                )
            )
    if "19" in wf and "seed" in wf["19"].get("inputs", {}):
        wf["19"]["inputs"]["seed"] = seed
    # Option A: swap master face onto PoseAndFaceDetection crops, keep eyelid/mouth motion.
    if mode == "face" and _is_reactor_prep(seg.get("face_images_prep")):
        fast = _is_fast_profile(seg)
        detect = FAST_REACTOR_DETECT if fast else "retinaface_resnet50"
        wf["22"] = {
            "class_type": "ReActorFaceSwap",
            "inputs": {
                "enabled": True,
                "input_image": ["9", 1],
                "swap_model": str(seg.get("reactor_swap_model") or FACE_PREP_REACTOR_MODEL),
                "facedetection": detect,
                "face_restore_model": "none",
                "face_restore_visibility": 1.0,
                "codeformer_weight": 0.5,
                "detect_gender_input": "no",
                "detect_gender_source": "no",
                "input_faces_index": "0",
                "source_faces_index": "0",
                "console_log_level": 1,
                "source_image": ["7", 0],
            },
        }
        if "17" in wf:
            wf["17"]["inputs"]["face_images"] = ["22", 0]
            # Full mimicry: crops already carry master identity.
            wf["17"]["inputs"]["face_strength"] = float(seg.get("face_strength", 1.0))
    if mode == "face":
        _apply_face_vram_guard(wf, w, h, n, seg)
    if seg.get("test_preview"):
        prep = str(seg.get("face_images_prep") or "").lower()
        if _is_fast_profile(seg):
            tag = "Af"
        elif _is_reactor_prep(prep):
            tag = "A"
        else:
            tag = ""
        prefix = f"cut_video_face_swapper/{cfg['name']}/preview/{sid}{('_' + tag) if tag else ''}"
    else:
        prefix = f"cut_video_face_swapper/{cfg['name']}/parts/{sid}"
    if "20" in wf:
        wf["20"]["inputs"]["filename_prefix"] = f"{prefix}_decoded"
    if "21" in wf:
        wf["21"]["inputs"]["filename_prefix"] = prefix
    out = job / "workflows" / f"seg_{sid}_API.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(wf, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


class CVFSSaveMasterForSegment:
    """Save a Gemini master still into jobs/<name>/masters/XX.png for one SEG id."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "master_image": ("IMAGE",),
                "job_name": ("STRING", {"default": "comfy_detect_demo"}),
                "seg_id": ("INT", {"default": 0, "min": 0, "max": 999}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("saved_path", "preview")
    FUNCTION = "run"
    CATEGORY = "cut_video_face_swapper"

    def run(self, master_image, job_name, seg_id):
        import cv2

        job = _job_root(job_name)
        masters = job / "masters"
        masters.mkdir(parents=True, exist_ok=True)
        sid = f"{int(seg_id):02d}"
        path = masters / f"{sid}.png"
        fr = _tensor_batch_to_uint8(master_image)[0]
        cv2.imwrite(str(path), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        rel = f"cut_video_face_swapper/jobs/{job_name}/masters/{sid}.png"
        return (rel, master_image)


class CVFSBuildWanWorkflowsFromMasters:
    """For every masters/XX.png that exists, write workflows/seg_XX_API.json (any N)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "job_name": ("STRING", {"default": "comfy_detect_demo"}),
                "seed": ("INT", {"default": 210, "min": 0, "max": 0xFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("report", "built_count")
    FUNCTION = "run"
    CATEGORY = "cut_video_face_swapper"
    OUTPUT_NODE = True

    def run(self, job_name, seed):
        job = _job_root(job_name)
        cuts = json.loads((job / "cuts.json").read_text(encoding="utf-8"))
        cfg_path = job / "job.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        else:
            cfg = {"name": job_name}
        # merge filenames from cuts if present
        if cuts.get("driver_filename"):
            cfg["comfy_driver"] = cuts["driver_filename"]
        if cuts.get("mask_filename"):
            cfg["comfy_mask_rect"] = cuts["mask_filename"]
        cfg["name"] = job_name

        built = []
        missing = []
        for seg in cuts["segments"]:
            master = job / "masters" / f"{seg['id']}.png"
            if not master.exists():
                missing.append(seg["id"])
                continue
            out = _build_seg_workflow(job, cfg, seg, seed=seed)
            built.append(str(out.name))
            seg["status"] = "ready_to_run"
        (job / "cuts.json").write_text(json.dumps(cuts, indent=2, ensure_ascii=False), encoding="utf-8")
        report = {
            "built": built,
            "missing_masters": missing,
            "load_each": [f"cut_video_face_swapper/jobs/{job_name}/workflows/{b}" for b in built],
            "hint": "Queue each seg_XX_API.json (one by one is fine). Then python cvfs.py assemble " + job_name,
        }
        return (json.dumps(report, indent=2, ensure_ascii=False), len(built))


def _resolve_job_videos(job: Path, cuts: dict, cfg: dict, driver_filename: str = "", mask_filename: str = "") -> dict:
    """Prefer job-local baked videos from Detect; filename args are optional leftovers."""
    del driver_filename, mask_filename
    cfg = dict(cfg)
    job_name = cfg.get("name") or cuts.get("job") or job.name
    driver, mask = _prefer_job_media(str(job_name), cuts, cfg)
    cfg["comfy_driver"] = driver
    cfg["comfy_mask_rect"] = mask
    cuts["driver_filename"] = driver
    cuts["mask_filename"] = mask
    return cfg


def _queue_prompt(comfy_url: str, wf: dict) -> str:
    import urllib.request

    payload = json.dumps({"prompt": wf}).encode("utf-8")
    req = urllib.request.Request(
        comfy_url.rstrip("/") + "/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=120).read().decode())
    if resp.get("node_errors"):
        raise RuntimeError(str(resp["node_errors"])[:500])
    return str(resp.get("prompt_id", "?"))


def _part_mp4(job: Path, sid: str) -> Path:
    return job / "parts" / f"{sid}.mp4"


def _output_part_is_vace(job_name: str, sid: str) -> bool:
    """True when the newest-looking part was actually produced by WanVideoVACEEncode."""
    for p in _find_output_parts(job_name, sid):
        png = p.with_suffix(".png")
        if png.exists() and b"WanVideoVACEEncode" in png.read_bytes():
            return True
    return False


def queue_one_segment(job_name: str, seg_index: int, pad_percent: float = 20.0, comfy_url: str = "http://127.0.0.1:8188") -> dict:
    """Queue one SEG with the graph its mode requires. VACE uses the seg02_full graph, not the canvas Animate nodes."""
    job = _job_root(job_name)
    cuts_path = job / "cuts.json"
    if not cuts_path.exists():
        raise FileNotFoundError(f"No cuts.json — run Z1 Detect first ({job})")
    cuts = json.loads(cuts_path.read_text(encoding="utf-8"))
    segs = cuts.get("segments") or []
    i = int(seg_index)
    if i < 0 or i >= len(segs):
        raise IndexError(f"seg_index {i} out of range 0..{len(segs)-1}")
    seg = segs[i]
    cfg_path = job / "job.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {"name": job_name}
    cfg["name"] = job_name
    cfg = _resolve_job_videos(job, cuts, cfg)
    _assign_unlocked_mode(seg, _input_video(cfg["comfy_driver"]), _input_video(cfg["comfy_mask_rect"]))
    mode = _resolve_wan_mode(seg)
    seed_i = 231 if mode == "vace" else 210 + i
    wf_path = _build_seg_workflow(job, cfg, seg, seed=seed_i, pad_percent=float(pad_percent))
    cuts["segments"] = segs
    cuts_path.write_text(json.dumps(cuts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if wf_path.suffix.lower() != ".json":
        return {"job": job_name, "seg_id": seg["id"], "wan_mode": mode, "queued": False, "reason": "passthrough"}
    wf = json.loads(wf_path.read_text(encoding="utf-8"))
    prompt_id = _queue_prompt(comfy_url, wf)
    return {
        "job": job_name,
        "seg_id": seg["id"],
        "wan_mode": mode,
        "queued": True,
        "prompt_id": prompt_id,
        "workflow": str(wf_path),
        "seed": seed_i,
    }


def _find_output_parts(job_name: str, sid: str) -> list[Path]:
    """Find Wan VideoCombine outputs for this SEG (nested under output/**/parts/)."""
    try:
        out_root = Path(folder_paths.get_output_directory())
    except Exception:
        return []
    hits = list(out_root.glob(f"**/cut_video_face_swapper/**/parts/{sid}_*.mp4"))
    hits += list(out_root.glob(f"**/{job_name}/parts/{sid}_*.mp4"))
    # de-dupe, newest last
    uniq = {str(p.resolve()): p for p in hits if p.is_file() and p.stat().st_size > 0}
    return sorted(uniq.values(), key=lambda p: p.stat().st_mtime)


def _seg_has_part(job: Path, job_name: str, sid: str) -> bool:
    """True if local parts/XX.mp4 exists or Comfy already wrote one under output/."""
    local = _part_mp4(job, sid)
    if local.exists() and local.stat().st_size > 0:
        return True
    return bool(_find_output_parts(job_name, sid))


def _run_all_segments(
    job_name: str,
    *,
    pad_percent: float = 20.0,
    seed: int = 210,
    only_missing_wan: bool = True,
    queue_assemble_after: bool = True,
    comfy_url: str = "http://127.0.0.1:8188",
    seg_indices: list[int] | None = None,
    test_seconds: float = 0,
    face_images_prep: str = "",
    speed_profile: str = "",
) -> dict:
    """Build + enqueue Wan for every SEG that has a master (FIFO). Optionally assemble last."""
    job = _job_root(job_name)
    cuts_path = job / "cuts.json"
    if not cuts_path.exists():
        raise FileNotFoundError(f"No cuts.json — run Z1 Detect first ({job})")
    cuts = json.loads(cuts_path.read_text(encoding="utf-8"))
    segs = cuts["segments"]
    cfg_path = job / "job.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {"name": job_name}
    cfg["name"] = job_name
    cfg = _resolve_job_videos(job, cuts, cfg)

    (job / "parts").mkdir(parents=True, exist_ok=True)
    queued_ids: list[str] = []
    skipped: list[str] = []
    missing_masters: list[str] = []
    wanted = None if seg_indices is None else {int(i) for i in seg_indices}

    for i, seg in enumerate(segs):
        if wanted is not None and i not in wanted:
            continue
        sid = seg["id"]
        master = job / "masters" / f"{sid}.png"
        if not master.exists():
            missing_masters.append(sid)
            skipped.append(f"{sid}:no_master")
            continue
        _assign_unlocked_mode(
            seg,
            _input_video(cfg["comfy_driver"]),
            _input_video(cfg["comfy_mask_rect"]),
        )
        mode = _resolve_wan_mode(seg)
        seg_run = seg
        if test_seconds and float(test_seconds) > 0:
            seg_run = _clip_seg_for_test(seg, float(cfg.get("fps") or 24), float(test_seconds))
        prep = str(face_images_prep or "").strip()
        speed = str(speed_profile or "").strip()
        if prep or speed:
            if seg_run is seg:
                seg_run = dict(seg)
            if prep:
                seg_run["face_images_prep"] = prep
            if speed:
                seg_run["speed_profile"] = speed
            elif _is_fast_profile(seg_run):
                seg_run["speed_profile"] = "fast"
        if only_missing_wan and not seg_run.get("test_preview") and _seg_has_part(job, job_name, sid):
            # An old Animate part must not block the VACE graph.
            if not (mode == "vace" and not _output_part_is_vace(job_name, sid)):
                skipped.append(f"{sid}:already_has_part")
                continue
        seed_i = 231 if mode == "vace" else int(seed) + i
        wf_path = _build_seg_workflow(
            job, cfg, seg_run, seed=seed_i, pad_percent=float(pad_percent)
        )
        if wf_path.suffix.lower() != ".json":
            skipped.append(f"{sid}:passthrough_original")
            continue
        wf = json.loads(wf_path.read_text(encoding="utf-8"))
        queued_ids.append(f"{sid}:" + _queue_prompt(comfy_url, wf))

    cuts["segments"] = segs
    cuts_path.write_text(json.dumps(cuts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if queue_assemble_after and not (test_seconds and float(test_seconds) > 0):
        assemble_wf = {
            "1": {
                "class_type": "CVFSAssembleJob",
                "inputs": {
                    "job_name": job_name,
                    "from_output": str(Path(folder_paths.get_output_directory())),
                },
            }
        }
        try:
            queued_ids.append("assemble:" + _queue_prompt(comfy_url, assemble_wf))
        except Exception as e:
            queued_ids.append(f"assemble_enqueue_failed:{e}")

    return {
        "job": job_name,
        "pad_percent": float(pad_percent),
        "queued": queued_ids,
        "skipped": skipped,
        "missing_masters": missing_masters,
        "segments_total": len(segs),
        "final_will_be": str(job / "final.mp4"),
        "note": "Comfy queue runs SEG prompts FIFO, then Assemble. Empty-matte SEGs stay original.",
    }


def _run_masters_batch(
    frames,
    job_name: str,
    seed: int,
    queue_wan: bool,
    queue_assemble_after: bool,
    comfy_url: str,
    driver_filename: str = "",
    mask_filename: str = "",
    source_names: list[str] | None = None,
    replace_existing_masters: bool = True,
    only_missing_wan: bool = True,
    pad_percent: float = 20.0,
):
    import cv2

    # list[RGB] of any sizes, or NHWC batch ndarray
    if isinstance(frames, np.ndarray):
        frame_list = [frames[i] for i in range(int(frames.shape[0]))]
    else:
        frame_list = list(frames)

    job = _job_root(job_name)
    cuts_path = job / "cuts.json"
    if not cuts_path.exists():
        raise FileNotFoundError(f"No cuts.json in {job} — run Detect Scene Cuts first (same job_name)")
    cuts = json.loads(cuts_path.read_text(encoding="utf-8"))
    segs = cuts["segments"]
    cfg_path = job / "job.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {"name": job_name}
    cfg = _resolve_job_videos(job, cuts, cfg, driver_filename, mask_filename)
    cfg["name"] = job_name
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    n = min(len(segs), len(frame_list))
    (job / "masters").mkdir(parents=True, exist_ok=True)
    (job / "parts").mkdir(parents=True, exist_ok=True)
    saved = []
    skipped_master = []
    for i in range(n):
        sid = segs[i]["id"]
        path = job / "masters" / f"{sid}.png"
        src = source_names[i] if source_names and i < len(source_names) else f"batch[{i}]"
        wrote = False
        if path.exists() and not replace_existing_masters:
            skipped_master.append(sid)
        else:
            fr = _normalize_rgb_u8(frame_list[i])
            cv2.imwrite(str(path), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            wrote = True
            segs[i]["master_source"] = src
        wire = {
            "order": i,
            "seg": sid,
            "master_file": src,
            "saved_as": f"masters/{sid}.png",
            "master_written": wrote,
            "driver": cfg["comfy_driver"],
            "driver_frames": f"{segs[i]['start']}-{segs[i]['end'] - 1}",
            "n_frames": segs[i]["n_frames"],
            "wan_workflow": f"workflows/seg_{sid}_API.json",
            "part_exists": _part_mp4(job, sid).exists(),
            "how": (
                f"картинка #{i} ({src}) → SEG{sid} → кусок {cfg['comfy_driver']} "
                f"кадры {segs[i]['start']}…{segs[i]['end'] - 1} (+ та же маска)"
            ),
        }
        saved.append(wire)
        if _part_mp4(job, sid).exists():
            segs[i]["status"] = "wan_done"
        else:
            segs[i]["status"] = "master_bound"
        segs[i]["master"] = f"masters/{sid}.png"
        _build_seg_workflow(job, cfg, segs[i], seed=seed + i, pad_percent=float(pad_percent))

    cuts["segments"] = segs
    cuts_path.write_text(json.dumps(cuts, indent=2, ensure_ascii=False), encoding="utf-8")

    queued_ids = []
    skipped_wan = []
    if queue_wan:
        for i in range(n):
            sid = segs[i]["id"]
            if segs[i].get("passthrough"):
                skipped_wan.append(f"{sid}:passthrough_original")
                continue
            if only_missing_wan and _seg_has_part(job, job_name, sid):
                skipped_wan.append(sid)
                continue
            wf_path = job / "workflows" / f"seg_{sid}_API.json"
            if not wf_path.exists():
                skipped_wan.append(f"{sid}:no_workflow")
                continue
            wf = json.loads(wf_path.read_text(encoding="utf-8"))
            try:
                queued_ids.append(f"{sid}:" + _queue_prompt(comfy_url, wf))
            except Exception as e:
                raise RuntimeError(f"Failed to queue seg {sid}: {e}") from e

    if queue_assemble_after:
        assemble_wf = {
            "1": {
                "class_type": "CVFSAssembleJob",
                "inputs": {
                    "job_name": job_name,
                    "from_output": str(Path(folder_paths.get_output_directory())),
                },
            }
        }
        try:
            queued_ids.append("assemble:" + _queue_prompt(comfy_url, assemble_wf))
        except Exception as e:
            queued_ids.append(f"assemble_enqueue_failed:{e}")

    report = {
        "job": job_name,
        "driver": cfg["comfy_driver"],
        "mask": cfg["comfy_mask_rect"],
        "порядок": "000_/001_/… префиксы аплоада → #0=SEG00…",
        "связка_с_видео": saved,
        "skipped_master_keep": skipped_master,
        "skipped_wan_already_done": skipped_wan,
        "segments_total": len(segs),
        "masters_given": len(frame_list),
        "queued": queued_ids,
        "final_will_be": str(job / "final.mp4"),
        "note": "Z3 Assemble отдельно (после того как Wan допишет parts/). only_missing_wan=true не трогает готовые SEG.",
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    return {
        "ui": {"text": [text]},
        "result": (text, len(queued_ids)),
        "bound": saved,
        "segs": segs[:n],
        "n": n,
    }


class CVFSRunFromMastersInOrder:
    """IMAGE batch in order = SEG00, SEG01, … (prefer CVFSRunFromMastersFolder for 100 scenes)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masters_in_order": ("IMAGE",),
                "job_name": ("STRING", {"default": "comfy_detect_demo"}),
                "seed": ("INT", {"default": 210, "min": 0, "max": 0xFFFFFFFF}),
                "queue_wan": ("BOOLEAN", {"default": True}),
                "queue_assemble_after": ("BOOLEAN", {"default": True}),
                "comfy_url": ("STRING", {"default": "http://127.0.0.1:8188"}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("report", "queued")
    FUNCTION = "run"
    CATEGORY = "cut_video_face_swapper"
    OUTPUT_NODE = True

    def run(
        self,
        masters_in_order,
        job_name,
        seed,
        queue_wan,
        queue_assemble_after,
        comfy_url,
    ):
        batch = _tensor_batch_to_uint8(masters_in_order)
        return _run_masters_batch(
            batch,
            job_name,
            seed,
            queue_wan,
            queue_assemble_after,
            comfy_url,
        )


class CVFSRunFromMastersFolder:
    """ONE node: upload a pack of masters (button in UI) OR path to folder.

    Button «Upload masters pack» = multi-select in OS dialog → auto folder under
    input/cvfs_master_packs/… Sorted by name → SEG 00,01,… → rename internally →
    queue Wan per segment → assemble.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "job_name": ("STRING", {"default": "comfy_detect_demo"}),
                "masters_folder": (
                    "STRING",
                    {
                        "default": "cvfs_master_packs",
                        "tooltip": "Set automatically by «Upload masters pack», or type any input/ subfolder.",
                    },
                ),
                "seed": ("INT", {"default": 210, "min": 0, "max": 0xFFFFFFFF}),
                "queue_wan": ("BOOLEAN", {"default": True}),
                "only_missing_wan": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Не ставить в очередь SEG, у которых уже есть parts/XX.mp4",
                    },
                ),
                "replace_existing_masters": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "False = не перезаписывать masters/XX.png если уже есть",
                    },
                ),
                "queue_assemble_after": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "True = после всех SEG поставить Assemble в ту же очередь Comfy",
                    },
                ),
                "comfy_url": ("STRING", {"default": "http://127.0.0.1:8188"}),
                "pad_percent": (
                    "FLOAT",
                    {
                        "default": 20.0,
                        "min": 0.0,
                        "max": 80.0,
                        "step": 1.0,
                        "tooltip": "Запас вокруг маски при queue_wan / сборке кропов",
                    },
                ),
            },
            "optional": {
                "video_frames": ("IMAGE",),
                "mask_frames": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "IMAGE")
    RETURN_NAMES = ("report", "queued", "masters_preview_batch")
    FUNCTION = "run"
    CATEGORY = "cut_video_face_swapper"
    OUTPUT_NODE = True

    def run(
        self,
        job_name,
        masters_folder,
        seed,
        queue_wan,
        only_missing_wan,
        replace_existing_masters,
        queue_assemble_after,
        comfy_url,
        pad_percent=20.0,
        video_frames=None,
        mask_frames=None,
    ):
        import cv2

        if video_frames is not None:
            _ensure_job_ready_from_frames(job_name, video_frames, mask_frames)

        input_root = Path(folder_paths.get_input_directory())
        folder = Path(masters_folder)
        if not folder.is_absolute():
            folder = input_root / folder
        folder.mkdir(parents=True, exist_ok=True)
        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        files = sorted([p for p in folder.iterdir() if p.suffix.lower() in exts], key=lambda p: p.name.lower())
        if not files:
            raise FileNotFoundError(
                f"Папка пустая: {folder}. Кинь туда все master-кадры по порядку имён (01.png, 02.png…)."
            )
        frames = []
        names = []
        for p in files:
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if im is None:
                continue
            frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            names.append(p.name)
        if not frames:
            raise FileNotFoundError(f"Не удалось прочитать картинки в {folder}")
        # Masters may have different HxW (Gemini vs texture etc.) — never np.stack raw.
        result = _run_masters_batch(
            frames,
            job_name,
            seed,
            queue_wan,
            queue_assemble_after,
            comfy_url,
            source_names=names,
            replace_existing_masters=replace_existing_masters,
            only_missing_wan=only_missing_wan,
            pad_percent=float(pad_percent),
        )
        preview = _uint8_to_tensor_batch(frames)
        text, queued = result["result"]
        return {"ui": result.get("ui", {}), "result": (text, queued, preview)}


def _vace_context_feather(h: int, w: int, left_frac: float = 0.06) -> np.ndarray:
    """Same paste as seg02_full: keep the left context strip, replace the rest."""
    import cv2

    alpha = np.zeros((h, w), np.float32)
    rx0 = int(w * left_frac)
    if rx0 < w:
        alpha[:, rx0:] = 1.0
    sigma = max(1.0, 4.0 * w / 544.0)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigma)
    return alpha[:, :, None]


def _edge_feather(h: int, w: int, border: int = 6) -> np.ndarray:
    feather = np.ones((h, w), np.float32)
    border = max(1, min(border, h // 2, w // 2))
    for b in range(border):
        alpha = b / float(border)
        feather[b, :] = np.minimum(feather[b, :], alpha)
        feather[h - 1 - b, :] = np.minimum(feather[h - 1 - b, :], alpha)
        feather[:, b] = np.minimum(feather[:, b], alpha)
        feather[:, w - 1 - b] = np.minimum(feather[:, w - 1 - b], alpha)
    return feather[:, :, None]


def _fit_paste_crop(
    crop: list[int], part_w: int, part_h: int, plate_w: int, plate_h: int
) -> list[int]:
    """Keep crop center; resize the paste box to the Wan part aspect (no stretch/squash)."""
    x0, y0, x1, y1 = [int(v) for v in crop]
    cw, ch = max(1, x1 - x0), max(1, y1 - y0)
    part_w, part_h = max(1, int(part_w)), max(1, int(part_h))
    part_ar = part_w / float(part_h)
    crop_ar = cw / float(ch)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    if abs(part_ar - crop_ar) < 0.03:
        nw, nh = cw, ch
    elif part_ar > crop_ar:
        nw, nh = cw, max(1, int(round(cw / part_ar)))
    else:
        nh, nw = ch, max(1, int(round(ch * part_ar)))
    nx0 = int(round(cx - nw / 2.0))
    ny0 = int(round(cy - nh / 2.0))
    nx0 = max(0, min(max(0, plate_w - nw), nx0))
    ny0 = max(0, min(max(0, plate_h - nh), ny0))
    return [nx0, ny0, nx0 + nw, ny0 + nh]


def _match_colors_border(src_bgr: np.ndarray, ref_bgr: np.ndarray, border: int = 24) -> np.ndarray:
    """Reinhard LAB match using only the rim, so the insert matches plate light without washing the face."""
    import cv2

    if src_bgr.shape[:2] != ref_bgr.shape[:2]:
        ref_bgr = cv2.resize(ref_bgr, (src_bgr.shape[1], src_bgr.shape[0]), interpolation=cv2.INTER_AREA)
    h, w = src_bgr.shape[:2]
    b = max(2, min(int(border), h // 4, w // 4))
    ring = np.zeros((h, w), np.bool_)
    ring[:b, :] = True
    ring[-b:, :] = True
    ring[:, :b] = True
    ring[:, -b:] = True
    src = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = src.copy()
    for c in range(3):
        s = src[:, :, c][ring]
        r = ref[:, :, c][ring]
        if s.size < 16 or r.size < 16:
            continue
        s_mean, s_std = float(s.mean()), float(s.std()) + 1e-5
        r_mean, r_std = float(r.mean()), float(r.std()) + 1e-5
        out[:, :, c] = (src[:, :, c] - s_mean) * (r_std / s_std) + r_mean
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def _paste_crop(
    plate_bgr: np.ndarray,
    diff_bgr: np.ndarray,
    crop: list[int],
    feather: np.ndarray,
    *,
    color_match: bool = True,
) -> np.ndarray:
    import cv2

    x0, y0, x1, y1 = [int(v) for v in crop]
    ch, cw = y1 - y0, x1 - x0
    if ch <= 0 or cw <= 0:
        return plate_bgr
    diff = diff_bgr
    if diff.shape[1] != cw or diff.shape[0] != ch:
        diff = cv2.resize(diff, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
    if feather.shape[0] != ch or feather.shape[1] != cw:
        feather = _edge_feather(ch, cw)
    out = plate_bgr.copy()
    base = out[y0:y1, x0:x1]
    if color_match:
        diff = _match_colors_border(diff, base, border=max(12, min(cw, ch) // 12))
    blended = diff.astype(np.float32) * feather + base.astype(np.float32) * (1.0 - feather)
    out[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def _seg_by_id(cuts: dict, seg_id: str) -> dict:
    for s in cuts.get("segments") or []:
        if str(s.get("id")) == str(seg_id):
            return s
    raise KeyError(f"SEG {seg_id} not in cuts.json")


def _paste_vace_full_bgr(job_name: str, seg_id: str, patches_bgr: list[np.ndarray]) -> list[np.ndarray]:
    """Drop the VACE crop back onto the full driver, same blend as seg02_full."""
    import cv2

    job = _job_root(job_name)
    cuts = json.loads((job / "cuts.json").read_text(encoding="utf-8"))
    seg = _seg_by_id(cuts, seg_id)
    crop = [int(v) for v in seg["wan_crop"]]
    n = int(seg["end"]) - int(seg["start"])
    driver_name, _mask_name = _prefer_job_media(job_name, cuts)
    driver = _input_video(driver_name)
    cap = cv2.VideoCapture(str(driver))
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(seg["start"]))
    x0, y0, x1, y1 = crop
    feather = _vace_context_feather(y1 - y0, x1 - x0)
    full = []
    for i in range(n):
        ok, fr = cap.read()
        if not ok:
            break
        patch = patches_bgr[min(i, len(patches_bgr) - 1)]
        full.append(_paste_crop(fr, patch, crop, feather))
    cap.release()
    if len(full) != n:
        raise RuntimeError(f"SEG{seg_id}: pasted {len(full)} frames, need {n}")
    return full


def _write_vace_full_mp4(job_name: str, seg_id: str, frames_bgr: list[np.ndarray]) -> Path:
    """Full-frame mp4 with the driver's audio, next to the crop part."""
    import subprocess

    job = _job_root(job_name)
    cuts = json.loads((job / "cuts.json").read_text(encoding="utf-8"))
    seg = _seg_by_id(cuts, seg_id)
    cfg = json.loads((job / "job.json").read_text(encoding="utf-8")) if (job / "job.json").exists() else {}
    fps = float(cfg.get("fps") or 24.0)
    driver_name, _mask_name = _prefer_job_media(job_name, cuts, cfg)
    driver = _input_video(driver_name)
    out_dir = Path(folder_paths.get_output_directory()) / "cut_video_face_swapper" / job_name / "vace_full"
    out_dir.mkdir(parents=True, exist_ok=True)
    silent = out_dir / f"{seg_id}_silent.mp4"
    final = out_dir / f"{seg_id}.mp4"
    _write_bgr_mp4(frames_bgr, silent, fps)
    start = int(seg["start"]) / fps
    dur = len(frames_bgr) / fps
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(silent),
            "-ss", f"{start:.6f}", "-t", f"{dur:.6f}", "-i", str(driver),
            "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            str(final),
        ],
        check=True,
    )
    return final


class CVFSPasteVacePlate:
    """Paste a VACE crop back into the full driver frames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "job_name": ("STRING", {"default": "comfy_detect_demo"}),
                "seg_id": ("STRING", {"default": "02"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "cut_video_face_swapper"

    def run(self, images, job_name, seg_id):
        import cv2

        batch = _tensor_batch_to_uint8(images)
        bgr = [cv2.cvtColor(fr, cv2.COLOR_RGB2BGR) for fr in batch]
        full = _paste_vace_full_bgr(job_name, seg_id, bgr)
        _write_vace_full_mp4(job_name, seg_id, full)
        rgb = [cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) for fr in full]
        return (_uint8_to_tensor_batch(rgb),)


class CVFSAssembleJob:
    """Paste each Wan 832×480 part back into the full driver, then concatenate."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "job_name": ("STRING", {"default": "comfy_detect_demo"}),
                "from_output": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("final_path",)
    FUNCTION = "run"
    CATEGORY = "cut_video_face_swapper"
    OUTPUT_NODE = True

    def run(self, job_name, from_output):
        import cv2
        import shutil

        job = _job_root(job_name)
        cuts = json.loads((job / "cuts.json").read_text(encoding="utf-8"))
        cfg = json.loads((job / "job.json").read_text(encoding="utf-8")) if (job / "job.json").exists() else {}
        cfg["name"] = job_name
        cfg = _resolve_job_videos(job, cuts, cfg)
        out_root = Path(from_output) if from_output.strip() else Path(folder_paths.get_output_directory())
        fps = float(cfg.get("fps") or 24)
        driver_name = cfg["comfy_driver"]
        driver_path = _input_video(driver_name)
        cap = cv2.VideoCapture(str(driver_path))
        if not cap.isOpened():
            raise FileNotFoundError(str(driver_path))
        plate_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        plate_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        parts_dir = job / "parts"
        parts_dir.mkdir(exist_ok=True)
        report_segs = []
        frames_out = []
        for s in cuts["segments"]:
            sid = s["id"]
            start, end = int(s["start"]), int(s["end"])
            n = end - start
            # Empty matte / no crop → keep original. Else paste Wan part (find under output/**/parts/).
            passthrough = (not s.get("wan_crop")) or (
                bool(s.get("passthrough")) and s.get("wan_crop_note") == "passthrough_original"
            )
            local = parts_dir / f"{sid}.mp4"
            if not passthrough:
                hits = _find_output_parts(job_name, sid)
                pick = None
                crop = s.get("wan_crop")
                if hits and crop:
                    tw = max(1, int(crop[2]) - int(crop[0]))
                    th = max(1, int(crop[3]) - int(crop[1]))
                    target = tw / float(th)

                    def _aspect(p: Path) -> float:
                        import cv2 as _cv

                        c = _cv.VideoCapture(str(p))
                        w, h = float(c.get(_cv.CAP_PROP_FRAME_WIDTH) or 1), float(c.get(_cv.CAP_PROP_FRAME_HEIGHT) or 1)
                        c.release()
                        return w / max(1.0, h)

                    best = min(abs(_aspect(p) - target) for p in hits)
                    close = [p for p in hits if abs(_aspect(p) - target) <= best + 0.02]
                    pick = max(close, key=lambda p: p.stat().st_mtime)
                elif hits:
                    pick = hits[-1]
                if pick is not None:
                    shutil.copy2(pick, local)
                elif not local.exists():
                    passthrough = True
                # refresh local existence after copy
                if not local.exists():
                    passthrough = True
            diff_frames = []
            if not passthrough:
                pc = cv2.VideoCapture(str(local))
                while True:
                    ok, fr = pc.read()
                    if not ok:
                        break
                    diff_frames.append(fr)
                pc.release()
                if not diff_frames:
                    passthrough = True
            feather = None
            crop = s.get("wan_crop")
            track = s.get("wan_track") if isinstance(s.get("wan_track"), list) else None
            paste_crop = crop
            if crop and not passthrough and diff_frames and not track:
                ph, pw = diff_frames[0].shape[:2]
                paste_crop = _fit_paste_crop(crop, pw, ph, plate_w, plate_h)
                ch = int(paste_crop[3]) - int(paste_crop[1])
                cw = int(paste_crop[2]) - int(paste_crop[0])
                if str(s.get("wan_mode")) == "vace":
                    feather = _vace_context_feather(ch, cw)
                else:
                    # Soft rim so color-matched face crops don't show a hard rectangle.
                    feather = _edge_feather(ch, cw, max(12, min(cw, ch) // 24))
            elif track and not passthrough and diff_frames:
                box0 = [int(v) for v in track[0]]
                ch = int(box0[3]) - int(box0[1])
                cw = int(box0[2]) - int(box0[0])
                feather = _edge_feather(ch, cw, max(12, min(cw, ch) // 24))
                paste_crop = box0
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))
            for i in range(n):
                ok, orig = cap.read()
                if not ok:
                    break
                if passthrough or feather is None or (paste_crop is None and not track):
                    frames_out.append(orig)
                else:
                    diff = diff_frames[min(i, len(diff_frames) - 1)]
                    box = paste_crop
                    if track:
                        box = [int(v) for v in track[min(i, len(track) - 1)]]
                        bh = int(box[3]) - int(box[1])
                        bw = int(box[2]) - int(box[0])
                        if feather.shape[0] != bh or feather.shape[1] != bw:
                            feather = _edge_feather(bh, bw, max(12, min(bw, bh) // 24))
                    frames_out.append(_paste_crop(orig, diff, box, feather))
            s["status"] = "passthrough" if passthrough else "assembled"
            report_segs.append(
                {"seg": sid, "mode": s["status"], "frames": n, "crop": crop, "paste_crop": paste_crop}
            )
        cap.release()
        final = job / "final.mp4"
        wr = cv2.VideoWriter(str(final), cv2.VideoWriter_fourcc(*"mp4v"), fps, (plate_w, plate_h))
        for fr in frames_out:
            if fr.shape[1] != plate_w or fr.shape[0] != plate_h:
                fr = cv2.resize(fr, (plate_w, plate_h))
            wr.write(fr)
        wr.release()
        (job / "cuts.json").write_text(json.dumps(cuts, indent=2, ensure_ascii=False), encoding="utf-8")
        msg = json.dumps(
            {"final": str(final), "frames": len(frames_out), "size": [plate_w, plate_h], "segments": report_segs},
            indent=2,
            ensure_ascii=False,
        )
        return {"ui": {"text": [msg]}, "result": (str(final),)}


class CVFSZoneSelect:
    """Pick which zone runs. JS mutes other [Z1]/[Z2]/[Z3] nodes so Queue only hits that stage."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "zone": (
                    ["1_detect", "2_masters_wan", "3_assemble"],
                    {"default": "1_detect"},
                ),
                "job_name": ("STRING", {"default": "comfy_detect_demo"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("zone", "job_name")
    FUNCTION = "run"
    CATEGORY = "cut_video_face_swapper"
    OUTPUT_NODE = True

    def run(self, zone, job_name):
        job = _job_root(job_name)
        cuts_path = job / "cuts.json"
        status = []
        if cuts_path.exists():
            cuts = json.loads(cuts_path.read_text(encoding="utf-8"))
            for s in cuts.get("segments", []):
                sid = s["id"]
                master_ok = (job / "masters" / f"{sid}.png").exists()
                part_ok = _part_mp4(job, sid).exists()
                status.append(
                    {
                        "seg": sid,
                        "frames": f"{s.get('start')}-{int(s.get('end', 0)) - 1}",
                        "status": s.get("status"),
                        "master": master_ok,
                        "part": part_ok,
                    }
                )
        hint = {
            "1_detect": "Z1: Load driver+mask → Detect. Потом Gemini masters.",
            "2_masters_wan": (
                "Z2: Upload masters → SEG picker Apply → Queue. "
                "На холсте полный Wan (видно прогресс). Потом seg_index+1 и снова Queue."
            ),
            "3_assemble": "Z3: вклеить 832×480 обратно в полный кадр и склеить final.mp4.",
        }.get(zone, "")
        report = {
            "active_zone": zone,
            "job_name": job_name,
            "hint": hint,
            "checkpoint": status,
            "final": str(job / "final.mp4") if (job / "final.mp4").exists() else None,
            "ui": "Смени zone → ноды других зон Mute. Потом Queue.",
        }
        text = json.dumps(report, indent=2, ensure_ascii=False)
        return {"ui": {"text": [text]}, "result": (zone, job_name)}


def _seg_info(
    job_name: str,
    seg_index: int,
    pad_percent: float = 20.0,
    wan_mode: str | None = None,
) -> dict:
    job = _job_root(job_name)
    cuts_path = job / "cuts.json"
    if not cuts_path.exists():
        raise FileNotFoundError(f"No cuts.json — run Z1 Detect first ({job})")
    cuts = json.loads(cuts_path.read_text(encoding="utf-8"))
    segs = cuts.get("segments") or []
    if not segs:
        raise ValueError("cuts.json has no segments")
    i = int(seg_index)
    if i < 0 or i >= len(segs):
        raise IndexError(f"seg_index {i} out of range 0..{len(segs)-1}")
    s = segs[i]
    sid = s["id"]
    cfg = json.loads((job / "job.json").read_text(encoding="utf-8")) if (job / "job.json").exists() else {}
    cfg["name"] = job_name
    cfg = _resolve_job_videos(job, cuts, cfg)
    driver, mask = cfg["comfy_driver"], cfg["comfy_mask_rect"]
    if wan_mode is not None and str(wan_mode).strip() and str(wan_mode).strip().lower() not in {"saved", "from_menu", "auto"}:
        s["wan_mode"] = _resolve_wan_mode({"wan_mode": wan_mode})
        s["wan_mode_locked"] = True
    else:
        _assign_unlocked_mode(s, _input_video(driver), _input_video(mask))
    mode = _resolve_wan_mode(s)
    s["wan_mode"] = mode if mode in {"face", "vace", "hair_back", "face_gate"} else "face"
    master_plate = job / "masters" / f"{sid}.png"
    if mode == "vace":
        prep = _prepare_seg_vace
    elif mode == "hair_back":
        prep = _prepare_seg_wan_crop_hair
    elif mode == "face_gate":
        prep = _prepare_seg_wan_crop_face_gate
    else:
        prep = _prepare_seg_wan_crop
    crop = prep(
        job,
        job_name,
        s,
        driver,
        mask,
        tw=WAN_CROP_W,
        th=WAN_CROP_H,
        fps=float(cfg.get("fps") or 24.0),
        pad_percent=float(pad_percent),
    )
    s["wan_crop"] = crop.get("crop")
    s["wan_crop_note"] = crop.get("crop_note")
    s["passthrough"] = bool(crop.get("passthrough"))
    if crop.get("width") and crop.get("height"):
        s["wan_size"] = [int(crop["width"]), int(crop["height"])]
    if mode == "vace" and not crop.get("passthrough"):
        _build_vace_seg_workflow(
            job,
            {"name": job_name, "comfy_driver": driver, "comfy_mask_rect": mask, "fps": float(cfg.get("fps") or 24.0)},
            s,
            seed=231,
            crop_info=crop,
            pad_percent=float(pad_percent),
        )
    cuts["segments"][i] = s
    cuts_path.write_text(json.dumps(cuts, indent=2, ensure_ascii=False), encoding="utf-8")
    how = (
        f"SEG{sid}: matte пустой — в final останется оригинальный кадр."
        if crop.get("passthrough")
        else (
            f"SEG{sid} [{mode}]: маска + {pad_percent:g}% → окно {crop['crop']} "
            f"({crop['window_src'][0]}×{crop['window_src'][1]}), Wan {crop['width']}×{crop['height']}"
            + (f", clamp_x={crop.get('clamp_x')}" if crop.get("clamp_x") else "")
            + "."
        )
    )
    if mode == "vace":
        face_strength = 0.0
        pose_strength = 0.0
    elif mode == "hair_back":
        face_strength = float(s.get("face_strength", 0.0))
        pose_strength = float(s.get("pose_strength", 0.25))
    elif mode == "face_gate":
        face_strength = float(s.get("face_strength", 1.0))
        pose_strength = float(s.get("pose_strength", 0.55))
    else:
        face_strength = 1.0
        pose_strength = 1.0
    return {
        "job": job_name,
        "seg_index": i,
        "seg_id": sid,
        "wan_mode": mode,
        "face_strength": face_strength,
        "pose_strength": pose_strength,
        "prepare_branch": crop.get("prepare_branch"),
        "face_probe_ratio": crop.get("face_probe_ratio"),
        "driver": crop["driver"],
        "mask": crop["mask"],
        "skip_first_frames": int(crop.get("skip_first_frames") or 0),
        "frame_load_cap": crop["n_frames"],
        "n_frames": crop["n_frames"],
        "raw_n_frames": crop["raw_n_frames"],
        "width": crop["width"],
        "height": crop["height"],
        "crop": crop.get("crop"),
        "crop_note": crop.get("crop_note"),
        "clamp_x": crop.get("clamp_x"),
        "passthrough": bool(crop.get("passthrough")),
        "matte_empty": crop.get("matte_empty", False),
        "plate": crop["plate"],
        "master_image": crop["master_image"],
        "master_exists": master_plate.exists() and (crop["master_exists"] or crop.get("passthrough")),
        "part_exists": _part_mp4(job, sid).exists(),
        "filename_prefix": f"cut_video_face_swapper/{job_name}/parts/{sid}",
        "seed": 210 + i,
        "num_segments": len(segs),
        "how": how,
    }


def set_seg_wan_mode(job_name: str, seg_index: int, wan_mode: str, locked: bool = True) -> dict:
    """Persist wan_mode on one SEG. Manual edits are locked against the auto probe."""
    job = _job_root(job_name)
    cuts_path = job / "cuts.json"
    if not cuts_path.exists():
        raise FileNotFoundError(f"No cuts.json — run Z1 Detect first ({job})")
    cuts = json.loads(cuts_path.read_text(encoding="utf-8"))
    segs = cuts.get("segments") or []
    i = int(seg_index)
    if i < 0 or i >= len(segs):
        raise IndexError(f"seg_index {i} out of range 0..{len(segs)-1}")
    mode = _resolve_wan_mode({"wan_mode": wan_mode})
    if mode not in {"face", "vace"}:
        raise ValueError("режим здесь только face или vace")
    segs[i]["wan_mode"] = mode
    segs[i]["wan_mode_locked"] = bool(locked)
    cuts["segments"] = segs
    cuts_path.write_text(json.dumps(cuts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"job": job_name, "seg_index": i, "seg_id": segs[i]["id"], "wan_mode": mode, "locked": bool(locked)}


def _probe_subject_face(driver_path: Path, mask_path: Path, start: int, n: int) -> float:
    """Fraction of samples where a face sits inside the matte. Low → back of head."""
    import cv2

    if n <= 0 or not driver_path.exists():
        return 0.0
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        return 0.0
    step = max(1, n // 8)
    cap_d = cv2.VideoCapture(str(driver_path))
    cap_m = cv2.VideoCapture(str(mask_path)) if mask_path.exists() else None
    hits = checked = 0
    for i in range(0, n, step):
        cap_d.set(cv2.CAP_PROP_POS_FRAMES, start + i)
        ok, fr = cap_d.read()
        if not ok:
            break
        checked += 1
        region = fr
        if cap_m is not None:
            cap_m.set(cv2.CAP_PROP_POS_FRAMES, start + i)
            okm, mk = cap_m.read()
            if okm:
                g = cv2.cvtColor(mk, cv2.COLOR_BGR2GRAY) if mk.ndim == 3 else mk
                ys, xs = np.where(g > 32)
                if len(xs):
                    x0, x1 = int(xs.min()), int(xs.max()) + 1
                    y0, y1 = int(ys.min()), int(ys.max()) + 1
                    region = fr[y0:y1, x0:x1]
        if region.size == 0:
            continue
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region
        found = cascade.detectMultiScale(gray, 1.08, 4, minSize=(40, 40))
        if found is not None and len(found) > 0:
            hits += 1
    cap_d.release()
    if cap_m is not None:
        cap_m.release()
    return (hits / checked) if checked else 0.0


def _assign_unlocked_mode(seg: dict, driver: Path, mask: Path, force: bool = False) -> str:
    """face if a face sits in the matte, otherwise the seg02_full VACE path. Locked stays."""
    if seg.get("wan_mode_locked"):
        mode = _resolve_wan_mode(seg)
        return mode if mode in {"face", "vace"} else "face"
    mode = _resolve_wan_mode(seg)
    if not force and mode in {"face", "vace"} and seg.get("face_probe_ratio") is not None:
        return mode
    n = int(seg["end"]) - int(seg["start"])
    ratio = _probe_subject_face(driver, mask, int(seg["start"]), n)
    mode = "face" if ratio >= 0.25 else "vace"
    seg["wan_mode"] = mode
    seg["face_probe_ratio"] = round(float(ratio), 3)
    return mode


def list_seg_modes(job_name: str, assign_auto: bool = True, force: bool = False) -> dict:
    """Modes for the master-order panel. Unlocked SEGs get face vs vace from a face probe."""
    job = _job_root(job_name)
    cuts_path = job / "cuts.json"
    if not cuts_path.exists():
        raise FileNotFoundError(f"No cuts.json — run Z1 Detect first ({job})")
    cuts = json.loads(cuts_path.read_text(encoding="utf-8"))
    driver_name, mask_name = _prefer_job_media(job_name, cuts)
    driver = _input_video(driver_name)
    mask = _input_video(mask_name)
    changed = False
    rows = []
    for i, s in enumerate(cuts.get("segments") or []):
        before = (s.get("wan_mode"), s.get("face_probe_ratio"))
        mode = _assign_unlocked_mode(s, driver, mask, force=force) if assign_auto else _resolve_wan_mode(s)
        if (s.get("wan_mode"), s.get("face_probe_ratio")) != before:
            changed = True
        rows.append(
            {
                "seg_index": i,
                "seg_id": s["id"],
                "frames": f"{s.get('start')}-{int(s.get('end', 0)) - 1}",
                "wan_mode": mode if mode in {"face", "vace"} else "face",
                "locked": bool(s.get("wan_mode_locked")),
                "face_probe_ratio": s.get("face_probe_ratio"),
                "master_exists": (job / "masters" / f"{s['id']}.png").exists(),
            }
        )
    if changed:
        cuts_path.write_text(json.dumps(cuts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"job": job_name, "segments": rows}


def set_seg_wan_modes(job_name: str, modes: list) -> dict:
    saved = []
    for item in modes:
        saved.append(
            set_seg_wan_mode(
                job_name,
                int(item.get("seg_index", item.get("index", 0))),
                item.get("wan_mode") or item.get("mode") or "face",
                locked=True,
            )
        )
    return {"job": job_name, "saved": saved}


class CVFSSegPicker:
    """Pick SEG index for the visible Wan graph on the canvas (no silent /prompt)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "job_name": ("STRING", {"default": "comfy_detect_demo"}),
                "seg_index": ("INT", {"default": 0, "min": 0, "max": 999}),
                "pad_percent": (
                    "FLOAT",
                    {
                        "default": 20.0,
                        "min": 0.0,
                        "max": 80.0,
                        "step": 1.0,
                        "tooltip": "Запас вокруг крайних точек маски, % с каждой стороны",
                    },
                ),
                "wan_mode": (
                    ["from_menu", "face", "vace", "hair_back", "face_gate"],
                    {
                        "default": "from_menu",
                        "tooltip": "Не используется. Режим берётся из окна порядка мастеров.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("seg_info_json", "seg_index")
    FUNCTION = "run"
    CATEGORY = "cut_video_face_swapper"
    OUTPUT_NODE = True

    def run(self, job_name, seg_index, pad_percent, wan_mode="from_menu"):
        del wan_mode
        info = _seg_info(
            job_name, seg_index, pad_percent=float(pad_percent), wan_mode=None
        )
        if not info["master_exists"]:
            raise FileNotFoundError(
                f"Нет masters/{info['seg_id']}.png — сначала Upload masters на [Z2]."
            )
        # Ensure seg workflow json stays in sync (optional debug / reload)
        if info.get("passthrough"):
            text = json.dumps(info, indent=2, ensure_ascii=False)
            return {"ui": {"text": [text]}, "result": (text, int(seg_index))}
        cfg_path = _job_root(job_name) / "job.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {"name": job_name}
        cuts = json.loads((_job_root(job_name) / "cuts.json").read_text(encoding="utf-8"))
        cfg = _resolve_job_videos(_job_root(job_name), cuts, cfg, info["driver"], info["mask"])
        cfg["name"] = job_name
        cfg["width"] = info["width"]
        cfg["height"] = info["height"]
        cfg["fps"] = float(cfg.get("fps") or 24.0)
        seg = cuts["segments"][int(seg_index)]
        seg["wan_mode"] = info.get("wan_mode") or "face"
        _build_seg_workflow(
            _job_root(job_name),
            cfg,
            seg,
            seed=210 + int(seg_index),
            crop_info={
                "passthrough": False,
                "driver": info["driver"],
                "mask": info["mask"],
                "master_image": info["master_image"],
                "width": info["width"],
                "height": info["height"],
                "n_frames": info["n_frames"],
                "crop": info.get("crop"),
                "crop_note": info.get("crop_note"),
            },
        )
        text = json.dumps(info, indent=2, ensure_ascii=False)
        return {"ui": {"text": [text]}, "result": (text, int(seg_index))}


class CVFSRectFromMask:
    """v1.4 wipe: padded AABB rect + temporal envelope from a face/hair matte. No prebaked square video."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "pad_x": ("INT", {"default": 32, "min": 0, "max": 512}),
                "pad_y": ("INT", {"default": 24, "min": 0, "max": 512}),
                "envelope": ("INT", {"default": 2, "min": 0, "max": 16}),
            },
            "optional": {
                "pad_x_ratio": ("FLOAT", {"default": FACE_PAD_X_RATIO, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("rect_mask",)
    FUNCTION = "run"
    CATEGORY = "cut_video_face_swapper"

    def run(self, mask, pad_x, pad_y, envelope, pad_x_ratio=FACE_PAD_X_RATIO):
        arr = mask.detach().cpu().numpy()
        if arr.ndim == 2:
            arr = arr[None, ...]
        u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        rect = _rect_from_matte_u8(
            u8,
            pad_x=int(pad_x),
            pad_y=int(pad_y),
            envelope=int(envelope),
            pad_x_ratio=float(pad_x_ratio),
        )
        t = torch.from_numpy(rect.astype(np.float32) / 255.0)
        return (t,)


class CVFSFeatherMask:
    """Soft hair/face matte: dilate then Gaussian feather. For hair_back — no hard wipe rect."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "expand": ("INT", {"default": 4, "min": 0, "max": 64, "step": 1}),
                "blur": ("INT", {"default": 7, "min": 0, "max": 63, "step": 2}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("soft_mask",)
    FUNCTION = "run"
    CATEGORY = "cut_video_face_swapper"

    def run(self, mask, expand=4, blur=7):
        import cv2

        arr = mask.detach().cpu().numpy()
        if arr.ndim == 2:
            arr = arr[None, ...]
        out = np.empty_like(arr, dtype=np.float32)
        k_exp = max(0, int(expand))
        k_blur = max(0, int(blur))
        if k_blur > 0 and k_blur % 2 == 0:
            k_blur += 1
        kernel = None
        if k_exp > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_exp * 2 + 1, k_exp * 2 + 1))
        for i in range(arr.shape[0]):
            m = np.clip(arr[i], 0.0, 1.0).astype(np.float32)
            if kernel is not None:
                m = cv2.dilate(m, kernel)
            if k_blur > 0:
                m = cv2.GaussianBlur(m, (k_blur, k_blur), 0)
            out[i] = np.clip(m, 0.0, 1.0)
        return (torch.from_numpy(out),)


class CVFSBlendMasked:
    """Blend ref still into video under a soft mask — V2V init for hair identity without high denoise."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE",),
                "ref": ("IMAGE",),
                "mask": ("MASK",),
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.55,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "0=keep video, 1=full ref in masked area",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended",)
    FUNCTION = "run"
    CATEGORY = "cut_video_face_swapper"

    def run(self, video, ref, mask, strength=0.55):
        import torch.nn.functional as F

        vid = video
        if vid.ndim != 4:
            raise ValueError(f"video expected BHWC, got {tuple(vid.shape)}")
        ref0 = ref[0:1]
        if ref0.shape[1] != vid.shape[1] or ref0.shape[2] != vid.shape[2]:
            # BHWC -> BCHW for interpolate
            r = ref0.movedim(-1, 1).float()
            r = F.interpolate(r, size=(vid.shape[1], vid.shape[2]), mode="bilinear", align_corners=False)
            ref0 = r.movedim(1, -1)
        ref_b = ref0.expand(vid.shape[0], -1, -1, -1)
        m = mask
        if m.ndim == 2:
            m = m[None, ...]
        if m.shape[0] == 1 and vid.shape[0] > 1:
            m = m.expand(vid.shape[0], -1, -1)
        elif m.shape[0] != vid.shape[0]:
            # repeat/truncate
            if m.shape[0] < vid.shape[0]:
                reps = (vid.shape[0] + m.shape[0] - 1) // m.shape[0]
                m = m.repeat(reps, 1, 1)[: vid.shape[0]]
            else:
                m = m[: vid.shape[0]]
        if m.shape[1] != vid.shape[1] or m.shape[2] != vid.shape[2]:
            mm = m[:, None, ...].float()
            mm = F.interpolate(mm, size=(vid.shape[1], vid.shape[2]), mode="bilinear", align_corners=False)
            m = mm[:, 0]
        alpha = (m.clamp(0, 1) * float(strength))[..., None]
        out = vid.float() * (1.0 - alpha) + ref_b.float() * alpha
        return (out.clamp(0, 1),)


class CVFSCropRefInRect:
    """v1.4: master → crop to wipe rect → cover-fill 512 (identity face/hair)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "size": ("INT", {"default": 512, "min": 64, "max": 1024, "step": 16}),
            },
            "optional": {
                "job_name": ("STRING", {"default": ""}),
                "seg_index": ("INT", {"default": 0, "min": 0, "max": 999}),
                "ref_index": ("INT", {"default": -1, "min": -1, "max": 99999}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("ref_512",)
    FUNCTION = "run"
    CATEGORY = "cut_video_face_swapper"

    def run(self, image, mask, size, job_name="", seg_index=0, ref_index=-1):
        img = _tensor_batch_to_uint8(image)[0]
        m = mask.detach().cpu().numpy()
        if m.ndim == 2:
            m = m[None, ...]
        m_u8 = np.clip(m * 255.0, 0, 255).astype(np.uint8)
        ih, iw = img.shape[:2]
        rect = None
        # Prefer the wipe at the picked master frame inside THIS clip (test or full).
        if int(ref_index) >= 0 and m_u8.ndim >= 3:
            idx = min(int(ref_index), m_u8.shape[0] - 1)
            rect = m_u8[idx]
            if rect.ndim == 3:
                rect = rect[..., 0]
        if rect is None:
            rect = _picked_matte_rect(job_name, int(seg_index))
        # Disk rect from an older window size would crop the wrong part of this master.
        if rect is None or rect.shape[0] != ih or rect.shape[1] != iw:
            rect = _largest_mask_frame(m_u8)
        out = _crop_ref_in_rect(img, rect, size=int(size))
        t = torch.from_numpy(out.astype(np.float32) / 255.0)[None, ...]
        return (t,)


def _face_kp_conf(meta) -> np.ndarray | None:
    """Face-keypoint confidences from a VitPose meta (dict or AAPoseMeta)."""
    if isinstance(meta, dict) and "keypoints_face" in meta:
        kpf = np.asarray(meta["keypoints_face"], dtype=np.float32)
        if kpf.ndim == 2 and kpf.shape[1] >= 3:
            return kpf[:, 2]
        return None
    if hasattr(meta, "kps_face_p") and meta.kps_face_p is not None:
        return np.asarray(meta.kps_face_p, dtype=np.float32).reshape(-1)
    return None


def _scale_face_signal(meta, weight: float):
    """Scale nose/eyes/face mesh by weight. 1 = face visible, 0 = back of head. Ears and body stay."""
    import copy

    meta = copy.copy(meta)
    w = float(np.clip(weight, 0.0, 1.0))
    if hasattr(meta, "kps_face_p") and meta.kps_face_p is not None:
        meta.kps_face_p = np.array(meta.kps_face_p, copy=True) * w
    if hasattr(meta, "kps_body_p") and meta.kps_body_p is not None:
        b = np.array(meta.kps_body_p, copy=True)
        for idx in (0, 14, 15):  # nose, right eye, left eye
            if idx < len(b):
                b[idx] *= w
        meta.kps_body_p = b
    if isinstance(meta, dict) and "keypoints_face" in meta:
        kf = np.array(meta["keypoints_face"], copy=True)
        if kf.ndim == 2 and kf.shape[1] >= 3:
            kf[:, 2] *= w
        meta["keypoints_face"] = kf
    return meta


def _smooth_weights(weights: np.ndarray, radius: int = 2) -> np.ndarray:
    """Short moving average so a head turn fades across a few frames instead of popping."""
    if radius <= 0 or len(weights) < 2:
        return weights
    out = np.empty_like(weights)
    for i in range(len(weights)):
        sl = weights[max(0, i - radius) : min(len(weights), i + radius + 1)]
        out[i] = float(sl.mean())
    return out


class CVFSFaceGate:
    """TEST: fade the face crop with VitPose face confidence.

    Face visible → weight 1, same face path. Head turns away → weight falls to 0
    over a few frames, so the master (back of head) and the body/ear pose take over.
    Stock face workflows do not use this node. Relight is not touched here.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Driver frames (Wan window)"}),
                "face_images": ("IMAGE", {"tooltip": "Face crops from PoseAndFaceDetection"}),
                "min_face_conf": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.05,
                        "max": 0.95,
                        "step": 0.01,
                        "tooltip": "Median VitPose face-kp confidence to keep the frame",
                    },
                ),
                "min_kp_frac": (
                    "FLOAT",
                    {
                        "default": 0.25,
                        "min": 0.05,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Fraction of face keypoints above min_face_conf",
                    },
                ),
            },
            "optional": {
                "pose_data": ("POSEDATA",),
            },
        }

    RETURN_TYPES = ("IMAGE", "POSEDATA", "FLOAT", "STRING")
    RETURN_NAMES = ("face_images_gated", "pose_data", "keep_ratio", "report")
    FUNCTION = "run"
    CATEGORY = "cut_video_face_swapper/test"

    def run(self, images, face_images, min_face_conf=0.35, min_kp_frac=0.25, pose_data=None):
        faces = face_images.detach().cpu()
        if faces.ndim != 4:
            raise ValueError(f"face_images expected BHWC, got {tuple(faces.shape)}")
        n = int(faces.shape[0])
        raw = np.zeros(n, dtype=np.float32)
        reasons: list[str] = []

        metas = None
        if isinstance(pose_data, dict):
            metas = pose_data.get("pose_metas") or pose_data.get("pose_metas_original")

        conf_thr = max(1e-3, float(min_face_conf))
        frac_thr = max(1e-3, float(min_kp_frac))
        for i in range(n):
            weight = 0.0
            detail = "no_pose"
            if metas is not None and i < len(metas):
                conf = _face_kp_conf(metas[i])
                if conf is not None and conf.size:
                    frac = float((conf >= conf_thr).mean())
                    med = float(np.median(conf))
                    weight = float(np.clip(min(med / conf_thr, frac / frac_thr), 0.0, 1.0))
                    detail = f"med={med:.2f} frac={frac:.2f}"
                else:
                    detail = "face_kps_missing"
            raw[i] = weight
            reasons.append(detail)

        weights = _smooth_weights(raw, radius=2)
        out = faces.clone()
        for i in range(n):
            out[i] *= float(weights[i])
        gated_pose = pose_data
        if isinstance(pose_data, dict) and metas is not None:
            gated_pose = dict(pose_data)
            gated_pose["pose_metas"] = [
                _scale_face_signal(meta, float(weights[i]) if i < n else 1.0)
                for i, meta in enumerate(metas)
            ]
        ratio = float(weights.mean()) if n else 0.0
        report = (
            f"CVFSFaceGate mean_face_weight={ratio:.2f} "
            f"full={int((weights > 0.85).sum())} fade={int(((weights > 0.05) & (weights <= 0.85)).sum())} "
            f"back={int((weights <= 0.05).sum())}/{n}\n"
            + "\n".join(f"{i:02d} w={weights[i]:.2f} {reasons[i]}" for i in range(min(n, 48)))
        )
        return (out, gated_pose, ratio, report)


def _host_path(path: Path) -> str:
    """Prefer the Windows mount path when running inside the Docker Comfy container."""
    s = str(path.resolve())
    prefix = "/opt/comfyui/"
    if s.startswith(prefix) or s.startswith("/opt/comfyui"):
        rel = s[len("/opt/comfyui") :].lstrip("/").replace("/", "\\")
        return f"c:\\Work\\docker\\ComfyUI\\{rel}"
    return s


def _mid_frame_bgr(path: Path) -> np.ndarray | None:
    import cv2

    if not path.exists() or path.stat().st_size <= 0:
        return None
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
        im = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return im
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(max(0, n // 2)))
    ok, fr = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
        ok, fr = cap.read()
    cap.release()
    return fr if ok else None


def _newest_part(job_name: str, sid: str) -> Path | None:
    hits = _find_output_parts(job_name, sid)
    if not hits:
        local = _part_mp4(_job_root(job_name), sid)
        return local if local.exists() else None
    return max(hits, key=lambda p: p.stat().st_mtime)


def _vace_full_path(job_name: str, sid: str) -> Path | None:
    out = Path(folder_paths.get_output_directory()) / "cut_video_face_swapper" / job_name / "vace_full"
    cands = sorted(out.glob(f"{sid}*.mp4"), key=lambda p: p.stat().st_mtime)
    cands = [p for p in cands if "silent" not in p.name]
    return cands[-1] if cands else None


def list_job_results(job_name: str) -> dict:
    """Status + paths for the Scenes overlay / results node."""
    job = _job_root(job_name)
    cuts_path = job / "cuts.json"
    if not cuts_path.exists():
        raise FileNotFoundError(f"No cuts.json — run Z1 Detect first ({job})")
    cuts = json.loads(cuts_path.read_text(encoding="utf-8"))
    rows = []
    for i, s in enumerate(cuts.get("segments") or []):
        sid = s["id"]
        mode = _resolve_wan_mode(s)
        part = _newest_part(job_name, sid)
        full = _vace_full_path(job_name, sid) if mode == "vace" else None
        rows.append(
            {
                "seg_index": i,
                "seg_id": sid,
                "frames": f"{s.get('start')}-{int(s.get('end', 0)) - 1}",
                "wan_mode": mode if mode in {"face", "vace"} else "face",
                "master_exists": (job / "masters" / f"{sid}.png").exists(),
                "part_exists": bool(part and part.exists()),
                "full_exists": bool(full and full.exists()),
                "part_path": _host_path(part) if part and part.exists() else None,
                "full_path": _host_path(full) if full and full.exists() else None,
                "preview_src": str(full or part) if (full or part) else None,
            }
        )
    final = job / "final.mp4"
    return {
        "job": job_name,
        "parts_dir": _host_path(Path(folder_paths.get_output_directory()) / "cut_video_face_swapper" / job_name / "parts"),
        "vace_full_dir": _host_path(Path(folder_paths.get_output_directory()) / "cut_video_face_swapper" / job_name / "vace_full"),
        "final_path": _host_path(final) if final.exists() else None,
        "segments": rows,
    }


class CVFSJobResults:
    """Show mid-frame of each rendered SEG plus absolute paths."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "job_name": ("STRING", {"default": "comfy_detect_demo"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("previews", "paths")
    FUNCTION = "run"
    CATEGORY = "cut_video_face_swapper"
    OUTPUT_NODE = True

    def run(self, job_name):
        import cv2

        info = list_job_results(job_name)
        frames = []
        lines = [
            f"parts: {info['parts_dir']}",
            f"vace_full: {info['vace_full_dir']}",
            f"final: {info['final_path'] or '(ещё нет)'}",
            "",
        ]
        for row in info["segments"]:
            mode = "VACE" if row["wan_mode"] == "vace" else "Wan"
            shown = row.get("full_path") or row.get("part_path")
            lines.append(
                f"SEG{row['seg_id']} [{mode}] frames {row['frames']}: "
                + (shown or "нет файла")
            )
            src = Path(row["preview_src"]) if row.get("preview_src") else None
            fr = _mid_frame_bgr(src) if src else None
            if fr is None:
                blank = np.zeros((180, 320, 3), np.uint8)
                cv2.putText(
                    blank,
                    f"SEG{row['seg_id']} empty",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (180, 180, 180),
                    2,
                )
                frames.append(cv2.cvtColor(blank, cv2.COLOR_BGR2RGB))
            else:
                h, w = fr.shape[:2]
                scale = 320 / max(w, 1)
                fr = cv2.resize(
                    fr,
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
                label = f"SEG{row['seg_id']} {mode}"
                cv2.putText(fr, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
                cv2.putText(fr, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        if not frames:
            frames = [np.zeros((180, 320, 3), np.uint8)]
        text = "\n".join(lines)
        return {"ui": {"text": [text]}, "result": (_uint8_to_tensor_batch(frames), text)}


NODE_CLASS_MAPPINGS = {
    "CVFSDetectSceneCuts": CVFSDetectSceneCuts,
    "CVFSSaveMasterForSegment": CVFSSaveMasterForSegment,
    "CVFSBuildWanWorkflowsFromMasters": CVFSBuildWanWorkflowsFromMasters,
    "CVFSRunFromMastersInOrder": CVFSRunFromMastersInOrder,
    "CVFSRunFromMastersFolder": CVFSRunFromMastersFolder,
    "CVFSAssembleJob": CVFSAssembleJob,
    "CVFSZoneSelect": CVFSZoneSelect,
    "CVFSSegPicker": CVFSSegPicker,
    "CVFSRectFromMask": CVFSRectFromMask,
    "CVFSFeatherMask": CVFSFeatherMask,
    "CVFSBlendMasked": CVFSBlendMasked,
    "CVFSCropRefInRect": CVFSCropRefInRect,
    "CVFSFaceGate": CVFSFaceGate,
    "CVFSPasteVacePlate": CVFSPasteVacePlate,
    "CVFSJobResults": CVFSJobResults,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CVFSDetectSceneCuts": "CutSwapWan Detect Scenes",
    "CVFSSaveMasterForSegment": "CutSwapWan Save Master For SEG",
    "CVFSBuildWanWorkflowsFromMasters": "CutSwapWan Build Wan From Masters",
    "CVFSRunFromMastersInOrder": "CutSwapWan Run Masters (batch wire)",
    "CVFSRunFromMastersFolder": "CutSwapWan Run Masters Folder",
    "CVFSAssembleJob": "CutSwapWan Assemble (final.mp4)",
    "CVFSZoneSelect": "CutSwapWan Zone Select (1/2/3)",
    "CVFSSegPicker": "CutSwapWan SEG Picker",
    "CVFSRectFromMask": "CutSwapWan Rect From Matte (wipe)",
    "CVFSFeatherMask": "CutSwapWan Feather Mask",
    "CVFSBlendMasked": "CutSwapWan Blend Masked",
    "CVFSCropRefInRect": "CutSwapWan Crop Ref In Rect",
    "CVFSFaceGate": "CutSwapWan Face Gate",
    "CVFSPasteVacePlate": "CutSwapWan Paste VACE Plate",
    "CVFSJobResults": "CutSwapWan Job Results",
}
