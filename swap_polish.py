"""SwapPolish — ReActor base → optional tight face clone → opaque hair from matte∖face.

No full Wan redraw. Fast local path for CutSwapWan Test P.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def _largest_face(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def _kps5(face) -> np.ndarray | None:
    k = getattr(face, "kps", None)
    if k is None:
        return None
    k = np.asarray(k, dtype=np.float32)
    return k if k.shape == (5, 2) else None


def _face_bbox(face) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in face.bbox]
    return x1, y1, x2, y2


def _ellipse_mask(h: int, w: int, cx: float, cy: float, rx: float, ry: float, soft: int = 0) -> np.ndarray:
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (int(cx), int(cy)), (max(1, int(rx)), max(1, int(ry))), 0, 0, 360, 1.0, -1)
    if soft > 0:
        k = soft | 1
        m = cv2.GaussianBlur(m, (k, k), 0)
        m = np.clip(m, 0, 1)
    return m


def _face_core_mask(h: int, w: int, face, scale: float = 0.95, soft: int = 0) -> np.ndarray:
    x1, y1, x2, y2 = _face_bbox(face)
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2) + 0.02 * (y2 - y1)
    rx = 0.48 * (x2 - x1) * scale
    ry = 0.55 * (y2 - y1) * scale
    return _ellipse_mask(h, w, cx, cy, rx, ry, soft=soft)


def _matte_f(matte: np.ndarray) -> np.ndarray:
    g = matte.max(axis=2) if matte.ndim == 3 else matte
    return np.clip(g.astype(np.float32) / 255.0, 0, 1)


def _similarity_from_kps(src_kps: np.ndarray, dst_kps: np.ndarray) -> np.ndarray | None:
    """Rigid+scale (partial affine). More stable than free affine for head paste."""
    M, _ = cv2.estimateAffinePartial2D(
        src_kps.astype(np.float32),
        dst_kps.astype(np.float32),
        method=cv2.LMEDS,
        ransacReprojThreshold=3.0,
    )
    return M


def _crop_master_head(master_bgr: np.ndarray, master_face, pad: float = 1.35) -> tuple[np.ndarray, object]:
    """Tight head crop so warp cannot drag studio shoulders/bg into the frame."""
    h, w = master_bgr.shape[:2]
    x1, y1, x2, y2 = _face_bbox(master_face)
    bw, bh = x2 - x1, y2 - y1
    # more room above for hair; less below
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    half_w = 0.55 * bw * pad
    half_h = 0.70 * bh * pad
    top = cy - half_h * 1.25
    bot = cy + half_h * 0.95
    left = cx - half_w
    right = cx + half_w
    x0 = int(max(0, np.floor(left)))
    y0 = int(max(0, np.floor(top)))
    x1i = int(min(w, np.ceil(right)))
    y1i = int(min(h, np.ceil(bot)))
    crop = master_bgr[y0:y1i, x0:x1i].copy()

    # shift face object coords into crop space (lightweight shim)
    class _F:
        pass

    f = _F()
    f.bbox = np.asarray(
        [master_face.bbox[0] - x0, master_face.bbox[1] - y0, master_face.bbox[2] - x0, master_face.bbox[3] - y0],
        dtype=np.float32,
    )
    k = _kps5(master_face)
    f.kps = None if k is None else (k - np.array([x0, y0], dtype=np.float32))
    f.embedding = getattr(master_face, "embedding", None)
    return crop, f


def _warp_head(master_crop: np.ndarray, master_face, dst_face, out_hw: tuple[int, int]) -> np.ndarray | None:
    sk = _kps5(master_face)
    dk = _kps5(dst_face)
    if sk is None or dk is None:
        return None
    M = _similarity_from_kps(sk, dk)
    if M is None:
        return None
    h, w = out_hw
    return cv2.warpAffine(
        master_crop,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _master_hair_mask(crop_bgr: np.ndarray, face) -> np.ndarray:
    """Hair band on master crop: matte-like dark region above/around face, not skin."""
    h, w = crop_bgr.shape[:2]
    x1, y1, x2, y2 = _face_bbox(face)
    face_m = _face_core_mask(h, w, face, scale=1.05, soft=0)
    # candidate: everything above mid-nose, outside face core
    yy = np.arange(h, dtype=np.float32)[:, None]
    brow_y = float(y1 + 0.28 * (y2 - y1))
    upper = (yy < (brow_y + 0.08 * (y2 - y1))).astype(np.float32)
    upper = np.repeat(upper, w, axis=1)
    # dark / non-skin heuristic in HSV
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    v = hsv[:, :, 2] / 255.0
    s = hsv[:, :, 1] / 255.0
    dark = ((v < 0.55) | ((v < 0.72) & (s > 0.18))).astype(np.float32)
    hair = np.clip(upper * (1.0 - face_m) * dark, 0, 1)
    # keep only largest blob near crown
    bin8 = (hair > 0.35).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin8, connectivity=8)
    if n > 1:
        # skip background label 0
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        hair = (labels == best).astype(np.float32)
    else:
        hair = (hair > 0.35).astype(np.float32)
    hair = cv2.GaussianBlur(hair, (5, 5), 0)
    return np.clip(hair, 0, 1)


def _hair_mask_dst(matte: np.ndarray, face, shrink_face: float = 1.12) -> np.ndarray:
    """Driver hair region = FaceHair matte minus expanded face oval. Hard core + short feather."""
    h, w = matte.shape[:2]
    m = _matte_f(matte)
    # hard threshold first — kill soft bleed outside silhouette
    hard = (m > 0.45).astype(np.float32)
    face_m = _face_core_mask(h, w, face, scale=shrink_face, soft=0)
    # dilate face a bit so hair starts above forehead, not on cheeks
    k = max(3, int(0.04 * min(h, w)) | 1)
    face_d = cv2.dilate(face_m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    hair = np.clip(hard - face_d, 0, 1)
    # keep upper half of head (hair / under hat), drop neck/shoulders in matte
    x1, y1, x2, y2 = _face_bbox(face)
    chin = float(y2)
    yy = np.arange(h, dtype=np.float32)[:, None]
    upper = (yy < (chin - 0.05 * (y2 - y1))).astype(np.float32)
    upper = np.repeat(upper, w, axis=1)
    hair = hair * upper
    # erode then tiny feather — opaque core, soft 3–5px rim only
    if float(hair.max()) < 0.05:
        return hair
    bin8 = (hair > 0.5).astype(np.uint8)
    er = cv2.erode(bin8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    feather = cv2.GaussianBlur(er.astype(np.float32), (7, 7), 0)
    return np.clip(feather, 0, 1)


def _shape_polish(
    swap_bgr: np.ndarray,
    warped_head: np.ndarray | None,
    swap_face,
    strength: float = 0.2,
) -> np.ndarray:
    """Very light, tight clone — never a translucent second silhouette."""
    if warped_head is None or strength <= 0.01:
        return swap_bgr
    h, w = swap_bgr.shape[:2]
    # hard core only; tiny feather
    core = _face_core_mask(h, w, swap_face, scale=0.88, soft=0)
    k = max(3, int(0.03 * min(h, w)) | 1)
    a = cv2.GaussianBlur(core, (k, k), 0)
    a = (a * float(np.clip(strength, 0.0, 0.35)))[:, :, None]
    # only blend where warped has signal (non-black)
    valid = (warped_head.max(axis=2) > 8).astype(np.float32)[:, :, None]
    a = a * valid
    out = swap_bgr.astype(np.float32) * (1.0 - a) + warped_head.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def _hair_polish(
    base_bgr: np.ndarray,
    warped_head: np.ndarray | None,
    master_hair_on_dst: np.ndarray | None,
    matte: np.ndarray,
    swap_face,
    strength: float = 0.95,
) -> np.ndarray:
    """Opaque hair paste into matte∖face. No soft whole-plate mix."""
    if warped_head is None:
        return base_bgr
    h, w = base_bgr.shape[:2]
    dst_hair = _hair_mask_dst(matte, swap_face)
    if float(dst_hair.max()) < 0.05:
        return base_bgr
    src_hair = master_hair_on_dst
    if src_hair is None:
        src_hair = (warped_head.max(axis=2) > 12).astype(np.float32)
    # intersection: only paint where both say hair
    hair = np.clip(dst_hair * src_hair, 0, 1)
    if float(hair.max()) < 0.05:
        # fallback: still paint dst hair zone from warped (may help under-hat)
        hair = dst_hair
    a = (hair * float(np.clip(strength, 0.0, 1.0)))[:, :, None]
    # near-opaque: lift mid alphas so we don't get ghost plates
    a = np.clip(a * 1.15, 0, 1)
    out = base_bgr.astype(np.float32) * (1.0 - a) + warped_head.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def _load_insightface(model_path: Path):
    from insightface.app import FaceAnalysis
    from insightface.model_zoo.inswapper import INSwapper

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    swapper = INSwapper(str(model_path))
    return app, swapper


def run_swap_polish_segment(
    *,
    job: Path,
    job_name: str,
    seg: dict,
    driver_path: Path,
    mask_path: Path,
    master_path: Path,
    model_path: Path,
    fps: float = 24.0,
    shape_strength: float = 0.18,
    hair_strength: float = 0.95,
    output_root: Path | None = None,
) -> dict:
    """Write preview/<sid>_P_*.mp4 (+ mid compare). Returns paths/meta."""
    sid = str(seg["id"])
    start = int(seg["start"])
    end = int(seg["end"])
    n = max(1, end - start)
    ref_abs = int(seg.get("ref_frame", start))
    ref_abs = min(max(ref_abs, start), max(start, end - 1))

    if output_root is None:
        comfy_root = Path(job).parents[3]
        out_root = comfy_root / "output" / "cut_video_face_swapper" / job_name / "preview"
    else:
        out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    app, swapper = _load_insightface(model_path)
    master_full = cv2.imread(str(master_path), cv2.IMREAD_COLOR)
    if master_full is None:
        raise FileNotFoundError(f"No master: {master_path}")
    mfaces = app.get(master_full)
    if not mfaces:
        raise RuntimeError("No face on master still")
    master_face_full = _largest_face(mfaces)
    master_crop, master_face = _crop_master_head(master_full, master_face_full, pad=1.4)
    src_face = master_face_full  # embeddings/bbox for INSwapper from full detect
    master_hair_local = _master_hair_mask(master_crop, master_face)

    cap_d = cv2.VideoCapture(str(driver_path))
    cap_m = cv2.VideoCapture(str(mask_path))
    if not cap_d.isOpened() or not cap_m.isOpened():
        raise FileNotFoundError("Cannot open driver/mask")
    for _ in range(start):
        if not cap_d.read()[0] or not cap_m.read()[0]:
            raise RuntimeError("driver/mask shorter than seg start")

    frames_out: list[np.ndarray] = []
    mid_idx = min(max(0, ref_abs - start), n - 1)
    compare_triple = None

    for i in range(n):
        ok_d, drv = cap_d.read()
        ok_m, msk = cap_m.read()
        if not ok_d or not ok_m:
            break
        faces = app.get(drv)
        if not faces:
            frames_out.append(drv)
            continue
        tgt = _largest_face(faces)
        swapped = swapper.get(drv, tgt, src_face, paste_back=True)
        sfaces = app.get(swapped) or [tgt]
        sface = _largest_face(sfaces)

        warped = _warp_head(master_crop, master_face, sface, swapped.shape[:2])
        hair_w = None
        if warped is not None:
            # warp master hair mask with same transform
            sk = _kps5(master_face)
            dk = _kps5(sface)
            M = _similarity_from_kps(sk, dk) if sk is not None and dk is not None else None
            if M is not None:
                hh, ww = swapped.shape[:2]
                hair_w = cv2.warpAffine(
                    master_hair_local,
                    M,
                    (ww, hh),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                hair_w = np.clip(hair_w, 0, 1)

        polished = _shape_polish(swapped, warped, sface, strength=shape_strength)
        polished = _hair_polish(polished, warped, hair_w, msk, sface, strength=hair_strength)
        frames_out.append(polished)
        if i == mid_idx:
            compare_triple = (drv, swapped, polished)

    cap_d.release()
    cap_m.release()
    if not frames_out:
        raise RuntimeError("No frames polished")

    existing = sorted(out_root.glob(f"{sid}_P_*.mp4"))
    idx = 1
    if existing:
        try:
            idx = int(existing[-1].stem.split("_")[-1]) + 1
        except Exception:
            idx = len(existing) + 1
    stem = f"{sid}_P_{idx:05d}"
    mp4_path = out_root / f"{stem}.mp4"
    png_path = out_root / f"{stem}.png"
    cmp_path = out_root / f"{stem}_compare.jpg"

    h, w = frames_out[0].shape[:2]
    wr = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps or 24), (w, h), True)
    for fr in frames_out:
        wr.write(fr)
    wr.release()
    cv2.imwrite(str(png_path), frames_out[mid_idx])

    if compare_triple is not None:
        a, b, c = compare_triple
        S = 360
        row = np.hstack([cv2.resize(a, (S, S)), cv2.resize(b, (S, S)), cv2.resize(c, (S, S))])
        cv2.putText(row, "VIDEO", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(row, "SWAP", (S + 8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(row, "POLISH", (2 * S + 8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imwrite(str(cmp_path), row)

    meta = {
        "seg_id": sid,
        "mode": "swap_polish",
        "n_frames": len(frames_out),
        "start": start,
        "end": start + len(frames_out),
        "ref_frame": ref_abs,
        "shape_strength": shape_strength,
        "hair_strength": hair_strength,
        "mp4": str(mp4_path),
        "png": str(png_path),
        "compare": str(cmp_path) if cmp_path.exists() else "",
    }
    (out_root / f"{stem}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return meta
