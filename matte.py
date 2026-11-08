"""Background matte after a pinned ReActor+Wan clip.

One scene at a time: 2-sec test → choose mask → full render → ready pool.
White in the mask keeps the generation. Black returns the original plate.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import threading
from pathlib import Path

import folder_paths

_MODEL = None
_LOCK = threading.Lock()


def _input_root() -> Path:
    return Path(folder_paths.get_input_directory())


def _output_root() -> Path:
    return Path(folder_paths.get_output_directory())


def job_dir(job_name: str) -> Path:
    return _input_root() / "cut_video_face_swapper" / "jobs" / job_name


def preview_dir(job_name: str) -> Path:
    return _output_root() / "cut_video_face_swapper" / job_name / "preview"


def ready_dir(job_name: str) -> Path:
    return job_dir(job_name) / "ready"


def _seg(job_name: str, seg_index: int) -> tuple[dict, str]:
    cuts = json.loads((job_dir(job_name) / "cuts.json").read_text(encoding="utf-8"))
    segs = cuts.get("segments") or []
    if seg_index < 0 or seg_index >= len(segs):
        raise IndexError(f"Нет сцены {seg_index}")
    seg = segs[seg_index]
    return seg, str(seg["id"])


def bundle_dir(job_name: str, seg_index: int, kind: str = "test") -> Path:
    _seg_row, sid = _seg(job_name, seg_index)
    name = f"{sid}_test" if kind == "test" else sid
    return job_dir(job_name) / "segs" / name


def _progress_path(job_name: str, seg_index: int, kind: str = "test") -> Path:
    return bundle_dir(job_name, seg_index, kind) / "matte_progress.json"


def _choice_path(job_name: str, seg_index: int) -> Path:
    return bundle_dir(job_name, seg_index, "test") / "matte_choice.json"


def _write_progress(job_name: str, seg_index: int, kind: str = "test", **payload) -> None:
    path = _progress_path(job_name, seg_index, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _read_progress(job_name: str, seg_index: int, kind: str = "test") -> dict:
    path = _progress_path(job_name, seg_index, kind)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_choice(job_name: str, seg_index: int, mode: str, mask_name: str = "") -> dict:
    path = _choice_path(job_name, seg_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"mode": mode, "mask_name": mask_name}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def read_choice(job_name: str, seg_index: int) -> dict:
    path = _choice_path(job_name, seg_index)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _read_video(path: Path, n: int | None = None) -> tuple[list, float]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        if n is not None and len(frames) >= n:
            break
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames, float(fps)


def _h264(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    if dst.is_file() and dst.stat().st_mtime >= src.stat().st_mtime:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(src),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(dst),
            ],
            check=True,
            capture_output=True,
        )
        return dst.is_file()
    except (OSError, subprocess.CalledProcessError):
        return False


def _view(path: Path) -> dict | None:
    if not path.is_file():
        return None
    root_in = _input_root()
    root_out = _output_root()
    try:
        rel = path.resolve().relative_to(root_in.resolve())
        kind = "input"
    except ValueError:
        try:
            rel = path.resolve().relative_to(root_out.resolve())
            kind = "output"
        except ValueError:
            return None
    return {
        "filename": rel.name,
        "subfolder": str(rel.parent).replace("\\", "/"),
        "type": kind,
    }


def newest_generation(job_name: str, sid: str, since: float = 0.0, kind: str = "test") -> Path | None:
    hits = []
    if kind == "test":
        folder = preview_dir(job_name)
        if folder.is_dir():
            for pattern in (f"{sid}_AQ4_*.mp4", f"{sid}_A_*.mp4"):
                for path in folder.glob(pattern):
                    name = path.name
                    if any(tok in name for tok in ("norel", "decoded", "matte", "reswap", "fullplate", "rvm")):
                        continue
                    if since and path.stat().st_mtime + 0.5 < since:
                        continue
                    hits.append(path)
    else:
        parts = job_dir(job_name) / "parts"
        out_parts = _output_root() / "cut_video_face_swapper" / job_name / "parts"
        for folder in (parts, out_parts):
            if not folder.is_dir():
                continue
            for path in folder.glob(f"{sid}*.mp4"):
                if since and path.stat().st_mtime + 0.5 < since:
                    continue
                hits.append(path)
        # Full runs may also land in preview without _test in the name when tagged.
        folder = preview_dir(job_name)
        if folder.is_dir():
            for path in folder.glob(f"{sid}_AQ4_*.mp4"):
                if any(tok in path.name for tok in ("norel", "decoded", "matte", "reswap", "fullplate", "rvm")):
                    continue
                if since and path.stat().st_mtime + 0.5 < since:
                    continue
                hits.append(path)
    if not hits:
        return None
    return max(hits, key=lambda p: p.stat().st_mtime)


def pin_generation(job_name: str, seg_index: int, source: Path, kind: str = "test") -> dict:
    bundle = bundle_dir(job_name, seg_index, kind)
    bundle.mkdir(parents=True, exist_ok=True)
    dest = bundle / "generated.mp4"
    if source.resolve() != dest.resolve():
        shutil.copyfile(source, dest)
    play = bundle / "generated_play.mp4"
    _h264(dest, play)
    meta = {
        "stage": "generated",
        "kind": kind,
        "video": "generated.mp4",
        "from": str(source),
        "what": "ReActor + Wan. No background matte yet.",
    }
    (bundle / "generated.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_progress(job_name, seg_index, kind, stage="generated", done=0, total=0, error="")
    return status(job_name, seg_index, kind=kind)


def status(job_name: str, seg_index: int, since: float = 0.0, kind: str = "test") -> dict:
    seg, sid = _seg(job_name, seg_index)
    bundle = bundle_dir(job_name, seg_index, kind)
    generated = bundle / "generated.mp4"
    play = bundle / "generated_play.mp4"
    if generated.is_file():
        _h264(generated, play)
    preview = preview_dir(job_name)
    frame = preview / f"matte_{sid}_{kind}_frame.jpg"
    if kind == "test" and not frame.is_file():
        legacy = preview / f"matte_{sid}_frame.jpg"
        if legacy.is_file():
            frame = legacy
        elif sid == "00":
            old = preview / "00_AQ4_norel_mask_vs_gen_ben2_object.jpg"
            if old.is_file():
                frame = old
    result = preview / f"matte_{sid}_{kind}_play.mp4"
    if not result.is_file():
        result = preview / f"matte_{sid}_{kind}.mp4"
    if kind == "test" and not result.is_file():
        alt = preview / f"matte_{sid}_play.mp4"
        if alt.is_file():
            result = alt
    ready = ready_dir(job_name) / f"{sid}.mp4"
    choice = read_choice(job_name, seg_index)
    latest = newest_generation(job_name, sid, 0, kind)
    fresh = newest_generation(job_name, sid, since, kind) if since else None
    prog = _read_progress(job_name, seg_index, kind)
    return {
        "job": job_name,
        "seg_id": sid,
        "seg_index": seg_index,
        "kind": kind,
        "frames": f"{seg.get('start')}-{int(seg.get('end', 0)) - 1}",
        "generated": _view(play if play.is_file() else generated),
        "frame": _view(frame) if frame.is_file() else None,
        "result": _view(result) if result.is_file() else None,
        "ready": _view(ready) if ready.is_file() else None,
        "choice": choice,
        "latest_generation": _view(latest) if latest else None,
        "fresh_generation": _view(fresh) if fresh else None,
        "progress": prog,
    }


def list_pool(job_name: str) -> dict:
    cuts = json.loads((job_dir(job_name) / "cuts.json").read_text(encoding="utf-8"))
    rows = []
    for i, seg in enumerate(cuts.get("segments") or []):
        sid = str(seg["id"])
        ready = ready_dir(job_name) / f"{sid}.mp4"
        test_gen = bundle_dir(job_name, i, "test") / "generated.mp4"
        choice = read_choice(job_name, i)
        rows.append(
            {
                "seg_index": i,
                "seg_id": sid,
                "frames": f"{seg.get('start')}-{int(seg.get('end', 0)) - 1}",
                "in_pool": ready.is_file(),
                "has_test": test_gen.is_file(),
                "choice": choice.get("mode") or "",
                "ready": _view(ready) if ready.is_file() else None,
            }
        )
    return {
        "job": job_name,
        "ready_dir": str(ready_dir(job_name)),
        "final": str(job_dir(job_name) / "final.mp4"),
        "segments": rows,
        "pool_count": sum(1 for r in rows if r["in_pool"]),
    }


def _ben():
    global _MODEL
    if _MODEL is None:
        import torch
        import safetensors.torch as st
        from ben2 import AutoModel

        # mmgp replaces safetensors.load_file with a narrower function.
        # safetensors 0.8 passes backend= into it and the load crashes.
        if not getattr(st.load_file, "_cvfs_backend", False):
            orig = st.load_file

            def load_file(*args, **kwargs):
                kwargs.pop("backend", None)
                return orig(*args, **kwargs)

            load_file._cvfs_backend = True
            st.load_file = load_file

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _MODEL = AutoModel.from_pretrained("PramaLLC/BEN2").to(device).eval()
    return _MODEL


def _ben_alpha(bgr):
    import cv2
    import numpy as np
    from PIL import Image

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    fg = np.array(_ben().inference(Image.fromarray(rgb), refine_foreground=True))
    if fg.ndim == 3 and fg.shape[2] == 4:
        return fg[:, :, 3].astype(np.float32) / 255.0
    return np.clip(fg.astype(np.float32).mean(axis=2) / 255.0, 0, 1)


def auto_alpha(bgr):
    """Person matte. A tube whose main body is outside the matte is removed, shadow face included."""
    import cv2
    import numpy as np

    alpha = _ben_alpha(bgr)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.sqrt((lab[:, :, 1] - 128.0) ** 2 + (lab[:, :, 2] - 128.0) ** 2)
    yellow = (lab[:, :, 2] - 128.0) - (lab[:, :, 1] - 128.0)
    person = alpha > 0.50
    outside = alpha < 0.18
    rim = cv2.dilate(person.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    rim = (rim > 0) & outside
    seed0 = rim & (chroma > 18.0) & (yellow > 6.0) & (lab[:, :, 0] > 70.0)
    if int(seed0.sum()) < 30:
        return np.clip(alpha, 0, 1)
    color = np.median(lab[seed0], axis=0)
    dist = np.linalg.norm(lab - color, axis=2)
    hanging = outside & (dist < 18.0) & (chroma > 16.0) & (yellow > 4.0)
    n, labels = cv2.connectedComponents(hanging.astype(np.uint8), connectivity=8)
    hang_ids = [i for i in range(1, n) if (labels == i).sum() > 40 and ((labels == i) & seed0).any()]
    main = np.isin(labels, hang_ids) if hang_ids else seed0
    if int(main.sum()) < 30:
        main = seed0
    color = np.median(lab[main], axis=0)
    spread = np.linalg.norm(lab[main] - color, axis=1)
    radius = float(min(20.0, max(12.0, np.percentile(spread, 90) + 4.0)))
    dist = np.linalg.norm(lab - color, axis=2)
    same = (dist < radius) & (chroma > 16.0) & (yellow > 4.0)
    n, labels = cv2.connectedComponents(same.astype(np.uint8), connectivity=8)
    drop = np.zeros(alpha.shape, bool)
    for i in range(1, n):
        comp = labels == i
        if not (comp & main).any():
            continue
        outside_n = int((comp & outside).sum())
        inside_n = int((comp & person).sum())
        if outside_n >= 80 and outside_n >= 0.30 * (outside_n + inside_n):
            drop |= comp & (alpha > 0.05)
    light = main | (drop & (chroma > 14.0) & (yellow > 2.0))
    core = cv2.erode(person.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0
    bright_L = float(np.median(lab[:, :, 0][light])) if light.any() else 150.0
    green = (lab[:, :, 1] < 116.0) & (chroma > 10.0)
    skin = (lab[:, :, 1] > 145.0) & (yellow < 6.0)
    highlight = (lab[:, :, 0] > 165.0) & (chroma < 26.0)
    tube = light.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(5):
        fringe = cv2.dilate(tube.astype(np.uint8), k) > 0
        add = (
            fringe
            & ~tube
            & ~core
            & ~green
            & ~skin
            & ~highlight
            & (lab[:, :, 0] < bright_L - 12.0)
            & (lab[:, :, 0] > 16.0)
        )
        if not add.any():
            break
        tube |= add
    drop = (tube & (alpha > 0.05)) | (drop & (alpha > 0.05))
    cut = alpha.copy()
    cut[drop] = 0
    return cv2.GaussianBlur(np.clip(cut, 0, 1), (0, 0), 0.6)


def _compare(gen, alpha, dest: Path) -> None:
    import cv2
    import numpy as np

    mask_bgr = cv2.cvtColor((np.clip(alpha, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    red = np.zeros_like(gen)
    red[:, :, 2] = 255
    aa = alpha[:, :, None] * 0.45
    overlay = np.clip(gen.astype(np.float32) * (1 - aa) + red.astype(np.float32) * aa, 0, 255).astype(np.uint8)
    height = 520

    def fit(img):
        h0, w0 = img.shape[:2]
        s = height / h0
        return cv2.resize(img, (max(1, int(w0 * s)), height), interpolation=cv2.INTER_AREA)

    tiles = [fit(gen), fit(mask_bgr), fit(overlay)]
    labels = ["generation", "mask", "mask on gen"]
    row = np.hstack(tiles)
    x = 8
    for lab, tile in zip(labels, tiles):
        cv2.putText(row, lab, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        x += tile.shape[1]
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), row)


def _load_generated(job_name: str, seg_index: int, kind: str = "test"):
    bundle = bundle_dir(job_name, seg_index, kind)
    video = bundle / "generated.mp4"
    if not video.is_file():
        raise FileNotFoundError("Нет сохранённого прогона. Сначала сделай тест или финальный рендер.")
    track_path = bundle / "track.json"
    if not track_path.is_file():
        raise FileNotFoundError(f"Нет track.json в {bundle}")
    tracks = json.loads(track_path.read_text(encoding="utf-8"))
    gens, fps = _read_video(video)
    n = min(len(gens), len(tracks))
    plate_path = job_dir(job_name) / "driver.mp4"
    seg, _sid = _seg(job_name, seg_index)
    start = int(seg.get("start") or 0)
    if kind == "test":
        # Test window is a short slice around the ref; track boxes match that slice.
        plates, _ = _read_video(plate_path, start + max(n, 1) + 64)
        # Prefer aligning by the first track box against plate frames near start.
        plates = plates[start : start + n]
        if len(plates) < n:
            plates, _ = _read_video(plate_path)
            # Fall back: use the middle of the segment.
            mid = (int(seg.get("start") or 0) + int(seg.get("end") or 0)) // 2
            i0 = max(0, mid - n // 2)
            plates = plates[i0 : i0 + n]
    else:
        end = int(seg.get("end") or (start + n))
        plates, _ = _read_video(plate_path, end)
        plates = plates[start:end][:n]
    if len(plates) < n:
        raise RuntimeError("Оригинал короче прогона.")
    return gens[:n], plates[:n], tracks[:n], fps


def _paste(gen, plate, box, alpha):
    import cv2
    import numpy as np

    h, w = plate.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    cw, ch = x1 - x0, y1 - y0
    if cw < 2 or ch < 2:
        return plate
    local = cv2.resize(gen, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
    m = cv2.resize(alpha, (cw, ch), interpolation=cv2.INTER_LINEAR)
    border = max(18, min(cw, ch) // 12)
    yy = np.arange(ch, dtype=np.float32)
    xx = np.arange(cw, dtype=np.float32)
    dist = np.minimum(np.minimum(yy, ch - 1 - yy)[:, None], np.minimum(xx, cw - 1 - xx)[None, :])
    rim = np.clip(dist / float(border), 0, 1)
    rim = cv2.GaussianBlur(rim, (0, 0), border * 0.35)
    m = np.clip(m * rim, 0, 1)[:, :, None]
    patch = plate[y0:y1, x0:x1]
    blended = local.astype(np.float32) * m + patch.astype(np.float32) * (1.0 - m)
    out = plate.copy()
    out[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def _write_result(job_name: str, sid: str, frames, fps: float, kind: str) -> Path:
    import cv2

    preview = preview_dir(job_name)
    preview.mkdir(parents=True, exist_ok=True)
    raw = preview / f"matte_{sid}_{kind}.mp4"
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h), True)
    for fr in frames:
        writer.write(fr)
    writer.release()
    play = preview / f"matte_{sid}_{kind}_play.mp4"
    if not _h264(raw, play):
        play = raw
    mid = frames[len(frames) // 2]
    cv2.imwrite(str(preview / f"matte_{sid}_{kind}.jpg"), mid)
    return play


def _smooth(alphas):
    import numpy as np

    stack = np.stack(alphas, axis=0)
    out = []
    n = stack.shape[0]
    for i in range(n):
        sl = stack[max(0, i - 1) : min(n, i + 2)]
        out.append(np.median(sl, axis=0).astype(np.float32))
    return out


def frame_preview(job_name: str, seg_index: int, kind: str = "test") -> dict:
    seg, sid = _seg(job_name, seg_index)
    gens, _plates, _tracks, _fps = _load_generated(job_name, seg_index, kind)
    mid = len(gens) // 2
    _write_progress(job_name, seg_index, kind, stage="frame", done=0, total=1, error="")
    alpha = auto_alpha(gens[mid])
    dest = preview_dir(job_name) / f"matte_{sid}_{kind}_frame.jpg"
    _compare(gens[mid], alpha, dest)
    _write_progress(job_name, seg_index, kind, stage="pause", done=1, total=1, error="")
    return status(job_name, seg_index, kind=kind)


def _mask_video_or_image(path: Path) -> list:
    import cv2

    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise FileNotFoundError(path)
        return [im]
    frames, _fps = _read_video(path)
    if not frames:
        raise RuntimeError("Пустая маска.")
    return frames


def _fit_mask(mask, gen_shape, plate_shape, box):
    import cv2
    import numpy as np

    if mask.ndim == 3:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if mask.shape[2] == 3 else mask[:, :, 0]
    else:
        gray = mask
    alpha = gray.astype(np.float32)
    if alpha.max() > 1.5:
        alpha = alpha / 255.0
    gh, gw = gen_shape[:2]
    ph, pw = plate_shape[:2]
    if gray.shape[0] == ph and gray.shape[1] == pw:
        x0, y0, x1, y1 = [int(v) for v in box]
        alpha = alpha[max(0, y0) : y1, max(0, x0) : x1]
    if alpha.shape[0] != gh or alpha.shape[1] != gw:
        alpha = cv2.resize(alpha, (gw, gh), interpolation=cv2.INTER_LINEAR)
    return np.clip(alpha, 0, 1)


def _run_alphas(job_name: str, seg_index: int, gens, kind: str, manual_path: Path | None = None):
    if manual_path is not None and manual_path.is_file():
        srcs = _mask_video_or_image(manual_path)
        plates_needed = True
        gens2, plates, tracks, _fps = _load_generated(job_name, seg_index, kind)
        alphas = []
        total = len(gens)
        for i in range(total):
            _write_progress(job_name, seg_index, kind, stage="manual", done=i, total=total, error="")
            src = srcs[min(i, len(srcs) - 1)]
            alphas.append(_fit_mask(src, gens[i].shape, plates[i].shape, tracks[i]))
        return alphas
    alphas = []
    total = len(gens)
    for i, fr in enumerate(gens):
        _write_progress(job_name, seg_index, kind, stage="auto", done=i, total=total, error="")
        alphas.append(auto_alpha(fr))
    return _smooth(alphas)


def _save_to_pool(job_name: str, sid: str, frames, fps: float) -> Path:
    import cv2

    ready = ready_dir(job_name)
    ready.mkdir(parents=True, exist_ok=True)
    raw = ready / f"{sid}.mp4"
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h), True)
    for fr in frames:
        writer.write(fr)
    writer.release()
    play = ready / f"{sid}_play.mp4"
    _h264(raw, play)
    (ready / f"{sid}.json").write_text(
        json.dumps({"seg_id": sid, "frames": len(frames), "video": raw.name}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return raw


def _finish(job_name: str, seg_index: int, alphas, kind: str, to_pool: bool) -> dict:
    seg, sid = _seg(job_name, seg_index)
    gens, plates, tracks, fps = _load_generated(job_name, seg_index, kind)
    n = min(len(gens), len(plates), len(tracks), len(alphas))
    frames = [_paste(gens[i], plates[i], tracks[i], alphas[i]) for i in range(n)]
    _write_result(job_name, sid, frames, fps, kind)
    if to_pool or kind == "full":
        _save_to_pool(job_name, sid, frames, fps)
    _write_progress(job_name, seg_index, kind, stage="done", done=n, total=n, error="")
    return status(job_name, seg_index, kind=kind)


def start_auto(job_name: str, seg_index: int, kind: str = "test", to_pool: bool = False) -> dict:
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("Маска уже считается.")
    save_choice(job_name, seg_index, "auto")

    def work():
        try:
            gens, _p, _t, _fps = _load_generated(job_name, seg_index, kind)
            alphas = _run_alphas(job_name, seg_index, gens, kind)
            _finish(job_name, seg_index, alphas, kind, to_pool=to_pool or kind == "full")
        except Exception as exc:
            _write_progress(job_name, seg_index, kind, stage="error", done=0, total=0, error=str(exc))
        finally:
            _LOCK.release()

    _write_progress(job_name, seg_index, kind, stage="auto", done=0, total=0, error="")
    threading.Thread(target=work, name=f"cvfs-matte-{job_name}", daemon=True).start()
    return {"status": "started", "choice": "auto"}


def start_manual(job_name: str, seg_index: int, mask_path: Path, kind: str = "test", to_pool: bool = False) -> dict:
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("Маска уже считается.")
    bundle = bundle_dir(job_name, seg_index, kind)
    saved = bundle / f"manual_mask{mask_path.suffix.lower()}"
    saved.parent.mkdir(parents=True, exist_ok=True)
    if mask_path.resolve() != saved.resolve():
        shutil.copyfile(mask_path, saved)
    # Keep a copy on the test bundle so full render can reuse it.
    test_copy = bundle_dir(job_name, seg_index, "test") / saved.name
    test_copy.parent.mkdir(parents=True, exist_ok=True)
    if test_copy.resolve() != saved.resolve():
        shutil.copyfile(saved, test_copy)
    save_choice(job_name, seg_index, "manual", mask_name=saved.name)

    def work():
        try:
            gens, _p, _t, _fps = _load_generated(job_name, seg_index, kind)
            alphas = _run_alphas(job_name, seg_index, gens, kind, manual_path=saved)
            _finish(job_name, seg_index, alphas, kind, to_pool=to_pool or kind == "full")
        except Exception as exc:
            _write_progress(job_name, seg_index, kind, stage="error", done=0, total=0, error=str(exc))
        finally:
            _LOCK.release()

    _write_progress(job_name, seg_index, kind, stage="manual", done=0, total=0, error="")
    threading.Thread(target=work, name=f"cvfs-matte-manual-{job_name}", daemon=True).start()
    return {"status": "started", "choice": "manual"}


def apply_choice_to_kind(job_name: str, seg_index: int, kind: str = "full") -> dict:
    """After a full Wan pin, apply the mask choice made on the test."""
    choice = read_choice(job_name, seg_index)
    mode = choice.get("mode") or "auto"
    if mode == "manual":
        name = choice.get("mask_name") or ""
        path = bundle_dir(job_name, seg_index, "test") / name
        if not path.is_file():
            path = bundle_dir(job_name, seg_index, kind) / name
        if not path.is_file():
            raise FileNotFoundError("Своя маска не найдена. Сначала выбери файл на тесте.")
        return start_manual(job_name, seg_index, path, kind=kind, to_pool=True)
    return start_auto(job_name, seg_index, kind=kind, to_pool=True)


def assemble_from_pool(job_name: str) -> dict:
    """Glue ready full-plate scenes back into the original timeline. Gaps stay original."""
    import cv2

    job = job_dir(job_name)
    cuts = json.loads((job / "cuts.json").read_text(encoding="utf-8"))
    driver = job / "driver.mp4"
    plates, fps = _read_video(driver)
    if not plates:
        raise FileNotFoundError(driver)
    H, W = plates[0].shape[:2]
    out = [fr.copy() for fr in plates]
    used = []
    for seg in cuts.get("segments") or []:
        sid = str(seg["id"])
        ready = ready_dir(job_name) / f"{sid}.mp4"
        if not ready.is_file():
            continue
        start = int(seg["start"])
        end = int(seg["end"])
        ready_frames, _ = _read_video(ready)
        for i, fr in enumerate(ready_frames):
            idx = start + i
            if idx >= end or idx >= len(out):
                break
            if fr.shape[0] != H or fr.shape[1] != W:
                fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
            out[idx] = fr
        used.append(sid)
    final = job / "final.mp4"
    writer = cv2.VideoWriter(str(final), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (W, H), True)
    for fr in out:
        writer.write(fr)
    writer.release()
    play = job / "final_play.mp4"
    _h264(final, play)
    return {
        "final": str(final),
        "play": _view(play if play.is_file() else final),
        "used": used,
        "frames": len(out),
    }
