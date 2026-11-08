from .nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    _run_all_segments,
    _seg_info,
    detect_status,
    ensure_scene_scrub,
    list_master_board,
    list_master_pack,
    list_job_results,
    masters_sources_zip,
    save_master_upload,
    scene_source_file,
    list_scene_ranges,
    list_seg_modes,
    queue_one_segment,
    reorder_master_pack,
    save_scene_ref_frame,
    start_detect_upload,
    scene_scrub_file,
    set_seg_wan_mode,
    set_seg_wan_modes,
)

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]


def _register_routes():
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return

    @PromptServer.instance.routes.get("/cvfs/seg_info")
    async def cvfs_seg_info(request):
        job = request.rel_url.query.get("job", "")
        seg = request.rel_url.query.get("seg", "0")
        try:
            info = _seg_info(
                job,
                int(seg),
                pad_percent=float(request.rel_url.query.get("pad", "20")),
                wan_mode=request.rel_url.query.get("wan_mode"),
            )
            return web.json_response(info)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @PromptServer.instance.routes.get("/cvfs/seg_modes")
    async def cvfs_seg_modes(request):
        job = request.rel_url.query.get("job", "")
        force = request.rel_url.query.get("force", "") in {"1", "true", "yes"}
        try:
            return web.json_response(list_seg_modes(job, assign_auto=True, force=force))
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @PromptServer.instance.routes.get("/cvfs/scene_ranges")
    async def cvfs_scene_ranges(request):
        job = request.rel_url.query.get("job", "")
        try:
            return web.json_response(list_scene_ranges(job))
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @PromptServer.instance.routes.post("/cvfs/scrub_prepare")
    async def cvfs_scrub_prepare(request):
        try:
            import asyncio

            body = await request.json()
            job = body.get("job_name") or body.get("job") or ""
            result = await asyncio.get_running_loop().run_in_executor(None, lambda: ensure_scene_scrub(job))
            return web.json_response(result)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @PromptServer.instance.routes.get("/cvfs/scene_frame")
    async def cvfs_scene_frame(request):
        job = request.rel_url.query.get("job", "")
        try:
            path = scene_scrub_file(
                job,
                int(request.rel_url.query.get("seg", "0")),
                int(request.rel_url.query.get("offset", "0")),
            )
            return web.FileResponse(path)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @PromptServer.instance.routes.post("/cvfs/scene_ref")
    async def cvfs_scene_ref(request):
        try:
            import asyncio

            body = await request.json()
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: save_scene_ref_frame(
                    body.get("job_name") or body.get("job") or "",
                    int(body.get("seg_index", body.get("seg", 0))),
                    int(body.get("offset", 0)),
                ),
            )
            return web.json_response(result)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @PromptServer.instance.routes.get("/cvfs/job_results")
    async def cvfs_job_results(request):
        job = request.rel_url.query.get("job", "")
        try:
            return web.json_response(list_job_results(job))
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @PromptServer.instance.routes.post("/cvfs/queue_seg")
    async def cvfs_queue_seg(request):
        try:
            import asyncio

            body = await request.json()
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: queue_one_segment(
                    body.get("job_name") or body.get("job") or "",
                    int(body.get("seg_index", body.get("seg", 0))),
                    pad_percent=float(body.get("pad_percent", body.get("pad", 20))),
                ),
            )
            return web.json_response(result)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @PromptServer.instance.routes.post("/cvfs/set_wan_modes")
    async def cvfs_set_wan_modes(request):
        try:
            body = await request.json()
            result = set_seg_wan_modes(body.get("job_name") or body.get("job") or "", list(body.get("modes") or []))
            return web.json_response(result)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @PromptServer.instance.routes.post("/cvfs/set_wan_mode")
    async def cvfs_set_wan_mode(request):
        try:
            body = await request.json()
            result = set_seg_wan_mode(
                body.get("job_name") or body.get("job") or "",
                int(body.get("seg_index", body.get("seg", 0))),
                body.get("wan_mode") or "face",
            )
            return web.json_response(result)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @PromptServer.instance.routes.post("/cvfs/detect_upload")
    async def cvfs_detect_upload(request):
        try:
            from pathlib import Path

            import folder_paths

            data = await request.post()
            job = str(data.get("job_name") or "").strip()
            video = data.get("video")
            mask = data.get("mask")
            if not job:
                return web.json_response({"error": "Пустое имя задачи"}, status=400)
            if video is None or not getattr(video, "file", None):
                return web.json_response({"error": "Нет видео"}, status=400)
            if mask is None or not getattr(mask, "file", None):
                return web.json_response({"error": "Нет маски"}, status=400)
            folder = Path(folder_paths.get_input_directory()) / "cut_video_face_swapper" / "jobs" / job / "upload"
            folder.mkdir(parents=True, exist_ok=True)
            video_path = folder / "driver.mp4"
            mask_path = folder / "mask.mp4"
            video_path.write_bytes(video.file.read())
            mask_path.write_bytes(mask.file.read())
            return web.json_response(start_detect_upload(job, video_path, mask_path))
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @PromptServer.instance.routes.get("/cvfs/detect_status")
    async def cvfs_detect_status(request):
        job = request.rel_url.query.get("job", "")
        try:
            return web.json_response(detect_status(job))
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @PromptServer.instance.routes.get("/cvfs/master_board")
    async def cvfs_master_board(request):
        job = request.rel_url.query.get("job", "")
        try:
            return web.json_response(list_master_board(job))
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @PromptServer.instance.routes.get("/cvfs/scene_source")
    async def cvfs_scene_source(request):
        job = request.rel_url.query.get("job", "")
        try:
            import asyncio

            path = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: scene_source_file(job, int(request.rel_url.query.get("seg", "0"))),
            )
            sid = path.stem.split("_")[0]
            return web.FileResponse(
                path,
                headers={"Content-Disposition": f'attachment; filename="SEG{sid}.png"'},
            )
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @PromptServer.instance.routes.get("/cvfs/scenes_zip")
    async def cvfs_scenes_zip(request):
        job = request.rel_url.query.get("job", "")
        try:
            import asyncio

            path = await asyncio.get_running_loop().run_in_executor(None, lambda: masters_sources_zip(job))
            return web.FileResponse(
                path,
                headers={"Content-Disposition": f'attachment; filename="{job}_scenes.zip"'},
            )
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @PromptServer.instance.routes.post("/cvfs/master_upload")
    async def cvfs_master_upload(request):
        try:
            import asyncio

            data = await request.post()
            job = str(data.get("job_name") or "")
            seg = int(data.get("seg_index") or 0)
            field = data.get("master")
            if field is None or not getattr(field, "file", None):
                return web.json_response({"error": "Файл не пришёл"}, status=400)
            raw = field.file.read()
            result = await asyncio.get_running_loop().run_in_executor(
                None, lambda: save_master_upload(job, seg, raw)
            )
            return web.json_response(result)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @PromptServer.instance.routes.get("/cvfs/masters_list")
    async def cvfs_masters_list(request):
        folder = request.rel_url.query.get("folder", "")
        try:
            return web.json_response(list_master_pack(folder))
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @PromptServer.instance.routes.post("/cvfs/masters_reorder")
    async def cvfs_masters_reorder(request):
        try:
            body = await request.json()
            result = reorder_master_pack(
                body.get("folder") or "",
                list(body.get("names") or []),
                body.get("job_name") or "",
            )
            return web.json_response(result)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @PromptServer.instance.routes.post("/cvfs/run_all")
    async def cvfs_run_all(request):
        """Kick off crop+enqueue in a background thread — preparing mattes can exceed HTTP timeouts."""
        try:
            import logging
            import threading

            body = await request.json()
            job_name = body.get("job_name") or ""
            pad_percent = float(body.get("pad_percent", 20))
            seed = int(body.get("seed", 210))
            only_missing_wan = bool(body.get("only_missing_wan", True))
            queue_assemble_after = bool(body.get("queue_assemble_after", True))
            comfy_url = str(body.get("comfy_url") or "http://127.0.0.1:8188")
            raw_indices = body.get("seg_indices")
            seg_indices = None if raw_indices is None else [int(i) for i in raw_indices]
            test_seconds = float(body.get("test_seconds") or 0)
            face_images_prep = str(body.get("face_images_prep") or "").strip()
            speed_profile = str(body.get("speed_profile") or "").strip()
            if not job_name.strip():
                return web.json_response({"error": "job_name required"}, status=400)

            def _worker():
                try:
                    report = _run_all_segments(
                        job_name,
                        pad_percent=pad_percent,
                        seed=seed,
                        only_missing_wan=only_missing_wan,
                        queue_assemble_after=queue_assemble_after,
                        comfy_url=comfy_url,
                        seg_indices=seg_indices,
                        test_seconds=test_seconds,
                        face_images_prep=face_images_prep,
                        speed_profile=speed_profile,
                    )
                    logging.getLogger("cvfs").info("run_all done: %s", report)
                except Exception:
                    logging.getLogger("cvfs").exception("run_all failed job=%s", job_name)

            threading.Thread(target=_worker, name=f"cvfs-run-all-{job_name}", daemon=True).start()
            return web.json_response(
                {
                    "status": "started",
                    "job": job_name,
                    "pad_percent": pad_percent,
                    "only_missing_wan": only_missing_wan,
                    "queue_assemble_after": queue_assemble_after,
                    "note": "Crops + queue run in background. Watch Comfy queue — SEG prompts appear as each window is ready.",
                }
            )
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @PromptServer.instance.routes.get("/cvfs/matte_status")
    async def cvfs_matte_status(request):
        try:
            import asyncio

            from .matte import status

            job = request.rel_url.query.get("job", "")
            seg = int(request.rel_url.query.get("seg", "0"))
            since = float(request.rel_url.query.get("since", "0") or 0)
            kind = str(request.rel_url.query.get("kind") or "test")
            result = await asyncio.get_running_loop().run_in_executor(
                None, lambda: status(job, seg, since, kind)
            )
            return web.json_response(result)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @PromptServer.instance.routes.get("/cvfs/pool")
    async def cvfs_pool(request):
        try:
            import asyncio

            from .matte import list_pool

            job = request.rel_url.query.get("job", "")
            result = await asyncio.get_running_loop().run_in_executor(None, lambda: list_pool(job))
            return web.json_response(result)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @PromptServer.instance.routes.post("/cvfs/pool_assemble")
    async def cvfs_pool_assemble(request):
        try:
            import asyncio

            from .matte import assemble_from_pool

            body = await request.json()
            job = body.get("job_name") or body.get("job") or ""
            result = await asyncio.get_running_loop().run_in_executor(None, lambda: assemble_from_pool(job))
            return web.json_response(result)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @PromptServer.instance.routes.post("/cvfs/matte_pin")
    async def cvfs_matte_pin(request):
        try:
            import asyncio
            from pathlib import Path as _Path

            import folder_paths

            from .matte import job_dir, newest_generation, pin_generation, preview_dir

            body = await request.json()
            job = body.get("job_name") or body.get("job") or ""
            seg = int(body.get("seg_index", body.get("seg", 0)))
            since = float(body.get("since") or 0)
            name = str(body.get("filename") or "")
            sid = str(body.get("seg_id") or f"{seg:02d}")
            kind = str(body.get("kind") or "test")
            source = None
            if name:
                cand = preview_dir(job) / name
                if not cand.is_file():
                    cand = job_dir(job) / "parts" / name
                if not cand.is_file():
                    cand = (
                        _Path(folder_paths.get_output_directory())
                        / "cut_video_face_swapper"
                        / job
                        / "parts"
                        / name
                    )
                source = cand if cand.is_file() else None
            if source is None:
                source = newest_generation(job, sid, since, kind)
            if source is None or not source.is_file():
                return web.json_response({"error": "Новый прогон ещё не появился"}, status=400)
            result = await asyncio.get_running_loop().run_in_executor(
                None, lambda: pin_generation(job, seg, source, kind)
            )
            return web.json_response(result)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @PromptServer.instance.routes.post("/cvfs/matte_frame")
    async def cvfs_matte_frame(request):
        try:
            import asyncio

            from .matte import frame_preview

            body = await request.json()
            job = body.get("job_name") or body.get("job") or ""
            seg = int(body.get("seg_index", body.get("seg", 0)))
            kind = str(body.get("kind") or "test")
            result = await asyncio.get_running_loop().run_in_executor(
                None, lambda: frame_preview(job, seg, kind)
            )
            return web.json_response(result)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @PromptServer.instance.routes.post("/cvfs/matte_auto")
    async def cvfs_matte_auto(request):
        try:
            from .matte import start_auto

            body = await request.json()
            job = body.get("job_name") or body.get("job") or ""
            seg = int(body.get("seg_index", body.get("seg", 0)))
            kind = str(body.get("kind") or "test")
            to_pool = bool(body.get("to_pool"))
            return web.json_response(start_auto(job, seg, kind=kind, to_pool=to_pool))
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @PromptServer.instance.routes.post("/cvfs/matte_manual")
    async def cvfs_matte_manual(request):
        try:
            from pathlib import Path

            from .matte import bundle_dir, start_manual

            data = await request.post()
            job = str(data.get("job_name") or "")
            seg = int(data.get("seg_index") or 0)
            kind = str(data.get("kind") or "test")
            to_pool = str(data.get("to_pool") or "") in {"1", "true", "True"}
            field = data.get("mask")
            if field is None or not getattr(field, "file", None):
                return web.json_response({"error": "Файл маски не пришёл"}, status=400)
            name = Path(str(getattr(field, "filename", "") or "mask.png")).name
            dest = bundle_dir(job, seg, kind) / f"upload_{name}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(field.file.read())
            save_only = str(data.get("save_only") or "") in {"1", "true", "True"}
            if save_only:
                from .matte import save_choice

                save_choice(job, seg, "manual", mask_name=dest.name)
                return web.json_response({"status": "saved", "choice": "manual"})
            return web.json_response(start_manual(job, seg, dest, kind=kind, to_pool=to_pool))
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @PromptServer.instance.routes.post("/cvfs/matte_apply_choice")
    async def cvfs_matte_apply_choice(request):
        try:
            from .matte import apply_choice_to_kind

            body = await request.json()
            job = body.get("job_name") or body.get("job") or ""
            seg = int(body.get("seg_index", body.get("seg", 0)))
            kind = str(body.get("kind") or "full")
            return web.json_response(apply_choice_to_kind(job, seg, kind))
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)


_register_routes()
