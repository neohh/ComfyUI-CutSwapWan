from .nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    _run_all_segments,
    _seg_info,
    ensure_scene_scrub,
    list_master_pack,
    list_job_results,
    list_scene_ranges,
    list_seg_modes,
    queue_one_segment,
    reorder_master_pack,
    save_scene_ref_frame,
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


_register_routes()
