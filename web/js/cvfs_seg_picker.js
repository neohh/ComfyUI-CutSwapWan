import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

/** Set widget by name if present. */
function setWidget(node, name, value) {
  const w = node.widgets?.find((x) => x.name === name);
  if (!w) return false;
  w.value = value;
  w.callback?.(value);
  return true;
}

function findByTitleIncludes(substr) {
  return (app.graph?._nodes || []).filter((n) => (n.title || "").includes(substr));
}

/**
 * Patch visible [Z2] Wan nodes from /cvfs/seg_info for one SEG.
 */
async function applySegToWanNodes(jobName, segIndex, padPercent, wanMode) {
  const pad = padPercent ?? 20;
  const resp = await api.fetchApi(
    `/cvfs/seg_info?job=${encodeURIComponent(jobName)}&seg=${encodeURIComponent(segIndex)}&pad=${encodeURIComponent(pad)}`
  );
  const info = await resp.json();
  if (!resp.ok || info.error) {
    alert(`SEG info failed: ${info.error || resp.status}`);
    return null;
  }
  if (!info.master_exists) {
    alert(`Нет master для SEG${info.seg_id}. Сначала Upload masters.`);
    return null;
  }

  const z2 = (app.graph?._nodes || []).filter((n) => (n.title || "").includes("[Z2]"));
  if (info.wan_mode === "vace") {
    for (const n of z2) {
      if ((n.type || n.comfyClass) === "CVFSSegPicker") continue;
      n.mode = 2;
    }
    app.graph?.setDirtyCanvas?.(true, true);
    const q = await api.fetchApi("/cvfs/queue_seg", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_name: jobName, seg_index: segIndex, pad_percent: pad }),
    });
    const queued = await q.json();
    if (!q.ok || queued.error) {
      alert(`VACE не встал в очередь: ${queued.error || q.status}`);
      return null;
    }
    info.queued = queued;
    return info;
  }
  for (const n of z2) n.mode = 0;

  for (const n of findByTitleIncludes("[Z2] DRIVER")) {
    setWidget(n, "video", info.driver);
    setWidget(n, "skip_first_frames", info.skip_first_frames);
    setWidget(n, "frame_load_cap", info.frame_load_cap);
    setWidget(n, "custom_width", info.width);
    setWidget(n, "custom_height", info.height);
  }
  for (const n of findByTitleIncludes("[Z2] MASK")) {
    setWidget(n, "video", info.mask);
    setWidget(n, "skip_first_frames", info.skip_first_frames);
    setWidget(n, "frame_load_cap", info.frame_load_cap);
    setWidget(n, "custom_width", info.width);
    setWidget(n, "custom_height", info.height);
  }
  for (const n of findByTitleIncludes("[Z2] MASTER")) {
    setWidget(n, "image", info.master_image);
  }
  for (const n of findByTitleIncludes("[Z2] PoseAndFaceDetection")) {
    setWidget(n, "width", info.width);
    setWidget(n, "height", info.height);
  }
  for (const n of findByTitleIncludes("[Z2] DrawViTPose")) {
    setWidget(n, "width", info.width);
    setWidget(n, "height", info.height);
  }
  for (const n of findByTitleIncludes("[Z2] Wan AnimateEmbeds")) {
    setWidget(n, "width", info.width);
    setWidget(n, "height", info.height);
    setWidget(n, "num_frames", info.n_frames);
    if (info.face_strength != null) setWidget(n, "face_strength", info.face_strength);
    if (info.pose_strength != null) setWidget(n, "pose_strength", info.pose_strength);
  }
  for (const n of findByTitleIncludes("[Z2] Wan Decode")) {
    setWidget(n, "filename_prefix", `${info.filename_prefix}_decoded`);
  }
  for (const n of findByTitleIncludes("[Z2] Wan Sampler")) {
    if (info.seed != null) setWidget(n, "seed", info.seed);
  }
  for (const n of findByTitleIncludes("[Z2] VideoCombine")) {
    setWidget(n, "filename_prefix", info.filename_prefix);
  }

  app.graph?.setDirtyCanvas?.(true, true);
  return info;
}

function masterUrl(job, segId) {
  const q = new URLSearchParams({
    filename: `${segId}.png`,
    type: "input",
    subfolder: `cut_video_face_swapper/jobs/${job}/masters`,
  });
  return api.apiURL(`/view?${q.toString()}`);
}

function openWork(node, scenes, selected) {
  const job = node.widgets?.find((w) => w.name === "job_name")?.value || "";
  const pad = node.widgets?.find((w) => w.name === "pad_percent")?.value ?? 20;
  const seg = scenes[selected];
  if (!seg) return;
  setWidget(node, "seg_index", seg.seg_index);
  if (typeof window.cvfsOpenMatte !== "function") {
    alert("Окно «Одна сцена» не загрузилось. Обнови страницу Comfy (Ctrl+F5).");
    return;
  }
  window.cvfsOpenMatte({
    job,
    segIndex: seg.seg_index,
    segId: seg.seg_id,
    pad,
    scenes,
  });
}

/**
 * Multi-scene overview + pool assemble. Per-scene work is «Одна сцена».
 */
function openScenes(node) {
  const job = node.widgets?.find((w) => w.name === "job_name")?.value || "";
  const pad = node.widgets?.find((w) => w.name === "pad_percent")?.value ?? 20;
  const overlay = document.createElement("div");
  overlay.style.cssText =
    "position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:10000;display:flex;align-items:center;justify-content:center;";
  const panel = document.createElement("div");
  panel.style.cssText =
    "background:#1e1e1e;color:#ddd;padding:16px;border-radius:10px;max-width:920px;width:94%;max-height:88vh;overflow:auto;font:14px sans-serif;";
  panel.innerHTML =
    "<b>Сцены и пул</b>" +
    "<div style='opacity:.75;font-size:12px;margin-top:4px'>Все сцены видео. Открой любую — тест и выбор идут по одной, готовые копятся в пуле.</div>" +
    "<div class='cvfs-dirs' style='opacity:.8;font-size:12px;margin-top:8px;white-space:pre-wrap'></div>" +
    "<div class='cvfs-scenes' style='margin-top:10px'></div>";
  const actions = document.createElement("div");
  actions.style.cssText = "margin-top:12px;display:flex;gap:8px;flex-wrap:wrap;";
  const workBtn = document.createElement("button");
  workBtn.textContent = "Открыть сцену";
  workBtn.style.cssText = "padding:10px 14px;background:#3d6ea8;color:#fff;border:0;border-radius:8px;cursor:pointer;";
  const assembleBtn = document.createElement("button");
  assembleBtn.textContent = "Склеить финал из пула";
  assembleBtn.style.cssText = "padding:10px 14px;background:#2f7d4a;color:#fff;border:0;border-radius:8px;cursor:pointer;";
  const cancel = document.createElement("button");
  cancel.textContent = "Закрыть";
  actions.append(workBtn, assembleBtn, cancel);
  panel.appendChild(actions);
  overlay.appendChild(panel);
  document.body.appendChild(overlay);
  cancel.onclick = () => overlay.remove();
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) overlay.remove();
  });

  const dirsEl = panel.querySelector(".cvfs-dirs");
  const listEl = panel.querySelector(".cvfs-scenes");
  let scenes = [];
  let selected = 0;

  function modeLabel(mode) {
    return mode === "vace" ? "Без лица — VACE" : "Лицо — Wan";
  }

  function statusText(seg) {
    if (seg.in_pool) return "в пуле";
    if (seg.has_test) return "есть тест";
    if (seg.full_exists || seg.part_exists) return "есть рендер";
    return "не начата";
  }

  function render() {
    listEl.innerHTML = "";
    scenes.forEach((seg, i) => {
      const row = document.createElement("div");
      row.style.cssText =
        "display:flex;gap:10px;align-items:center;padding:6px;margin-bottom:6px;background:#2a2a2a;border-radius:6px;cursor:pointer;border:2px solid " +
        (i === selected ? "#6af" : "transparent") +
        ";";
      const img = document.createElement("img");
      img.alt = "";
      img.style.cssText = "width:160px;height:90px;object-fit:cover;background:#000;";
      if (seg.master_exists) img.src = masterUrl(job, seg.seg_id);
      const text = document.createElement("div");
      text.innerHTML =
        `<b>SEG${seg.seg_id}</b> <span style="opacity:.7">${seg.frames || ""}</span><br>` +
        `<span style="opacity:.85">${modeLabel(seg.wan_mode)}</span><br>` +
        `<span style="opacity:.7">${statusText(seg)}${seg.choice ? " · маска " + seg.choice : ""}</span>`;
      row.append(img, text);
      row.onclick = () => {
        selected = i;
        render();
      };
      row.ondblclick = () => {
        selected = i;
        openWork(node, scenes, selected);
        overlay.remove();
      };
      listEl.appendChild(row);
    });
  }

  workBtn.onclick = () => {
    openWork(node, scenes, selected);
    overlay.remove();
  };
  assembleBtn.onclick = async () => {
    const resp = await api.fetchApi("/cvfs/pool_assemble", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_name: job }),
    });
    const data = await resp.json();
    if (!resp.ok || data.error) {
      alert(data.error || "склейка не вышла");
      return;
    }
    alert(`Финал: ${data.final}\nИз пула: ${(data.used || []).map((s) => "SEG" + s).join(", ") || "пусто"}`);
  };

  if (!job) {
    listEl.textContent = "На ноде пустой job_name.";
    return;
  }
  listEl.textContent = "Загрузка…";
  Promise.all([
    api.fetchApi(`/cvfs/seg_modes?job=${encodeURIComponent(job)}`).then((r) => r.json()),
    api.fetchApi(`/cvfs/job_results?job=${encodeURIComponent(job)}`).then((r) => r.json()),
    api.fetchApi(`/cvfs/pool?job=${encodeURIComponent(job)}`).then((r) => r.json()),
  ])
    .then(([modesData, resultsData, poolData]) => {
      if (modesData.error) {
        listEl.textContent = modesData.error;
        return;
      }
      dirsEl.textContent =
        `пул: ${poolData.ready_dir || ""}\n` +
        `в пуле: ${poolData.pool_count || 0}\n` +
        `final: ${poolData.final || "(ещё нет)"}`;
      const byId = {};
      for (const row of resultsData.segments || []) byId[row.seg_id] = row;
      const poolById = {};
      for (const row of poolData.segments || []) poolById[row.seg_id] = row;
      scenes = (modesData.segments || []).map((s) => ({
        ...s,
        ...(byId[s.seg_id] || {}),
        ...(poolById[s.seg_id] || {}),
      }));
      const current = Number(node.widgets?.find((w) => w.name === "seg_index")?.value || 0);
      selected = Math.max(
        0,
        scenes.findIndex((s) => s.seg_index === current)
      );
      if (selected < 0) selected = 0;
      if (!scenes.length) listEl.textContent = "Нет сцен. Сначала Detect.";
      else render();
    })
    .catch((e) => {
      listEl.textContent = String(e);
    });
}

app.registerExtension({
  name: "cut_video_face_swapper.seg_picker",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "CVFSSegPicker") return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);

      const legacy = ["Apply SEG → Wan nodes", "Next SEG + Apply", "Queue ALL SEGs → Assemble", "Сцены", "Прогон и маска"];
      for (const w of [...(this.widgets || [])]) {
        if (legacy.includes(w.name)) {
          const idx = this.widgets.indexOf(w);
          if (idx >= 0) this.widgets.splice(idx, 1);
        }
      }

      this.addWidget("button", "Нарезка", null, () => {
        const job = this.widgets?.find((w) => w.name === "job_name")?.value || "";
        if (typeof window.cvfsOpenCut !== "function") {
          alert("Окно нарезки не загрузилось. Ctrl+F5.");
          return;
        }
        window.cvfsOpenCut({
          job,
          onScenes: () => window.cvfsOpenMasters?.({ job }),
        });
      });
      this.addWidget("button", "Мастера", null, () => {
        const job = this.widgets?.find((w) => w.name === "job_name")?.value || "";
        if (typeof window.cvfsOpenMasters !== "function") {
          alert("Окно мастеров не загрузилось. Ctrl+F5.");
          return;
        }
        window.cvfsOpenMasters({ job });
      });
      this.addWidget("button", "Работа", null, () => {
        const job = this.widgets?.find((w) => w.name === "job_name")?.value || "";
        const pad = this.widgets?.find((w) => w.name === "pad_percent")?.value ?? 20;
        const segIndex = Number(this.widgets?.find((w) => w.name === "seg_index")?.value || 0);
        api
          .fetchApi(`/cvfs/pool?job=${encodeURIComponent(job)}`)
          .then((r) => r.json())
          .then((pool) => {
            const scenes = (pool.segments || []).map((s) => ({
              seg_index: s.seg_index,
              seg_id: s.seg_id,
              frames: s.frames,
            }));
            const seg = scenes[segIndex] || { seg_index: segIndex, seg_id: String(segIndex).padStart(2, "0") };
            if (typeof window.cvfsOpenMatte !== "function") {
              alert("Окно не загрузилось. Ctrl+F5.");
              return;
            }
            window.cvfsOpenMatte({
              job,
              segIndex: seg.seg_index,
              segId: seg.seg_id,
              pad,
              scenes: scenes.length ? scenes : [seg],
            });
          })
          .catch((e) => alert(String(e)));
      });
      this.addWidget("button", "Сцены / пул", null, () => openScenes(this));

      return r;
    };
  },
});

// Keep applySegToWanNodes reachable for future canvas wiring.
window.cvfsApplySegToWanNodes = applySegToWanNodes;
