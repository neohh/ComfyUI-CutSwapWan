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
  const mode = wanMode || "face";
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
    "<b>Сцены</b>" +
    "<div style='opacity:.75;font-size:12px;margin-top:4px'>Выбери сцену по мастеркадру. Режим из «Порядок и режимы». Рендер в очередь, потом финал.</div>" +
    "<div class='cvfs-dirs' style='opacity:.8;font-size:12px;margin-top:8px;white-space:pre-wrap'></div>" +
    "<div class='cvfs-scenes' style='margin-top:10px'></div>";
  const actions = document.createElement("div");
  actions.style.cssText = "margin-top:12px;display:flex;gap:8px;flex-wrap:wrap;";
  const one = document.createElement("button");
  one.textContent = "Рендер выбранную";
  const all = document.createElement("button");
  all.textContent = "Перерендерить все с нуля";
  const checkedBtn = document.createElement("button");
  checkedBtn.textContent = "Перерендерить отмеченные";
  const testBtn = document.createElement("button");
  testBtn.textContent = "Тест 2 сек";
  const testABtn = document.createElement("button");
  testABtn.textContent = "Тест A (swap mimic)";
  const testAfBtn = document.createElement("button");
  testAfBtn.textContent = "Тест Af (fast)";
  const testPBtn = document.createElement("button");
  testPBtn.textContent = "Тест P (polish)";
  const cancel = document.createElement("button");
  cancel.textContent = "Закрыть";
  actions.append(testBtn, testABtn, testAfBtn, testPBtn, one, all, checkedBtn, cancel);
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

  async function queue(indices, title, extra) {
    const testSeconds = Number(extra?.test_seconds) || 0;
    const facePrep = String(extra?.face_images_prep || "");
    const speedProfile = String(extra?.speed_profile || "");
    const resp = await api.fetchApi("/cvfs/run_all", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        job_name: job,
        pad_percent: pad,
        seed: 210,
        only_missing_wan: false,
        queue_assemble_after: !testSeconds && extra?.assemble !== false,
        seg_indices: indices,
        test_seconds: testSeconds,
        face_images_prep: facePrep,
        speed_profile: speedProfile,
      }),
    });
    const data = await resp.json();
    if (!resp.ok || data.error) {
      alert(data.error || "не удалось поставить в очередь");
      return;
    }
    node.title = `[Z2] SEG picker — ${title}`;
    let note = "Смотри очередь Comfy. В конце соберётся final.mp4.";
    if (testSeconds && speedProfile === "fast") {
      note = "Тест Af: swap mimic + окно ~640, VRAM-safe swap≥20. Файл preview/XX_Af_*.mp4.";
    } else if (facePrep === "polish" || facePrep === "p" || facePrep === "swap_polish") {
      note = "Тест P (SwapPolish): ReActor + лёгкая форма/волосы, без Wan. Файл preview/XX_P_*.mp4.";
    } else if (testSeconds && facePrep) {
      note = "Тест A: ReActor на face_images → Wan. Файл preview/XX_A_*.mp4. Финал не собирается.";
    } else if (testSeconds) {
      note = "Короткий кусок вокруг выбранного кадра. Финал не собирается.";
    }
    alert(`${title}\n${note}`);
    overlay.remove();
  }

  function render() {
    listEl.innerHTML = "";
    scenes.forEach((seg, i) => {
      const row = document.createElement("div");
      row.style.cssText =
        "display:flex;gap:10px;align-items:center;padding:6px;margin-bottom:6px;background:#2a2a2a;border-radius:6px;cursor:pointer;border:2px solid " +
        (i === selected ? "#6af" : "transparent") +
        ";";
      const box = document.createElement("input");
      box.type = "checkbox";
      box.checked = !!seg.checked;
      box.onclick = (e) => e.stopPropagation();
      box.onchange = () => {
        seg.checked = box.checked;
      };
      const img = document.createElement("img");
      img.alt = "";
      img.style.cssText = "width:160px;height:90px;object-fit:cover;background:#000;";
      if (seg.master_exists) img.src = masterUrl(job, seg.seg_id);
      const done = seg.full_exists || seg.part_exists ? "готово" : "нет рендера";
      const text = document.createElement("div");
      text.innerHTML =
        `<b>SEG${seg.seg_id}</b> <span style="opacity:.7">${seg.frames || ""}</span><br>` +
        `<span style="opacity:.85">${modeLabel(seg.wan_mode)}</span><br>` +
        `<span style="opacity:.7">${done}</span>`;
      row.append(box, img, text);
      row.onclick = () => {
        selected = i;
        render();
      };
      listEl.appendChild(row);
    });
  }

  testBtn.onclick = () => {
    const seg = scenes[selected];
    if (!seg) return;
    queue([seg.seg_index], `тест 2 сек SEG${seg.seg_id}`, { test_seconds: 2, assemble: false });
  };
  testABtn.onclick = () => {
    const seg = scenes[selected];
    if (!seg) return;
    queue([seg.seg_index], `тест A swap-mimic SEG${seg.seg_id}`, {
      test_seconds: 2,
      assemble: false,
      face_images_prep: "reactor",
    });
  };
  testAfBtn.onclick = () => {
    const seg = scenes[selected];
    if (!seg) return;
    queue([seg.seg_index], `тест Af fast SEG${seg.seg_id}`, {
      test_seconds: 2,
      assemble: false,
      face_images_prep: "reactor_fast",
      speed_profile: "fast",
    });
  };
  testPBtn.onclick = () => {
    const seg = scenes[selected];
    if (!seg) return;
    queue([seg.seg_index], `тест P polish SEG${seg.seg_id}`, {
      test_seconds: 2,
      assemble: false,
      face_images_prep: "polish",
    });
  };
  one.onclick = () => {
    const seg = scenes[selected];
    if (!seg) return;
    queue([seg.seg_index], `рендер SEG${seg.seg_id}`);
  };
  all.onclick = () => {
    if (!confirm("Перерендерить все сцены с нуля и собрать финал? Старые куски не используются.")) return;
    queue(null, "все сцены с нуля");
  };
  checkedBtn.onclick = () => {
    const ids = scenes.filter((s) => s.checked).map((s) => s.seg_index);
    if (!ids.length) {
      alert("Отметь галочками хотя бы одну сцену.");
      return;
    }
    const names = scenes.filter((s) => s.checked).map((s) => `SEG${s.seg_id}`).join(", ");
    queue(ids, `отмеченные: ${names}`);
  };

  if (!job) {
    listEl.textContent = "На ноде пустой job_name.";
    return;
  }
  listEl.textContent = "Загрузка…";
  Promise.all([
    api.fetchApi(`/cvfs/seg_modes?job=${encodeURIComponent(job)}`).then((r) => r.json()),
    api.fetchApi(`/cvfs/job_results?job=${encodeURIComponent(job)}`).then((r) => r.json()),
  ])
    .then(([modesData, resultsData]) => {
      if (modesData.error) {
        listEl.textContent = modesData.error;
        return;
      }
      dirsEl.textContent =
        `parts: ${resultsData.parts_dir || ""}\n` +
        `vace_full: ${resultsData.vace_full_dir || ""}\n` +
        `final: ${resultsData.final_path || "(ещё нет)"}`;
      const byId = {};
      for (const row of resultsData.segments || []) byId[row.seg_id] = row;
      scenes = (modesData.segments || []).map((s) => ({ ...s, ...(byId[s.seg_id] || {}) }));
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

      const legacy = ["Apply SEG → Wan nodes", "Next SEG + Apply", "Queue ALL SEGs → Assemble"];
      for (const w of this.widgets || []) {
        if (legacy.includes(w.name)) w.callback = () => openScenes(this);
      }
      const scenesBtn = (this.widgets || []).find((w) => w.name === "Сцены");
      if (scenesBtn) scenesBtn.callback = () => openScenes(this);
      else this.addWidget("button", "Сцены", null, () => openScenes(this));

      return r;
    };
  },
});
