import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

function frameUrl(job, segIndex, offset) {
  const q = new URLSearchParams({
    job,
    seg: String(segIndex),
    offset: String(offset),
  });
  return api.apiURL(`/cvfs/scene_frame?${q.toString()}`);
}

function setWidget(node, name, value) {
  const w = node.widgets?.find((x) => x.name === name);
  if (!w) return;
  w.value = value;
  w.callback?.(value);
}

function pointCropAt(job, segIndex, forceSeg) {
  for (const n of app.graph?._nodes || []) {
    if ((n.type || n.comfyClass) !== "CVFSCropRefInRect") continue;
    const jobW = n.widgets?.find((w) => w.name === "job_name");
    if (job && (!jobW || !String(jobW.value || "").trim())) setWidget(n, "job_name", job);
    if (forceSeg) setWidget(n, "seg_index", segIndex);
  }
}

function jobName() {
  const nodes = app.graph?._nodes || [];
  for (const type of ["CVFSZoneSelect", "CVFSDetectSceneCuts"]) {
    const node = nodes.find((n) => (n.type || n.comfyClass) === type);
    const value = node?.widgets?.find((w) => w.name === "job_name")?.value;
    if (value) return String(value);
  }
  return "";
}

function removeWidget(node, name) {
  const widgets = node.widgets || [];
  for (let i = widgets.length - 1; i >= 0; i--) {
    if (widgets[i].name !== name) continue;
    widgets[i].onRemove?.();
    widgets.splice(i, 1);
  }
}

function isScenePreview(node) {
  const title = node.title || "";
  return title.includes("Scene starts") || title.includes("Кадры сцен");
}

function openSceneEditor() {
  const job = jobName();
  const overlay = document.createElement("div");
  overlay.style.cssText =
    "position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:10000;display:flex;align-items:center;justify-content:center;";
  const panel = document.createElement("div");
  panel.style.cssText =
    "background:#1e1e1e;color:#ddd;padding:16px;border-radius:10px;max-width:980px;width:94%;max-height:90vh;overflow:auto;font:14px sans-serif;";
  panel.innerHTML =
    "<b>Кадры сцен</b>" +
    "<div style='opacity:.75;font-size:12px;margin-top:4px'>Это уже нарезанные сцены. Выбери кадр и нажми «Редактировать» — ползунок ходит только внутри этой сцены.</div>" +
    "<div class='cvfs-ref-status' style='opacity:.8;font-size:12px;margin-top:8px'>Загрузка…</div>" +
    "<div class='cvfs-ref-grid' style='display:flex;flex-wrap:wrap;gap:8px;margin-top:12px'></div>" +
    "<div class='cvfs-ref-editor' style='display:none;margin-top:12px'>" +
    "<img class='cvfs-ref-img' alt='' style='width:100%;max-height:58vh;object-fit:contain;background:#000;border-radius:6px'/>" +
    "<input class='cvfs-ref-range' type='range' min='0' max='0' value='0' style='width:100%;margin-top:8px'/>" +
    "<div class='cvfs-ref-label' style='margin-top:4px;font-size:13px'></div>" +
    "</div>";
  const actions = document.createElement("div");
  actions.style.cssText = "margin-top:12px;display:flex;gap:8px;flex-wrap:wrap;";
  const edit = document.createElement("button");
  edit.textContent = "Редактировать";
  edit.disabled = true;
  const save = document.createElement("button");
  save.textContent = "Этот кадр";
  save.style.display = "none";
  const close = document.createElement("button");
  close.textContent = "Закрыть";
  actions.append(edit, save, close);
  panel.appendChild(actions);
  overlay.appendChild(panel);
  document.body.appendChild(overlay);
  close.onclick = () => overlay.remove();
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) overlay.remove();
  });

  const statusEl = panel.querySelector(".cvfs-ref-status");
  const grid = panel.querySelector(".cvfs-ref-grid");
  const editor = panel.querySelector(".cvfs-ref-editor");
  const img = panel.querySelector(".cvfs-ref-img");
  const range = panel.querySelector(".cvfs-ref-range");
  const label = panel.querySelector(".cvfs-ref-label");
  let scenes = [];
  let selected = -1;
  let editing = false;

  function current() {
    return scenes[selected];
  }

  function showFrame() {
    const seg = current();
    if (!seg) return;
    const offset = Number(range.value) || 0;
    img.src = frameUrl(job, seg.seg_index, offset);
    label.textContent = `SEG${seg.seg_id} · кадр ${offset + 1} из ${seg.n_frames} · в ролике #${seg.start + offset}`;
  }

  function renderGrid() {
    grid.innerHTML = "";
    scenes.forEach((seg, i) => {
      const card = document.createElement("button");
      card.type = "button";
      card.style.cssText =
        "width:160px;padding:0;border-radius:8px;overflow:hidden;cursor:pointer;background:#2a2a2a;color:#ddd;border:2px solid " +
        (i === selected ? "#6af" : "#333") +
        ";";
      const thumb = document.createElement("img");
      thumb.alt = "";
      thumb.style.cssText = "width:160px;height:120px;object-fit:cover;display:block;background:#000;";
      thumb.src = frameUrl(job, seg.seg_index, seg.ref_offset || 0);
      const cap = document.createElement("div");
      cap.style.cssText = "padding:6px 8px;font-size:12px;text-align:left;";
      cap.textContent = `SEG${seg.seg_id} · кадр ${seg.ref_frame}`;
      card.append(thumb, cap);
      card.onclick = () => {
        selected = i;
        editing = false;
        editor.style.display = "none";
        save.style.display = "none";
        edit.disabled = false;
        statusEl.textContent = `Выбрана SEG${seg.seg_id}. Нажми «Редактировать», чтобы двигать кадр внутри сцены.`;
        renderGrid();
      };
      grid.appendChild(card);
    });
  }

  range.oninput = () => showFrame();
  overlay.tabIndex = 0;
  overlay.onkeydown = (e) => {
    if (!editing) return;
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const step = e.key === "ArrowRight" ? 1 : -1;
    range.value = String(Math.max(0, Math.min(Number(range.max), Number(range.value) + step)));
    showFrame();
  };

  edit.onclick = () => {
    const seg = current();
    if (!seg) return;
    editing = true;
    editor.style.display = "";
    save.style.display = "";
    range.max = String(Math.max(0, seg.n_frames - 1));
    range.value = String(seg.ref_offset || 0);
    showFrame();
    statusEl.textContent = `SEG${seg.seg_id}: двигай ползунок в пределах сцены ${seg.start}–${seg.end - 1}.`;
    overlay.focus();
  };

  save.onclick = async () => {
    const seg = current();
    if (!seg) return;
    save.disabled = true;
    statusEl.textContent = "Сохраняю полный кадр…";
    try {
      const resp = await api.fetchApi("/cvfs/scene_ref", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_name: job,
          seg_index: seg.seg_index,
          offset: Number(range.value) || 0,
        }),
      });
      const data = await resp.json();
      if (!resp.ok || data.error) {
        statusEl.textContent = data.error || "не сохранилось";
        return;
      }
      seg.ref_frame = data.ref_frame;
      seg.ref_offset = data.offset;
      pointCropAt(job, seg.seg_index, true);
      renderGrid();
      statusEl.textContent = `SEG${data.seg_id}: запомнен кадр ${data.ref_frame}. Файл ${data.file}`;
    } catch (err) {
      statusEl.textContent = String(err);
    } finally {
      save.disabled = false;
    }
  };

  if (!job) {
    statusEl.textContent = "Не вижу job_name. Он стоит на зоне и на Detect.";
    return;
  }

  (async () => {
    const rangesResp = await api.fetchApi(`/cvfs/scene_ranges?job=${encodeURIComponent(job)}`);
    const ranges = await rangesResp.json();
    if (!rangesResp.ok || ranges.error) {
      statusEl.textContent = ranges.error || "Сцен ещё нет. Сначала Queue на Detect.";
      return;
    }
    scenes = ranges.segments || [];
    if (!scenes.length) {
      statusEl.textContent = "Сцен ещё нет. Сначала Queue на Detect — появятся начальные кадры.";
      return;
    }
    statusEl.textContent = "Собираю кадры сцен…";
    const prep = await api.fetchApi("/cvfs/scrub_prepare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_name: job }),
    });
    const prepared = await prep.json();
    if (!prep.ok || prepared.error) {
      statusEl.textContent = prepared.error || "не собрал кадры";
      return;
    }
    renderGrid();
    statusEl.textContent = "Выбери сцену и нажми «Редактировать».";
  })().catch((err) => {
    statusEl.textContent = String(err);
  });
}

function attachSceneEdit(node) {
  if (!isScenePreview(node)) return;
  const name = "Редактировать";
  const existing = (node.widgets || []).find((w) => w.name === name);
  if (existing) {
    existing.callback = () => openSceneEditor();
    return;
  }
  node.addWidget("button", name, null, () => openSceneEditor());
  node.setSize?.(node.computeSize());
}

app.registerExtension({
  name: "cut_video_face_swapper.ref_frame",
  nodeCreated(node) {
    const type = node.type || node.comfyClass || "";
    if (type === "CVFSDetectSceneCuts") removeWidget(node, "Кадр под мастер");
    attachSceneEdit(node);
  },
  async afterConfigureGraph() {
    for (const node of app.graph?._nodes || []) {
      const type = node.type || node.comfyClass || "";
      if (type === "CVFSDetectSceneCuts") removeWidget(node, "Кадр под мастер");
      attachSceneEdit(node);
    }
    const job = jobName();
    if (job) pointCropAt(job, 0, false);
  },
});
