import { api } from "../../../scripts/api.js";

function viewUrl(info) {
  if (!info) return "";
  const q = new URLSearchParams({
    filename: info.filename,
    type: info.type,
    subfolder: info.subfolder || "",
  });
  return api.apiURL(`/view?${q.toString()}`);
}

function btn(label, color) {
  const el = document.createElement("button");
  el.type = "button";
  el.textContent = label;
  el.style.cssText = `padding:8px 12px;font-size:14px;border:0;border-radius:8px;cursor:pointer;color:#fff;background:${color || "#444"};`;
  return el;
}

/**
 * Scene stills and masters in one window.
 * Download one frame or all of them, then drop the master onto the same scene.
 */
window.cvfsOpenCut = function openCut({ job, onScenes }) {
  const overlay = document.createElement("div");
  overlay.style.cssText =
    "position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:10002;display:flex;align-items:center;justify-content:center;";
  const panel = document.createElement("div");
  panel.style.cssText =
    "background:#1e1e1e;color:#ddd;padding:16px;border-radius:10px;max-width:520px;width:94%;font:14px sans-serif;";
  panel.innerHTML =
    `<b>Нарезка</b>` +
    `<div class="cvfs-cut-status" style="margin-top:10px;min-height:1.4em">Видео и маска с компьютера.</div>`;
  const video = document.createElement("input");
  video.type = "file";
  video.accept = "video/*";
  const mask = document.createElement("input");
  mask.type = "file";
  mask.accept = "video/*";
  const videoLabel = document.createElement("div");
  const maskLabel = document.createElement("div");
  videoLabel.style.marginTop = "12px";
  maskLabel.style.marginTop = "8px";
  videoLabel.textContent = "Видео";
  maskLabel.textContent = "Маска";
  video.style.display = "block";
  mask.style.display = "block";
  mask.style.marginTop = "4px";
  const actions = document.createElement("div");
  actions.style.cssText = "margin-top:14px;display:flex;gap:8px;flex-wrap:wrap;";
  const go = btn("Нарезать", "#3d6ea8");
  const scenes = btn("К сценам", "#2f7d4a");
  scenes.style.display = "none";
  const close = btn("Закрыть", "#444");
  actions.append(go, scenes, close);
  panel.append(videoLabel, video, maskLabel, mask, actions);
  overlay.appendChild(panel);
  document.body.appendChild(overlay);
  const statusEl = panel.querySelector(".cvfs-cut-status");
  close.onclick = () => overlay.remove();
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) overlay.remove();
  });
  scenes.onclick = () => {
    overlay.remove();
    if (typeof onScenes === "function") onScenes();
    else if (typeof window.cvfsOpenMasters === "function") window.cvfsOpenMasters({ job });
  };

  go.onclick = async () => {
    if (!video.files?.[0] || !mask.files?.[0]) {
      statusEl.textContent = "Нужны оба файла: видео и маска.";
      return;
    }
    go.disabled = true;
    statusEl.textContent = "Загружаю…";
    const body = new FormData();
    body.append("job_name", job);
    body.append("video", video.files[0]);
    body.append("mask", mask.files[0]);
    try {
      const resp = await api.fetchApi("/cvfs/detect_upload", { method: "POST", body });
      const data = await resp.json();
      if (!resp.ok || data.error) throw new Error(data.error || "не стартовало");
      for (let i = 0; i < 600; i++) {
        const cur = await api.fetchApi(`/cvfs/detect_status?job=${encodeURIComponent(job)}`).then((r) => r.json());
        if (cur.stage === "error") throw new Error(cur.error || "нарезка не вышла");
        if (cur.stage === "done") {
          statusEl.textContent = `Готово: ${cur.scenes} сцен. Кадры сохранены.`;
          scenes.style.display = "";
          go.disabled = false;
          return;
        }
        statusEl.textContent = cur.stage === "read" ? "Читаю ролик…" : "Режу на сцены…";
        await new Promise((r) => setTimeout(r, 1500));
      }
      throw new Error("Долго режется. Окно можно закрыть и открыть снова.");
    } catch (err) {
      statusEl.textContent = String(err.message || err);
      go.disabled = false;
    }
  };
};

window.cvfsOpenMasters = function openMasters({ job }) {
  const overlay = document.createElement("div");
  overlay.style.cssText =
    "position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:10002;display:flex;align-items:center;justify-content:center;";
  const panel = document.createElement("div");
  panel.style.cssText =
    "background:#1e1e1e;color:#ddd;padding:16px;border-radius:10px;max-width:980px;width:96%;max-height:92vh;overflow:auto;font:14px sans-serif;";
  panel.innerHTML =
    `<div style="display:flex;justify-content:space-between;gap:12px;align-items:center">` +
    `<b>Мастера</b><span class="cvfs-m-status" style="opacity:.8"></span></div>` +
    `<div class="cvfs-m-list" style="margin-top:12px;display:flex;flex-direction:column;gap:12px"></div>`;

  const actions = document.createElement("div");
  actions.style.cssText = "margin-top:14px;display:flex;gap:8px;flex-wrap:wrap;";
  const allBtn = btn("Скачать все", "#3d6ea8");
  const close = btn("Закрыть", "#444");
  actions.append(allBtn, close);
  panel.appendChild(actions);
  overlay.appendChild(panel);
  document.body.appendChild(overlay);
  close.onclick = () => overlay.remove();
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) overlay.remove();
  });

  const listEl = panel.querySelector(".cvfs-m-list");
  const statusEl = panel.querySelector(".cvfs-m-status");
  allBtn.onclick = () => {
    window.location.href = api.apiURL(`/cvfs/scenes_zip?job=${encodeURIComponent(job)}`);
  };

  function sourceUrl(seg) {
    if (seg.source) return viewUrl(seg.source) + "&t=" + (seg._rev || 0);
    const q = new URLSearchParams({ job, seg: String(seg.seg_index), offset: String(seg.ref_offset || 0) });
    return api.apiURL(`/cvfs/scene_frame?${q.toString()}`);
  }

  function scrubUrl(seg, offset) {
    const q = new URLSearchParams({ job, seg: String(seg.seg_index), offset: String(offset) });
    return api.apiURL(`/cvfs/scene_frame?${q.toString()}`);
  }

  async function commitFrame(seg, offset, cap) {
    const resp = await api.fetchApi("/cvfs/scene_ref", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_name: job, seg_index: seg.seg_index, offset }),
    });
    const data = await resp.json();
    if (!resp.ok || data.error) {
      cap.textContent = data.error || "кадр не сохранился";
      return;
    }
    seg.ref_offset = data.offset;
    seg.ref_frame = data.ref_frame;
    cap.textContent = `кадр ${data.offset + 1} / ${seg.n_frames} · опора для теста`;
  }

  function card(seg) {
    const row = document.createElement("div");
    row.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:12px;background:#222;border-radius:10px;padding:10px;";
    const left = document.createElement("div");
    const right = document.createElement("div");
    const src = document.createElement("img");
    src.alt = "";
    src.style.cssText = "width:100%;max-height:280px;object-fit:contain;background:#000;border-radius:6px;";
    src.src = sourceUrl(seg);
    const master = document.createElement("img");
    master.alt = "";
    master.style.cssText = "width:100%;max-height:280px;object-fit:contain;background:#111;border-radius:6px;";
    if (seg.master) master.src = viewUrl(seg.master) + "&t=" + Date.now();
    else master.style.display = "none";
    const empty = document.createElement("div");
    empty.textContent = "мастера нет";
    empty.style.cssText = "height:160px;display:flex;align-items:center;justify-content:center;background:#111;border-radius:6px;opacity:.6;";
    if (seg.master) empty.style.display = "none";

    const cap = document.createElement("div");
    cap.style.cssText = "margin-top:6px;font-size:12px;opacity:.75;";
    const n = Math.max(1, seg.n_frames || 1);
    cap.textContent = `кадр ${(seg.ref_offset || 0) + 1} / ${n}`;

    const range = document.createElement("input");
    range.type = "range";
    range.min = "0";
    range.max = String(n - 1);
    range.value = String(Math.max(0, Math.min(n - 1, seg.ref_offset || 0)));
    range.style.cssText = "width:100%;margin-top:8px;";
    range.oninput = () => {
      const offset = Number(range.value) || 0;
      seg.ref_offset = offset;
      cap.textContent = `кадр ${offset + 1} / ${n}`;
      src.src = scrubUrl(seg, offset);
    };
    range.onchange = () => commitFrame(seg, Number(range.value) || 0, cap);

    const leftBtns = document.createElement("div");
    leftBtns.style.cssText = "display:flex;gap:6px;margin-top:8px;align-items:center;";
    const mid = btn("Середина", "#333");
    const one = document.createElement("a");
    one.textContent = "↓";
    one.style.cssText = "padding:8px 12px;border-radius:8px;background:#3d6ea8;color:#fff;text-decoration:none;";
    one.href = api.apiURL(`/cvfs/scene_source?job=${encodeURIComponent(job)}&seg=${seg.seg_index}`);
    mid.onclick = () => {
      const offset = Math.floor((n - 1) / 2);
      range.value = String(offset);
      range.oninput();
      range.onchange();
    };
    leftBtns.append(mid, one);

    const file = document.createElement("input");
    file.type = "file";
    file.accept = "image/*";
    file.style.display = "none";
    const up = btn(seg.master ? "Заменить" : "Загрузить", "#2f7d4a");
    up.onclick = () => file.click();
    file.onchange = async () => {
      const picked = file.files && file.files[0];
      file.value = "";
      if (!picked) return;
      up.disabled = true;
      const body = new FormData();
      body.append("job_name", job);
      body.append("seg_index", String(seg.seg_index));
      body.append("master", picked);
      const resp = await api.fetchApi("/cvfs/master_upload", { method: "POST", body });
      const data = await resp.json();
      up.disabled = false;
      if (!resp.ok || data.error) {
        statusEl.textContent = data.error || "не сохранилось";
        return;
      }
      seg.master = data.master;
      seg.master_exists = true;
      master.src = viewUrl(data.master) + "&t=" + Date.now();
      master.style.display = "";
      empty.style.display = "none";
      up.textContent = "Заменить";
      statusEl.textContent = `SEG${seg.seg_id} сохранён`;
    };

    left.innerHTML = `<div style="margin-bottom:6px"><b>SEG${seg.seg_id}</b> <span style="opacity:.7">исходный</span></div>`;
    left.append(src, range, cap, leftBtns);
    right.innerHTML = `<div style="margin-bottom:6px"><b>мастер</b></div>`;
    right.append(master, empty, up, file);
    up.style.marginTop = "8px";
    row.append(left, right);
    return row;
  }

  statusEl.textContent = "Загрузка…";
  api
    .fetchApi(`/cvfs/master_board?job=${encodeURIComponent(job)}`)
    .then((r) => r.json())
    .then((data) => {
      if (data.error) {
        statusEl.textContent = data.error;
        return;
      }
      const segs = data.segments || [];
      const ready = segs.filter((s) => s.master_exists).length;
      statusEl.textContent = segs.length ? `${ready} / ${segs.length}` : "Нет сцен. Сначала нарезка.";
      listEl.innerHTML = "";
      segs.forEach((seg) => listEl.appendChild(card(seg)));
    })
    .catch((e) => {
      statusEl.textContent = String(e);
    });
};
