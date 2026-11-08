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
  el.textContent = label;
  el.style.cssText = `padding:11px 14px;font-size:14px;border:0;border-radius:8px;cursor:pointer;color:#fff;background:${color || "#444"};`;
  return el;
}

function stepLabel(n, text, active) {
  return `<div style="padding:8px 10px;border-radius:8px;background:${active ? "#2a3a4a" : "#252525"};border:1px solid ${active ? "#6af" : "#333"}"><b>${n}.</b> ${text}</div>`;
}

/**
 * Cuts and masters are once for the whole video.
 * Test and mask choice run on one scene, then the next, until the pool is full.
 */
window.cvfsOpenMatte = function openMatte({ job, segIndex, segId, pad, scenes }) {
  let index = Number(segIndex) || 0;
  let id = String(segId || "").padStart(2, "0");
  let sceneRows = Array.isArray(scenes) ? scenes.slice() : [];

  const overlay = document.createElement("div");
  overlay.style.cssText =
    "position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:10001;display:flex;align-items:center;justify-content:center;";
  const panel = document.createElement("div");
  panel.style.cssText =
    "background:#1e1e1e;color:#ddd;padding:16px;border-radius:10px;max-width:1080px;width:96%;max-height:92vh;overflow:auto;font:14px sans-serif;";
  panel.innerHTML =
    `<div style="display:flex;justify-content:space-between;gap:12px;align-items:baseline">` +
    `<b>Работа</b> <span class="cvfs-sid" style="opacity:.8">SEG${id}</span></div>` +
    `<div class="cvfs-scenes" style="display:flex;gap:6px;flex-wrap:wrap;margin-top:10px"></div>` +
    `<div class="cvfs-status" style="margin-top:10px;min-height:1.4em"></div>` +
    `<div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:10px">` +
    `<video class="cvfs-video" controls playsinline style="max-width:480px;max-height:360px;background:#000"></video>` +
    `<img class="cvfs-frame" alt="" style="max-width:480px;max-height:360px;background:#111;object-fit:contain"/>` +
    `</div>` +
    `<video class="cvfs-result" controls playsinline style="display:none;margin-top:10px;max-width:100%;max-height:320px;background:#000"></video>`;

  const actions = document.createElement("div");
  actions.style.cssText = "margin-top:14px;display:flex;gap:8px;flex-wrap:wrap;align-items:center;";
  const runBtn = btn("Тест", "#3d6ea8");
  const autoBtn = btn("Авто — вся сцена в пул", "#2f7d4a");
  const ownBtn = btn("Своя маска…", "#b86a2a");
  const fullBtn = btn("В пул", "#6a3d9a");
  fullBtn.style.display = "none";
  const nextBtn = btn("Следующая сцена", "#555");
  nextBtn.style.display = "none";
  const stitchBtn = btn("Склеить все", "#1f6b45");
  stitchBtn.style.display = "none";
  const mastersBtn = btn("Мастера", "#555");
  const close = btn("Закрыть", "#444");
  const file = document.createElement("input");
  file.type = "file";
  file.accept = "video/*,image/*";
  file.style.display = "none";
  actions.append(runBtn, autoBtn, ownBtn, fullBtn, nextBtn, stitchBtn, mastersBtn, close, file);
  mastersBtn.onclick = () => {
    if (typeof window.cvfsOpenMasters !== "function") {
      statusEl.textContent = "Окно мастеров не загрузилось. Ctrl+F5.";
      return;
    }
    window.cvfsOpenMasters({ job });
  };
  panel.appendChild(actions);
  overlay.appendChild(panel);
  document.body.appendChild(overlay);

  const statusEl = panel.querySelector(".cvfs-status");
  const sidEl = panel.querySelector(".cvfs-sid");
  const scenesEl = panel.querySelector(".cvfs-scenes");
  const videoEl = panel.querySelector(".cvfs-video");
  const frameEl = panel.querySelector(".cvfs-frame");
  const resultEl = panel.querySelector(".cvfs-result");
  let timer = 0;

  function setBusy(busy) {
    runBtn.disabled = busy;
    autoBtn.disabled = busy;
    ownBtn.disabled = busy;
    fullBtn.disabled = busy;
  }

  function chipMark(seg) {
    if (seg.in_pool) return "в пуле";
    if (seg.has_test) return "тест";
    if (seg.master_exists === false) return "нет мастера";
    return "ждать";
  }

  function paintScenes() {
    scenesEl.innerHTML = "";
    const pos = sceneRows.findIndex((s) => Number(s.seg_index) === index);
    sidEl.textContent = sceneRows.length ? `сцена ${pos + 1} из ${sceneRows.length} · SEG${id}` : `SEG${id}`;
    sceneRows.forEach((seg, i) => {
      const on = Number(seg.seg_index) === index;
      const el = btn(`SEG${seg.seg_id} · ${chipMark(seg)}`, on ? "#3d6ea8" : "#333");
      el.style.padding = "6px 8px";
      el.style.fontSize = "12px";
      el.onclick = () => switchSeg(i);
      scenesEl.append(el);
    });
  }

  function nextOpenIndex() {
    if (!sceneRows.length) return -1;
    const cur = sceneRows.findIndex((s) => Number(s.seg_index) === index);
    for (let k = 1; k <= sceneRows.length; k++) {
      const s = sceneRows[(cur + k) % sceneRows.length];
      if (s && !s.in_pool && Number(s.seg_index) !== index) return (cur + k) % sceneRows.length;
    }
    return -1;
  }

  async function paintPrep() {
    try {
      const [modes, poolData] = await Promise.all([
        api.fetchApi(`/cvfs/seg_modes?job=${encodeURIComponent(job)}`).then((r) => r.json()),
        api.fetchApi(`/cvfs/pool?job=${encodeURIComponent(job)}`).then((r) => r.json()),
      ]);
      const mastersById = {};
      for (const s of modes.segments || []) mastersById[s.seg_id] = s.master_exists;
      sceneRows = (poolData.segments || []).map((s) => ({
        ...s,
        master_exists: mastersById[s.seg_id],
      }));
      stitchBtn.style.display = poolData.pool_count ? "" : "none";
    } catch (_) {}
    paintScenes();
  }

  function paint(data) {
    if (data.generated) {
      const url = viewUrl(data.generated);
      if (videoEl.getAttribute("src") !== url) videoEl.src = url;
    }
    frameEl.style.display = data.frame ? "" : "none";
    if (data.frame) frameEl.src = viewUrl(data.frame) + "&t=" + Date.now();
    if (data.result) {
      resultEl.style.display = "";
      const url = viewUrl(data.result);
      if (resultEl.getAttribute("src") !== url) resultEl.src = url;
    }
    const running = data.progress && ["auto", "manual", "frame"].includes(data.progress.stage);
    const hasGen = !!data.generated;
    const hasFrame = !!data.frame;
    const hasChoice = !!(data.choice && data.choice.mode);
    const hasResult = !!data.result;
    const inPool = !!data.ready;
    paintPrep();
    autoBtn.disabled = running || !hasGen || !hasFrame;
    ownBtn.disabled = running || !hasGen || !hasFrame;
    fullBtn.disabled = running || !hasChoice;
    const nxt = nextOpenIndex();
    nextBtn.style.display = inPool && nxt >= 0 ? "" : "none";
    const failed = data.progress && data.progress.stage === "error" && !hasFrame;
    if (failed) {
      statusEl.textContent = data.progress.error || "Ошибка";
    } else if (data.progress && (data.progress.stage === "auto" || data.progress.stage === "manual") && data.progress.total) {
      statusEl.textContent = `Считаю маску: ${data.progress.done} / ${data.progress.total}`;
    } else if (inPool && nxt >= 0) {
      statusEl.textContent = `SEG${id} в пуле. Дальше SEG${sceneRows[nxt].seg_id}: тот же Тест.`;
    } else if (inPool) {
      statusEl.textContent = `SEG${id} в пуле. Все сцены можно склеить.`;
    } else if (hasFrame && !hasChoice) {
      statusEl.textContent = "Пауза. Авто — эта маска на всю сцену. Своя — файл с компьютера.";
    } else if (hasResult && hasChoice) {
      statusEl.textContent = "Маска выбрана. Сцена ещё считается.";
    } else if (hasGen && !hasFrame) {
      statusEl.textContent = "Ролик есть. Считаю маску на одном кадре…";
    } else {
      statusEl.textContent = data.message || "Тест этой сцены.";
    }
  }

  async function refresh(kind = "test") {
    const resp = await api.fetchApi(
      `/cvfs/matte_status?job=${encodeURIComponent(job)}&seg=${encodeURIComponent(index)}&kind=${encodeURIComponent(kind)}`
    );
    const data = await resp.json();
    if (!resp.ok || data.error) {
      statusEl.textContent = data.error || "Не удалось прочитать состояние";
      return data;
    }
    paint(data);
    return data;
  }

  async function pollFresh(beforeName, kind = "test") {
    for (let i = 0; i < 240; i++) {
      const resp = await api.fetchApi(
        `/cvfs/matte_status?job=${encodeURIComponent(job)}&seg=${encodeURIComponent(index)}&kind=${encodeURIComponent(kind)}`
      );
      const data = await resp.json();
      const latest = data.latest_generation;
      if (latest && latest.filename !== beforeName) return latest;
      statusEl.textContent = kind === "full" ? "Идёт финальный рендер…" : "Идёт прогон 2 сек…";
      await new Promise((r) => setTimeout(r, 3000));
    }
    throw new Error("Прогон не появился. Смотри очередь Comfy.");
  }

  function watchUntilDone(kind = "test") {
    timer = window.setInterval(async () => {
      const cur = await refresh(kind);
      const stage = cur.progress && cur.progress.stage;
      if (stage === "done" || stage === "error") {
        window.clearInterval(timer);
        setBusy(false);
        await refresh("test");
      }
    }, 2000);
  }

  async function switchSeg(pos) {
    if (!sceneRows.length) return;
    const seg = sceneRows[Math.max(0, Math.min(sceneRows.length - 1, pos))];
    if (!seg) return;
    index = Number(seg.seg_index);
    id = String(seg.seg_id || index).padStart(2, "0");
    videoEl.removeAttribute("src");
    videoEl.load?.();
    frameEl.removeAttribute("src");
    resultEl.style.display = "none";
    resultEl.removeAttribute("src");
    nextBtn.style.display = "none";
    paintScenes();
    await refresh("test");
  }

  nextBtn.onclick = () => {
    const pos = nextOpenIndex();
    if (pos >= 0) switchSeg(pos);
  };
  stitchBtn.onclick = async () => {
    stitchBtn.disabled = true;
    statusEl.textContent = "Склеиваю сцены из пула по порядку видео…";
    try {
      const resp = await api.fetchApi("/cvfs/pool_assemble", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_name: job }),
      });
      const data = await resp.json();
      if (!resp.ok || data.error) throw new Error(data.error || "склейка не вышла");
      const used = (data.used || []).map((s) => "SEG" + s).join(", ");
      statusEl.textContent = used ? `Финал готов: ${used}` : "В пуле пусто.";
    } catch (err) {
      statusEl.textContent = String(err.message || err);
    } finally {
      stitchBtn.disabled = false;
    }
  };

  close.onclick = () => {
    window.clearInterval(timer);
    overlay.remove();
  };
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) {
      window.clearInterval(timer);
      overlay.remove();
    }
  });

  runBtn.onclick = async () => {
    setBusy(true);
      statusEl.textContent = "Тест: 2 сек, потом маска кадра…";
    const before = await refresh("test");
    const beforeName = before && before.latest_generation ? before.latest_generation.filename : "";
    try {
      const resp = await api.fetchApi("/cvfs/run_all", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_name: job,
          pad_percent: pad ?? 20,
          seed: 210,
          only_missing_wan: false,
          queue_assemble_after: false,
          seg_indices: [index],
          test_seconds: 2,
          face_images_prep: "reactor",
        }),
      });
      const queued = await resp.json();
      if (!resp.ok || queued.error) throw new Error(queued.error || "не встало в очередь");
      const fresh = await pollFresh(beforeName, "test");
      statusEl.textContent = "Прогон готов. Закрепляю и считаю маску одного кадра…";
      const pin = await api.fetchApi("/cvfs/matte_pin", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_name: job,
          seg_index: index,
          seg_id: id,
          filename: fresh.filename,
          kind: "test",
        }),
      });
      const pinned = await pin.json();
      if (!pin.ok || pinned.error) throw new Error(pinned.error || "не закрепилось");
      const frame = await api.fetchApi("/cvfs/matte_frame", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_name: job, seg_index: index, kind: "test" }),
      });
      const shown = await frame.json();
      if (!frame.ok || shown.error) throw new Error(shown.error || "маска кадра не вышла");
      paint(shown);
      setBusy(false);
    } catch (err) {
      statusEl.textContent = String(err.message || err);
      setBusy(false);
    }
  };

  autoBtn.onclick = () => fullBtn.onclick();

  ownBtn.onclick = () => file.click();
  file.onchange = async () => {
    const picked = file.files && file.files[0];
    file.value = "";
    if (!picked) return;
    setBusy(true);
    statusEl.textContent = "Своя маска на тест…";
    const body = new FormData();
    body.append("job_name", job);
    body.append("seg_index", String(index));
    body.append("kind", "test");
    body.append("save_only", "1");
    body.append("mask", picked);
    const resp = await api.fetchApi("/cvfs/matte_manual", { method: "POST", body });
    const data = await resp.json();
    if (!resp.ok || data.error) {
      statusEl.textContent = data.error || "файл не принялся";
      setBusy(false);
      return;
    }
    statusEl.textContent = "Файл сохранён. Вся сцена в пул…";
    fullBtn.onclick();
  };

  fullBtn.onclick = async () => {
    setBusy(true);
    statusEl.textContent = "Финал: вся сцена в очередь…";
    const before = await refresh("full");
    const beforeName = before && before.latest_generation ? before.latest_generation.filename : "";
    try {
      const resp = await api.fetchApi("/cvfs/run_all", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_name: job,
          pad_percent: pad ?? 20,
          seed: 210,
          only_missing_wan: false,
          queue_assemble_after: false,
          seg_indices: [index],
          face_images_prep: "reactor",
        }),
      });
      const queued = await resp.json();
      if (!resp.ok || queued.error) throw new Error(queued.error || "не встало в очередь");
      const fresh = await pollFresh(beforeName, "full");
      statusEl.textContent = "Сцена готова. Закрепляю и накладываю выбранную маску…";
      const pin = await api.fetchApi("/cvfs/matte_pin", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_name: job,
          seg_index: index,
          seg_id: id,
          filename: fresh.filename,
          kind: "full",
        }),
      });
      const pinned = await pin.json();
      if (!pin.ok || pinned.error) throw new Error(pinned.error || "не закрепилось");
      const apply = await api.fetchApi("/cvfs/matte_apply_choice", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_name: job, seg_index: index, kind: "full" }),
      });
      const applied = await apply.json();
      if (!apply.ok || applied.error) throw new Error(applied.error || "маска на финале не пошла");
      watchUntilDone("full");
    } catch (err) {
      statusEl.textContent = String(err.message || err);
      setBusy(false);
    }
  };

  paintPrep();
  refresh("test").then(async (data) => {
    if (data && data.generated && !data.frame) {
      setBusy(true);
      statusEl.textContent = "Ролик уже есть. Считаю маску на одном кадре…";
      try {
        const frame = await api.fetchApi("/cvfs/matte_frame", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ job_name: job, seg_index: index, kind: "test" }),
        });
        const shown = await frame.json();
        if (shown.error) statusEl.textContent = shown.error;
        else paint(shown);
      } catch (err) {
        statusEl.textContent = String(err);
      } finally {
        setBusy(false);
      }
    }
  });
};
