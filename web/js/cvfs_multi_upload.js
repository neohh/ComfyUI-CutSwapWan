import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

function viewUrl(name, subfolder) {
  const q = new URLSearchParams({ filename: name, type: "input", subfolder });
  return api.apiURL(`/view?${q.toString()}`);
}

function openReorder(node) {
  const folder = node.widgets?.find((w) => w.name === "masters_folder")?.value || "";
  const job = node.widgets?.find((w) => w.name === "job_name")?.value || "";
  const overlay = document.createElement("div");
  overlay.style.cssText =
    "position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:10000;display:flex;align-items:center;justify-content:center;";
  const panel = document.createElement("div");
  panel.style.cssText =
    "background:#1e1e1e;color:#ddd;padding:16px;border-radius:10px;max-width:720px;width:92%;max-height:86vh;overflow:auto;font:14px sans-serif;";
  panel.style.maxWidth = "860px";
  panel.innerHTML =
    "<b>Порядок мастеров и режим</b>" +
    "<div style='opacity:.75;font-size:12px;margin-top:4px'>Два режима. Лицо видно — обычный Wan. Лица нет — VACE. Ставится само, список справа можно поменять.</div>" +
    "<div class='cvfs-list' style='margin-top:10px'></div>";
  const actions = document.createElement("div");
  actions.style.cssText = "margin-top:12px;display:flex;gap:8px;";
  const save = document.createElement("button");
  save.textContent = "Сохранить порядок и режимы";
  const redetect = document.createElement("button");
  redetect.textContent = "Определить заново";
  const cancel = document.createElement("button");
  cancel.textContent = "Закрыть";
  actions.append(save, redetect, cancel);
  panel.appendChild(actions);
  overlay.appendChild(panel);
  document.body.appendChild(overlay);
  cancel.onclick = () => overlay.remove();
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) overlay.remove();
  });

  const listEl = panel.querySelector(".cvfs-list");
  let items = [];
  let modes = [];
  let dragFrom = -1;

  function modeAt(i) {
    return modes.find((m) => m.seg_index === i);
  }

  function render() {
    listEl.innerHTML = "";
    items.forEach((it, i) => {
      const row = document.createElement("div");
      row.draggable = true;
      row.style.cssText =
        "display:flex;gap:10px;align-items:center;padding:6px;margin-bottom:6px;background:#2a2a2a;border-radius:6px;cursor:grab;";
      row.innerHTML = `<img alt="" style="width:120px;height:68px;object-fit:cover;background:#000"/>
        <div style="flex:1"><b>SEG${String(i).padStart(2, "0")}</b><br><span style="opacity:.7;font-size:12px"></span></div>`;
      row.querySelector("img").src = viewUrl(it.name, it.subfolder);
      row.querySelector("span").textContent = it.name;
      const up = document.createElement("button");
      up.textContent = "↑";
      const down = document.createElement("button");
      down.textContent = "↓";
      up.onclick = () => {
        if (i === 0) return;
        [items[i - 1], items[i]] = [items[i], items[i - 1]];
        render();
      };
      down.onclick = () => {
        if (i === items.length - 1) return;
        [items[i + 1], items[i]] = [items[i], items[i + 1]];
        render();
      };
      row.addEventListener("dragstart", () => {
        dragFrom = i;
      });
      row.addEventListener("dragover", (e) => e.preventDefault());
      row.addEventListener("drop", (e) => {
        e.preventDefault();
        const from = dragFrom;
        const to = i;
        if (from < 0 || from === to) return;
        const [moved] = items.splice(from, 1);
        items.splice(to, 0, moved);
        dragFrom = -1;
        render();
      });
      const mode = modeAt(i);
      if (mode) {
        const sel = document.createElement("select");
        sel.style.cssText = "background:#111;color:#eee;padding:4px;";
        for (const [val, label] of [
          ["face", "Лицо — Wan"],
          ["vace", "Без лица — VACE"],
        ]) {
          const opt = document.createElement("option");
          opt.value = val;
          opt.textContent = label;
          if (mode.wan_mode === val) opt.selected = true;
          sel.appendChild(opt);
        }
        sel.onchange = () => {
          mode.wan_mode = sel.value;
        };
        row.appendChild(sel);
        const hint = document.createElement("div");
        hint.style.cssText = "font-size:11px;opacity:.7;min-width:72px;";
        const pct =
          mode.face_probe_ratio == null ? "" : ` ${Math.round(Number(mode.face_probe_ratio) * 100)}%`;
        hint.textContent = (mode.locked ? "вручную" : "авто") + pct;
        row.appendChild(hint);
      }
      row.append(up, down);
      listEl.appendChild(row);
    });
  }

  save.onclick = async () => {
    save.disabled = true;
    const resp = await api.fetchApi("/cvfs/masters_reorder", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder, job_name: job, names: items.map((it) => it.name) }),
    });
    const data = await resp.json();
    save.disabled = false;
    if (!resp.ok || data.error) {
      alert(data.error || "reorder failed");
      return;
    }
    if (job && modes.length) {
      const modeResp = await api.fetchApi("/cvfs/set_wan_modes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_name: job,
          modes: modes.map((m) => ({ seg_index: m.seg_index, wan_mode: m.wan_mode })),
        }),
      });
      const modeData = await modeResp.json();
      if (!modeResp.ok || modeData.error) {
        alert(modeData.error || "режим не записался");
        return;
      }
    }
    alert("Порядок и режимы записаны.\nЛицо идёт обычным Wan, без лица — VACE.\nДальше Apply SEG → Queue.");
    overlay.remove();
  };

  function load(force) {
    if (!job) {
      listEl.textContent = "На ноде пустой job_name — режим не к чему привязать.";
      return;
    }
    listEl.textContent = force ? "Смотрю, где есть лицо…" : "Загрузка…";
    const q = force ? "&force=1" : "";
    Promise.all([
      api.fetchApi(`/cvfs/masters_list?folder=${encodeURIComponent(folder)}`).then((r) => r.json()),
      api.fetchApi(`/cvfs/seg_modes?job=${encodeURIComponent(job)}${q}`).then((r) => r.json()),
    ])
      .then(([data, modeData]) => {
        modes = modeData.segments || [];
        if (modeData.error) {
          listEl.textContent = modeData.error;
          return;
        }
        if (data.error) {
          listEl.textContent = data.error;
          return;
        }
        items = (data.files || []).map((f) => ({ name: f.name, subfolder: data.folder }));
        if (!items.length) listEl.textContent = "В папке нет картинок. Сначала Upload.";
        else render();
      })
      .catch((e) => {
        listEl.textContent = String(e);
      });
  }

  redetect.onclick = () => load(true);
  load(false);
}

/**
 * Adds a "Upload masters pack" button to CVFSRunFromMastersFolder:
 * OS multi-select → uploads into input/cvfs_master_packs/<pack_id>/ → sets masters_folder.
 *
 * Order = FileList order from the OS dialog (as listed in the filename box),
 * NOT alphabetical and NOT visual left→right of thumbnails.
 * We prefix 000_, 001_, … so Python keeps that exact order.
 */
app.registerExtension({
  name: "cut_video_face_swapper.multi_upload",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "CVFSRunFromMastersFolder") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);

      const fileInput = document.createElement("input");
      fileInput.type = "file";
      fileInput.accept = "image/png,image/jpeg,image/webp,image/bmp";
      fileInput.multiple = true;
      fileInput.style.display = "none";
      document.body.appendChild(fileInput);

      fileInput.addEventListener("change", async () => {
        // Keep FileList order — do NOT sort by name (that was scrambling SEG mapping).
        const files = Array.from(fileInput.files || []);
        fileInput.value = "";
        if (!files.length) return;

        const previewLines = files
          .map((f, i) => `${i} → SEG${String(i).padStart(2, "0")}  ←  ${f.name}`)
          .join("\n");
        const ok = confirm(
          `Порядок мастеров (как в списке файлов диалога, сверху вниз / в поле имени):\n\n${previewLines}\n\n` +
            `SEG00 = первый в этом списке → первый кусок видео после Detect.\n` +
            `OK = загрузить так. Cancel = отмена (выбери заново кликами в нужном порядке).`
        );
        if (!ok) return;

        const packId = `pack_${Date.now()}`;
        const subfolder = `cvfs_master_packs/${packId}`;

        const zoned = (label) => {
          const prev = this.title || "";
          const tags = ["[Z1]", "[Z2]", "[Z3]"].filter((t) => prev.includes(t));
          return `${tags.length ? tags.join("") : "[Z2]"} ${label}`;
        };
        this.title = zoned(`Uploading 0/${files.length}…`);
        for (let i = 0; i < files.length; i++) {
          const f = files[i];
          const ext = (f.name.split(".").pop() || "png").toLowerCase();
          const safeName = `${String(i).padStart(3, "0")}_${f.name.replace(/[^\w.\-]+/g, "_")}`;
          const body = new FormData();
          body.append("image", new File([f], safeName, { type: f.type || `image/${ext}` }));
          body.append("overwrite", "true");
          body.append("type", "input");
          body.append("subfolder", subfolder);
          const resp = await api.fetchApi("/upload/image", { method: "POST", body });
          if (resp.status !== 200) {
            const t = await resp.text();
            alert(`Upload failed for ${f.name}: ${t}`);
            this.title = zoned("CVFS Run From Masters Folder");
            window.dispatchEvent(new Event("cvfs-apply-zones"));
            return;
          }
          this.title = zoned(`Uploading ${i + 1}/${files.length}…`);
        }

        const folderWidget = this.widgets?.find((w) => w.name === "masters_folder");
        if (folderWidget) {
          folderWidget.value = subfolder.replace(/\\/g, "/");
          this.setDirtyCanvas?.(true, true);
        }
        this.title = zoned(`CVFS Masters pack (${files.length})`);
        window.dispatchEvent(new Event("cvfs-apply-zones"));
        app.graph?.setDirtyCanvas?.(true, true);
        openReorder(this);
      });

      const uploadName = "Upload masters pack (many files)";
      const orderName = "Порядок и режимы";
      const uploadBtn = this.widgets?.find((w) => w.name === uploadName);
      if (uploadBtn) uploadBtn.callback = () => fileInput.click();
      else this.addWidget("button", uploadName, null, () => fileInput.click());

      const orderBtn = this.widgets?.find((w) => w.name === orderName || w.name === "Reorder masters");
      if (orderBtn) {
        orderBtn.name = orderName;
        orderBtn.label = orderName;
        orderBtn.callback = () => openReorder(this);
      } else {
        this.addWidget("button", orderName, null, () => openReorder(this));
      }

      return r;
    };
  },
  async afterConfigureGraph() {
    for (const node of app.graph?._nodes || []) {
      if ((node.type || node.comfyClass) !== "CVFSRunFromMastersFolder") continue;
      const seed = node.widgets?.find((w) => w.name === "seed");
      if (seed && (typeof seed.value !== "number" || Number.isNaN(seed.value))) seed.value = 210;
      const ctrl = node.widgets?.find((w) => w.name === "control_after_generate");
      if (ctrl && !["fixed", "increment", "decrement", "randomize"].includes(ctrl.value)) ctrl.value = "fixed";
      const url = node.widgets?.find((w) => w.name === "comfy_url");
      if (url && (typeof url.value !== "string" || !String(url.value).startsWith("http"))) {
        url.value = "http://127.0.0.1:8188";
      }
    }
  },
});
