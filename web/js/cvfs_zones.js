import { app } from "../../../scripts/app.js";

/**
 * CVFSZoneSelect → mute [Z1]/[Z2]/[Z3] titled nodes so Queue runs one stage.
 * LiteGraph modes: 0=NORMAL, 2=NEVER (mute), 4=BYPASS
 */
const ZONE_TO_TAG = {
  "1_detect": "[Z1]",
  "2_masters_wan": "[Z2]",
  "3_assemble": "[Z3]",
};

// Title tags win. These types are muted even if a script renamed the node and dropped [Z2].
const TYPE_ZONE = {
  CVFSDetectSceneCuts: "1_detect",
  CVFSRunFromMastersFolder: "2_masters_wan",
  CVFSRunFromMastersInOrder: "2_masters_wan",
  CVFSSegPicker: "2_masters_wan",
  CVFSSaveMasterForSegment: "2_masters_wan",
  CVFSBuildWanWorkflowsFromMasters: "2_masters_wan",
  CVFSAssembleJob: "3_assemble",
};

function applyZoneMutes(zone) {
  const activeTag = ZONE_TO_TAG[zone];
  if (!activeTag || !app.graph?._nodes) return;
  for (const node of app.graph._nodes) {
    const type = node.type || node.comfyClass || "";
    if (type === "CVFSZoneSelect") continue;
    let title = node.title || node.properties?.["Node name for S&R"] || "";
    if (
      (type === "CVFSRunFromMastersFolder" || type === "CVFSRunFromMastersInOrder") &&
      !title.includes("[Z1]") &&
      !title.includes("[Z2]") &&
      !title.includes("[Z3]")
    ) {
      title = `[Z2] ${title || "Masters pack"}`;
      node.title = title;
    }
    const tags = ["[Z1]", "[Z2]", "[Z3]"].filter((t) => title.includes(t));
    let on = null;
    if (tags.length) on = tags.includes(activeTag);
    else if (TYPE_ZONE[type]) on = TYPE_ZONE[type] === zone;
    if (on === null) continue;
    node.mode = on ? 0 : 2;
  }
  app.graph?.setDirtyCanvas?.(true, true);
}

function applyCurrentZone() {
  const node = (app.graph?._nodes || []).find((n) => (n.type || n.comfyClass) === "CVFSZoneSelect");
  if (!node) return;
  applyZoneMutes(readZoneFromNode(node));
}

window.addEventListener("cvfs-apply-zones", () => applyCurrentZone());

function readZoneFromNode(node) {
  const w = node.widgets?.find((x) => x.name === "zone");
  return w?.value || "1_detect";
}

function hookZoneSelect(node) {
  const w = node.widgets?.find((x) => x.name === "zone");
  if (!w || w.__cvfs_zone_hooked) return;
  w.__cvfs_zone_hooked = true;
  const prev = w.callback;
  w.callback = function (v) {
    prev?.apply(this, arguments);
    applyZoneMutes(v);
  };
  // Apply once when node appears
  applyZoneMutes(w.value || "1_detect");
}

app.registerExtension({
  name: "cut_video_face_swapper.zones",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "CVFSZoneSelect") return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      if (!(this.title || "").startsWith("[CTRL]")) {
        this.title = "[CTRL] Zone — 1 detect / 2 work / 3 assemble";
      }
      hookZoneSelect(this);
      this.addWidget("button", "Apply zone mutes now", null, () => {
        applyZoneMutes(readZoneFromNode(this));
      });
      return r;
    };
  },
  async afterConfigureGraph() {
    for (const node of app.graph?._nodes || []) {
      if ((node.type || node.comfyClass) === "CVFSZoneSelect") {
        hookZoneSelect(node);
        applyZoneMutes(readZoneFromNode(node));
      }
    }
  },
});
