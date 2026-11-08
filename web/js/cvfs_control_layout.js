import { app } from "../../../scripts/app.js";

/**
 * Pull control nodes left into a blue «УПРАВЛЕНИЕ» frame so they are not
 * lost among the Wan graph. Runs after the graph is loaded.
 */

const CTRL_TYPES = {
  CVFSZoneSelect: "[CTRL] Zone — 1 detect / 2 work / 3 assemble",
  CVFSSegPicker: "[CTRL] Одна сцена → тест / маска / пул",
  CVFSAssembleJob: "[CTRL] Assemble (legacy)",
  CVFSJobResults: "[CTRL] Results preview",
};

const GROUP_TITLE = "УПРАВЛЕНИЕ";
const NODE_COLOR = "#1a3a55";
const NODE_BG = "#243f5a";
const GROUP_COLOR = "#3d6ea8";

function controlNodes() {
  return (app.graph?._nodes || []).filter((n) => CTRL_TYPES[n.type || n.comfyClass]);
}

function findControlGroup() {
  const groups = app.graph?.groups || app.graph?._groups || [];
  return groups.find((g) => (g.title || "") === GROUP_TITLE) || null;
}

function layoutControls(force = false) {
  const nodes = controlNodes();
  if (!nodes.length) return false;

  // Stable order: zone → one scene → assemble → results
  const order = ["CVFSZoneSelect", "CVFSSegPicker", "CVFSAssembleJob", "CVFSJobResults"];
  nodes.sort((a, b) => order.indexOf(a.type || a.comfyClass) - order.indexOf(b.type || b.comfyClass));

  const already = nodes.every((n) => (n.title || "").startsWith("[CTRL]") && n.pos?.[0] < 0);
  if (already && findControlGroup() && !force) return true;

  const left = -560;
  let y = 70;
  const gap = 28;
  let maxW = 340;
  for (const n of nodes) {
    const type = n.type || n.comfyClass;
    n.title = CTRL_TYPES[type] || n.title;
    n.color = NODE_COLOR;
    n.bgcolor = NODE_BG;
    // Force a readable min width so the frame looks like a panel.
    if ((n.size?.[0] || 0) < 340) n.size = [340, n.size?.[1] || 120];
    n.pos = [left, y];
    maxW = Math.max(maxW, n.size?.[0] || 340);
    y += (n.size?.[1] || 120) + gap;
  }

  let group = findControlGroup();
  if (!group) {
    group = new LiteGraph.LGraphGroup();
    group.title = GROUP_TITLE;
    app.graph.add(group);
  }
  group.title = GROUP_TITLE;
  group.color = "#5aa0e0";
  group.font_size = 22;
  group.pos = [left - 40, 20];
  group.size = [maxW + 100, Math.max(420, y - 20 + 50)];
  // Keep the group pinned as a backdrop for these nodes.
  if (typeof group.recomputeInsideNodes === "function") {
    try {
      group.recomputeInsideNodes();
    } catch (_) {}
  }

  app.graph?.setDirtyCanvas?.(true, true);
  return true;
}

app.registerExtension({
  name: "cut_video_face_swapper.control_layout",
  async setup() {
    // After any graph load, pull controls aside.
    const prev = app.graph?.onNodeAdded;
    // Hook graph configure via afterConfigureGraph below.
  },
  async afterConfigureGraph() {
    // Delay one tick so sizes from widgets are final.
    setTimeout(() => layoutControls(false), 50);
    setTimeout(() => layoutControls(false), 400);
  },
});

window.cvfsLayoutControls = () => layoutControls(true);
