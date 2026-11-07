# CutSwapWan (CSW)

ComfyUI custom node pack: **cut a video into scenes → face-swap prep → Wan redraw inside a wipe → assemble**.

Short name: **CutSwapWan**. Internal node prefix stays `CVFS` for compatibility.

## What this pipeline does

1. **Detect** scene cuts from driver + FaceHair matte  
2. Pick a **ref frame** per scene and a **master** still (identity)  
3. Build a **head-tracked crop** and a **red wipe rect** from the yellow matte  
4. Optional **Test A**: ReActor (`inswapper_128`) fills `face_images`, then **Wan Animate** redraws the wipe  
5. **Assemble** crops back into the full plate → `final.mp4`

This is **not** a light “edit the swap” pass. Wan regenerates the wipe region; ReActor only conditions the face channel.

## Zones (UI)

| Zone | Role |
|------|------|
| Z1 | Detect scenes |
| Z2 | Masters / Wan per SEG |
| Z3 | Assemble |

## Quick tests (SEG picker)

| Button | Meaning | Output |
|--------|---------|--------|
| Тест 2 сек | Wan face, ~2s around ref | `preview/XX_*.mp4` |
| Тест A (swap mimic) | ReActor → Wan | `preview/XX_A_*.mp4` |
| Тест Af (fast) | Same + smaller window | `preview/XX_Af_*.mp4` |

## Requirements

- ComfyUI + Video Helper Suite  
- WanVideoWrapper (Animate)  
- ReActor (for Test A / Af)  
- Driver + matte videos aligned frame-for-frame  

## Install

Clone or copy into `ComfyUI/custom_nodes/ComfyUI-CutVideoFaceSwapper` (folder name kept for existing Docker mounts). Restart ComfyUI.

## Version

See GitHub Releases. Current branded line: **CutSwapWan**.
