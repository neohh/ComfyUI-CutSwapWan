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
| Тест P (polish) | **SwapPolish**: ReActor + light shape/hair, no Wan | `preview/XX_P_*.mp4` |

## Requirements

- ComfyUI + Video Helper Suite  
- WanVideoWrapper (Animate) — FP8 and/or **Q4 GGUF**  
- ReActor (for Test A / Af / P)  
- Driver + matte videos aligned frame-for-frame  

### 16GB VRAM (e.g. RTX 5080)

If `Wan2_2_Animate_14B_Q4_K_M.gguf` is in `models/diffusion_models/`, CutSwapWan prefers it over FP8.

- Block swap ~**36–40**, `use_non_blocking=false` (low swap + non_blocking hangs Sampling at 0%)  
- Default Q4 crop ~**512²** (safer smoke); raise later for sharper  
- Preview tag **`XX_AQ4_*.mp4`** = Test A on Q4  

FP8 (`Wan2_2-Animate-14B_fp8_scaled_…`) still works but needs heavier swap and is much slower on 16GB.

## Install

Clone or copy into `ComfyUI/custom_nodes/ComfyUI-CutVideoFaceSwapper` (folder name kept for existing Docker mounts). Restart ComfyUI.

## Version

See GitHub Releases. Current branded line: **CutSwapWan**.

### Changelog

**v1.1.0 — 16GB Q4 Animate + Test P** (vs **v1.0.0** FP8-only baseline)

- Auto-prefer **Animate Q4 GGUF** when present; `merge_loras=false`  
- 5080-safe VRAM guard for GGUF (high blockswap, non_blocking off)  
- **Test P (SwapPolish)**: local ReActor + shape/hair, no Wan (`preview/XX_P_*.mp4`)  
- Test A on Q4 writes `preview/XX_AQ4_*.mp4`  
