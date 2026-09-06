# CLONE-ME-AI Workflows

This directory contains the production-grade ComfyUI workflows powering **CLONE-ME-AI**.

---

## Workflows Overview

### 1. `clone_me_production_api.json` / `direct_face_preservation_api.json` (Active Production API Workflow)
- **Format**: ComfyUI API Prompt JSON format.
- **Engine**: IPAdapter FaceID Plus V2 + ControlNet OpenPose XL with EmptyLatent synthesis.
- **Identity Metric**: **~0.70 - 0.76** pure generation InsightFace cosine similarity without pixel compositing.
- **How It Works**:
  1. The user's uploaded reference photo is passed directly to IPAdapter FaceID Plus V2 (`weight: 1.25`, `weight_faceidv2: 1.50`, LoRA strength `0.85`) with InsightFace `buffalo_l` and CLIP Vision (`CLIP-ViT-H-14-laion2B-s32B-b79K`).
  2. The full human body pose is guided by `ControlNetApplyAdvanced` (strength `0.75`) using `standing_3_4_openpose_target.png`.
  3. `EmptyLatentImage` ($832 \times 1152$) synthesizes the body, clothing, and background naturally from scratch, preventing duplicate face hallucinations and mask seam distortions.
  4. SDXL Lightning checkpoint (`RealVisXL_V5_Lightning_Native_FP8_00001_.safetensors`) samples via `dpmpp_sde` / `sgm_uniform` at 8 steps, CFG 1.8, producing realistic anatomy, single head placement, and exact clothing styling.

### 2. `FINAL_IDENTITY_POSE_WORKFLOW.json` (ComfyUI Visual Graph)
- **Format**: ComfyUI visual graph format.
- **Usage**: Drag and drop directly into the ComfyUI web UI (`http://127.0.0.1:8188`) to view and execute the production visual graph.

### 3. `standing_3_4_openpose_target.png`
- The verified $832 \times 1152$ OpenPose skeleton template for natural 3/4 standing poses.

---

## Required Model Files & Locations

Place the following model files into your ComfyUI models directory:

| Model Type | File Name | Destination Directory |
|------------|-----------|----------------------|
| **Checkpoint** | `RealVisXL_V5_Lightning_Native_FP8_00001_.safetensors` | `ComfyUI/models/checkpoints/` |
| **ControlNet** | `thibaud_xl_openpose.safetensors` | `ComfyUI/models/controlnet/` |
| **CLIP Vision** | `CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors` | `ComfyUI/models/clip_vision/` |
| **IP-Adapter** | `ip-adapter-faceid-plusv2_sdxl.bin` | `ComfyUI/models/ipadapter/` |
| **FaceID LoRA** | `ip-adapter-faceid-plusv2_sdxl_lora.safetensors` | `ComfyUI/models/loras/` |
| **InsightFace** | `buffalo_l` (`1k3d68.onnx`, `2d106det.onnx`, `genderage.onnx`, `glintr100.onnx`, `scrfd_10g_bnkps.onnx`) | `ComfyUI/models/insightface/models/buffalo_l/` |

---

## Running Standalone via ComfyUI
To run these workflows without the full-stack web app:
1. Start ComfyUI (`run_nvidia_gpu.bat`).
2. Open `http://127.0.0.1:8188` in your browser.
3. Drag and drop `workflows/SDXL_DirectFacePreservation_Edit_MASTER.json` onto the ComfyUI canvas.
4. Set the input image path and click **Queue Prompt**.
