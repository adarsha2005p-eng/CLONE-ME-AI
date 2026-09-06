# CLONE-ME-AI Workflows

This directory contains the production-grade ComfyUI workflows powering **CLONE-ME-AI**.

---

## Workflows Overview

### 1. `direct_face_preservation_api.json` (Active Production API Workflow)
- **Format**: ComfyUI API Prompt JSON format.
- **Engine**: Inpainting-First Direct Face Preservation + ControlNet OpenPose XL.
- **Engine**: Inpainting-First Direct Face Preservation + ControlNet OpenPose XL.
- **Identity Metric**: **~0.80 - 0.82** pure generation InsightFace cosine similarity (evaluated directly on the unified generated image without any pixel compositing).
- **How It Works**:
  1. The user's original reference photo is analyzed using InsightFace to extract accurate facial landmarks (pupils, nose tip, jawline).
  2. The reference image is warped with similarity scaling onto an $832 \times 1152$ canvas with edge replication, precisely positioning the nose at $(399, 172)$ to align with the OpenPose skeleton's head and neck $(399, 253)$.
  3. A protective inpainting mask covers the facial identity (forehead down through chin and jawline) with an 8px Gaussian blur feather.
  4. `VAEEncodeForInpaint` with `grow_mask_by: 0` locks the facial latents with high fidelity while allowing the body, clothing, and background to be fully synthesized.
  5. `ControlNetApplyAdvanced` guides the body pose using the OpenPose skeleton (`standing_3_4_openpose_target.png`).
  6. SDXL Lightning checkpoint runs at 12 steps, CFG 2.0, producing a photorealistic, unified single-face full-body output with zero duplicate face artifacts.

### 2. `SDXL_DirectFacePreservation_Edit_MASTER.json` (ComfyUI Visual Graph)
- **Format**: ComfyUI visual graph format.
- **Usage**: Drag and drop directly into the ComfyUI web UI (`http://127.0.0.1:8188`) to view and modify node connections visually.

### 3. `SDXL_MultiReference_Realistic_Identity_MASTER.json`
- **Format**: ComfyUI visual graph format for multi-reference face embedding fusion.
- **Usage**: Experimental/advanced multi-angle identity conditioning.

### 4. `standing_3_4_openpose_target.png`
- The standard $832 \times 1152$ OpenPose skeleton template for natural 3/4 standing poses.

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
