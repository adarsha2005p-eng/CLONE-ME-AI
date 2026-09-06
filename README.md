# CLONE-ME-AI

[![License: MIT](https://img.shields.io/badge/License-MIT-indigo.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![ComfyUI API](https://img.shields.io/badge/Backend-ComfyUI%20API-emerald.svg)](https://github.com/comfyanonymous/ComfyUI)
[![FastAPI](https://img.shields.io/badge/Server-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![InsightFace](https://img.shields.io/badge/Identity-InsightFace%20(0.8757%20Sim)-orange.svg)](https://github.com/deepinsight/insightface)

**CLONE-ME-AI** is a full-stack, identity-preserving AI image-generation application. It solves the notorious "identity drift" problem in diffusion models by using an **Image-Editing / Inpainting-First Direct Face Preservation** pipeline coupled with **OpenPose ControlNet** guidance and **SDXL Lightning**.

Instead of hallucinating facial features from random latent noise, CLONE-ME-AI anchors the reference person's real facial pixels, applies a protective feathered mask, and conditions newly synthesized hair, skin, clothing, and background to create photorealistic full-body images in any outfit or pose.

---

## Key Features

- **Direct Face Preservation Architecture**: Achieves an exceptional **0.8757 InsightFace cosine similarity** (tested and verified against reference selfies).
- **Pose & Anatomy Control**: Integrates OpenPose XL (`thibaud_xl_openpose`) to eliminate distorted limbs and unnatural body proportions.
- **Lightning Fast**: Powered by `RealVisXL_V5_Lightning` (native FP8), generating $832 \times 1152$ full-body compositions in **12 steps (~25-35 seconds)** on a 6GB VRAM laptop GPU.
- **Full-Stack Web Interface**:
  - Drag-and-drop reference image upload with clipboard paste (`Ctrl+V`) support.
  - 1-click prompt presets (White crop top & jeans, smart blazer, summer dress, gym wear).
  - Real-time ComfyUI health check and generation progress milestones.
  - **Interactive Before / After Split Slider** for instant facial comparison.
  - Side-by-side viewer and one-click HD image download.
- **Production REST API**: FastAPI backend with health monitoring, automatic face alignment, canvas preparation, and ComfyUI job queueing.

---

## Architecture Diagram

```
+------------------------------------------------------------------------+
|                           User Web Browser                             |
|    Upload Reference Photo  |  Choose Outfit & Pose  |  Compare Result  |
+-----------------------------------+------------------------------------+
                                    | HTTP POST /api/generate
                                    v
+------------------------------------------------------------------------+
|                          FastAPI Backend                               |
|  1. InsightFace: Detect facial landmarks & extract head crop           |
|  2. Master Canvas: Place head at target skeleton coords (x=401, y=250) |
|  3. Mask Generator: 16px Gaussian feathered face protection mask       |
|  4. Workflow Dispatcher: Inject parameters & queue prompt in ComfyUI   |
+-----------------------------------+------------------------------------+
                                    | REST API (/prompt & /history)
                                    v
+------------------------------------------------------------------------+
|                           ComfyUI Engine                               |
|  - Checkpoint: RealVisXL V5 Lightning (FP8)                            |
|  - Inpaint Encoder: VAEEncodeForInpaint (Locks facial latents)         |
|  - Conditioning: IPAdapter FaceID Plus V2 + OpenPose ControlNet        |
|  - Sampler: KSampler (euler_ancestral, 12 steps, CFG 2.0, denoise 1.0)  |
+-----------------------------------+------------------------------------+
                                    | Output PNG + Cosine Similarity
                                    v
+------------------------------------------------------------------------+
|               Interactive Before / After Comparison UI                 |
+------------------------------------------------------------------------+
```

---

## Project Structure

```
CLONE-ME-AI/
├── backend/
│   ├── app.py                      # FastAPI server with face alignment & ComfyUI runner
│   ├── requirements.txt            # Python backend dependencies
│   ├── .env.example                # Configuration template
│   └── run_backend.bat             # 1-click Windows backend runner
├── frontend/
│   ├── index.html                  # Responsive modern web UI (Dark mode + Tailwind)
│   ├── style.css                   # Custom styles, split slider, animations
│   ├── app.js                      # Client-side reactivity, upload, slider, API polling
│   └── package.json                # Frontend package descriptor
├── workflows/
│   ├── direct_face_preservation_api.json           # ComfyUI API prompt workflow
│   ├── SDXL_DirectFacePreservation_Edit_MASTER.json# Master ComfyUI visual graph
│   ├── SDXL_MultiReference_Realistic_Identity_MASTER.json # Multi-reference workflow
│   ├── standing_3_4_openpose_target.png            # Default 3/4 pose skeleton template
│   └── README.md                                   # Workflow guide and node documentation
├── run_app.bat                     # Master 1-click launcher (ComfyUI + Backend + Web UI)
├── README.md                       # Comprehensive project documentation
└── .gitignore                      # Clean ignore rules for models, outputs, and binaries
```

---

## Required Model Weights & Setup

To run the generation pipeline, ensure the following models are downloaded and placed into your `ComfyUI/models/` folder:

| Model | Type | Target Folder | Source Link |
|---|---|---|---|
| **RealVisXL_V5_Lightning_Native_FP8** | Checkpoint | `ComfyUI/models/checkpoints/` | [HuggingFace - SG161222](https://huggingface.co/SG161222/RealVisXL_V5.0_Lightning) |
| **thibaud_xl_openpose** | ControlNet | `ComfyUI/models/controlnet/` | [HuggingFace - thibaud](https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0) |
| **CLIP-ViT-H-14-laion2B-s32B-b79K** | CLIP Vision | `ComfyUI/models/clip_vision/` | [HuggingFace - h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter) |
| **ip-adapter-faceid-plusv2_sdxl** | IP-Adapter | `ComfyUI/models/ipadapter/` | [HuggingFace - h94/IP-Adapter-FaceID](https://huggingface.co/h94/IP-Adapter-FaceID) |
| **ip-adapter-faceid-plusv2_sdxl_lora** | LoRA | `ComfyUI/models/loras/` | [HuggingFace - h94/IP-Adapter-FaceID](https://huggingface.co/h94/IP-Adapter-FaceID) |
| **buffalo_l** | InsightFace ONNX | `ComfyUI/models/insightface/models/buffalo_l/` | [DeepInsight GitHub](https://github.com/deepinsight/insightface) |

---

## Quick Start Guide

### Option 1: 1-Click Master Launch (Windows)
Double-click `run_app.bat` in the repository root.
It will:
1. Verify the Python environment.
2. Check if ComfyUI is responding on `http://127.0.0.1:8188` (and launch it if needed).
3. Start the FastAPI backend server on `http://127.0.0.1:8000`.
4. Automatically open `http://127.0.0.1:8000` in your default browser.

---

### Option 2: Manual Step-by-Step Setup

#### 1. Start ComfyUI Server
In your ComfyUI root directory, start the server:
```cmd
run_nvidia_gpu.bat
```
Verify that ComfyUI is running by navigating to `http://127.0.0.1:8188`.

#### 2. Install Backend Dependencies & Start Server
Using your Python interpreter (or portable `python_embeded\python.exe`):
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

#### 3. Open Web Interface
Open your browser and navigate to:
```
http://127.0.0.1:8000
```

---

## API Reference

### `GET /api/health`
Returns system status for both the FastAPI backend and the ComfyUI instance:
```json
{
  "status": "healthy",
  "backend": "online",
  "comfyui": "online",
  "comfyui_url": "http://127.0.0.1:8188"
}
```

### `POST /api/generate`
Accepts `multipart/form-data`:
- `image` (file): Uploaded reference photo.
- `prompt` (string, optional): Target clothing, pose, and background prompt.
- `negative_prompt` (string, optional): Anti-distortion conditioning.
- `seed` (integer, optional): Random seed (`-1` for random).
- `steps` (integer, optional): Sampler steps (default: `12`).
- `cfg` (float, optional): Guidance scale (default: `2.0`).

Returns:
```json
{
  "status": "success",
  "prompt_id": "39d848ed-1d74-4633-b17a-0a4d613311fd",
  "result_filename": "CLONE_ME_39d848ed_00001_.png",
  "result_url": "/api/result/CLONE_ME_39d848ed_00001_.png",
  "reference_url": "/api/reference/39d848ed_ref.png",
  "similarity_score": 0.8757,
  "seed": 42891741,
  "steps": 12,
  "cfg": 2.0,
  "duration_seconds": 28.4
}
```

### `GET /api/result/{filename}`
Streams the generated output image directly to the client.

### `GET /api/reference/{filename}`
Streams the uploaded reference image for side-by-side or split-slider comparison.

---

## Hardware & Performance Recommendations

- **Minimum GPU**: NVIDIA RTX 3050 (4GB / 6GB VRAM).
- **Tested Hardware**: NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM).
- **VRAM Optimization**:
  - Uses native FP8 model checkpoint (`RealVisXL_V5_Lightning_Native_FP8_00001_.safetensors`) to stay well within 5GB active VRAM.
  - Generates full $832 \times 1152$ resolution with zero out-of-memory errors.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
