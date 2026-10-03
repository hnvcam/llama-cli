## ComfyUI Environment & Paths

ComfyUI is separate from this repo. Know these paths before answering model or
workflow questions.

- Install: `~/ComfyUI/ComfyUI` (ComfyUI source tree; `main.py` is here).
- Models base (from `~/ComfyUI/ComfyUI/extra_model_paths.yaml`, `base_path`):
  `/media/hnvcam/AI/ComfyUI-Models`. Laid out by category: `checkpoints/`,
  `text_encoders/`, `clip_vision/`, `diffusion_models/`, `loras/`, `vae/`,
  `upscale_models/`, `model_patches/`, `geometry_estimation/`,
  `background_removal/`, `configs/`, `controlnet/`, `embeddings/`,
  `latent_upscale_models/`, `vae_approx/`.
  `diffusion_models/Qwen3-ASR/Qwen3-ASR-1.7B/` holds the ASR model (referenced by
  workflow as a directory, not individual files).
- User workflows: `~/ComfyUI/ComfyUI/user/default/workflows/` — the saved
  workflows the user actually runs. Prefer these over `app/` or `script_examples/`.

### Scanning models / workflows (reusable recipes)

- Unused models: list real model files under the models base (skip `put_*`
  placeholders, `*.md`, `*.yaml`, `.gitignore`, `config.json`, and
  `*.metadata`/`*.index.json` cache files), then keep the ones whose basename is
  NOT referenced by any workflow JSON. Reference extraction:
  `grep -rhoE '"[^"]*\.(safetensors|pth|pt|onnx|gguf|bin)"' <workflows> | sed 's#.*/##' | sort -u`.
  Some workflows (e.g. ASR) point at a model *directory*, so a file can be in use
  even if its own basename isn't matched — check for a parent-path match.
- Workflow roles (as of 2026-09-27: the two FLUX.2 Klein 9B workflows and their
  models `flux-2-klein-9b-fp8.safetensors` / `qwen_3_8b_fp8mixed.safetensors`
  were removed; the shared `flux2-vae.safetensors` and `full_encoder_small_decoder.safetensors` were kept):
  - Text-to-image: `image_flux2_text_to_image.json` (FLUX.2 Dev).
  - Image edit: `image_edit_flux2.json`, `qwen-image-edit-2511.json`.
  - Image gen + restore: `native_supir_example1.json` (SUPIR).
  - Text-to-image then vectorize: `SVG Generation.json`.
  - Video: `video_minimax_h3_i2v.json`, `video_minimax_h3_r2v.json`.
  - Audio: `audio_ace_step1_5_xl_sft.json` (music), `omnivoice-tts_example_workflow.json`
    (TTS), `qwen3-asr.json` (speech-to-text).
  - Subtitle align: `SRT Align.json`. Raster→vector of an existing image:
    `image to svg.json`, `silhouette to svg.json`. 3D: `3d_pixal3d_trellis2_image_to_model.json`.
- Known-unused (as of 2026-09): `upscale_models/4x-AnimeSharp.pth` (64 MB) — not
  referenced by any workflow; `RealESRGAN_x4.pth` is the one that is used.
- When reporting "missing models" for a workflow, do NOT flag dtype/quant variants
  as missing: an fp4, int8, int8-tensorwise, or int8-convrot variant of the same
  base model is an acceptable substitute (e.g. `flux2-dev-int8-tensorwise` stands
  in for `flux2_dev_fp8mixed`, `mistral_3_small_flux2_fp4_mixed` for
  `mistral_3_small_flux2_bf16`). Only flag a genuinely absent model (different
  model family/version), not a looser quantization of one that is present.
