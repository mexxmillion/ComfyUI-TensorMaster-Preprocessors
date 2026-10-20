# Tensor Master reference preprocessors

Text-guided image isolation using Grounding DINO Tiny and SAM 2.1 Tiny through
Transformers. Describe `shirt`, `jacket`, `person`, or another visible object.
Outputs an RGBA cutout and a foreground mask (1 = keep). All matches, highest
confidence, or numbered subjects ordered left to right can be selected.

Install this directory under ComfyUI/custom_nodes, install requirements in the
ComfyUI environment, then run `python sync_models.py --models-dir /path/to/ComfyUI/models`.
Add `ComfyUI-TensorMaster-Preprocessors` to an existing secure launcher's node
allowlist. Keep all existing allowlist entries. Restart only when the worker is idle.

Weights/configs are pinned and checksum verified. Inference is offline and does
not download models, execute remote model code, or use API credentials.
Temporary inference models are released after each run. No detections is an
explicit error, never a fallback to the entire image.

This first version supports still images. It produces segmentation masks, not
fine hair/transparency matting. Clothing selection is model dependent; inspect
the cutout before applying it. Point/brush correction and video are not included.

The Tensor Master integration queues preprocessing alongside normal GPU work,
requires the `reference-isolate-v1` worker capability, and preserves the original,
RGBA cutout, mask, and neutral-gray RGB conditioning image.

Original integration code: MIT. Upstream Grounding DINO and SAM 2 use Apache-2.0;
their license texts are included in `licenses/`. Weights
are downloaded from their official Hugging Face repositories, not committed here.
