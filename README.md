# Tensor Master reference preprocessors

Offline reference processing with Grounding DINO Tiny + SAM 2.1 Tiny for
text-guided isolation, and Depth Anything V2 Small for relative depth.

`TMPrepareReference` performs the configured steps together and outputs a regular
RGB conditioning image, a full-size RGBA cutout and its foreground mask. Describe
`black shirt`, `jacket`, `person`, or another visible object. Choose the highest
confidence match, all matches, or a numbered subject ordered left to right.
White, gray and black backgrounds and padded subject cropping are supported.
With isolation and depth enabled, depth is estimated from the original image,
then masked with the selected subject before RGB compositing.

In Tensor Master, click **Process** on an image reference, choose its options,
and **Save settings**. **Generate** processes and renders in one queued graph;
no preview is required. An optional **Process & cache now** action prepares a
reusable asset without starting generation. Original inputs remain available.

## Reuse and caching

**Reuse cached results** is off by default. Enable it per reference to reuse results. Isolation masks and depth maps have
separate content/settings/model-version cache keys. Source bytes are checked on every run, including replacements with the same filename. Changing background color or
crop reuses the existing mask; changing isolation text does not recompute depth.
Turning reuse off forces inference. Settings persist in drafts and generation
recipes, independent of whether preprocessing has run yet.

The worker saves component images and the RGB result under
`output/reference-cache`. Tensor Master returns them through an authenticated,
lease-checked upload into its private `input/h3_mobile/preprocessed-cache` source
library. Subsequent cloud workers receive applicable cached inputs with their
task. Inference is skipped when those component images are available; caches are
not limited to a single GPU or disposable cloud instance.

## Installation

Install this directory under ComfyUI/custom_nodes, install requirements in the
ComfyUI environment, then run `python sync_models.py --models-dir /path/to/ComfyUI/models`.
Add `ComfyUI-TensorMaster-Preprocessors` to an existing secure launcher's node
allowlist. Keep all existing allowlist entries. Restart only when the worker is idle.

Weights/configs are pinned and checksum verified. Inference is offline and does
not download models, execute remote model code, or use API credentials.
Temporary inference models are released after each run. For isolation, no detections is an
explicit error, never a fallback to the entire image.

This first version supports still images. It produces segmentation masks, not
fine hair/transparency matting. Clothing selection is model dependent. Point/brush
correction and video are not included. Depth output is a grayscale relative-depth
reference, not dedicated ControlNet conditioning or metric distance.

The Tensor Master integration queues preprocessing alongside normal GPU work,
requires the `reference-preprocess-v1` worker capability, and preserves the
original, RGBA cutout, depth map and RGB conditioning image. The capability is
advertised only after all three checkpoints and the preparation node are present.

Run cache tests without ComfyUI or a GPU model load:

```sh
python -m unittest discover -s tests -v
```

Original integration code: MIT. Grounding DINO, SAM 2 and Depth Anything V2 Small use Apache-2.0;
their license texts are included in `licenses/`. Weights
are downloaded from their official Hugging Face repositories, not committed here.

## Face and hair removal

**Tensor Master · Face + Hair Eraser (Gauss Fill)** removes face parts, ears,
hair, glasses, earrings and optional hats/neck. Connect its `clean_image` output
to the generation reference input. `mask_preview` shows coverage; white in
`removal_mask` means removed. Start with grow 16 px, feather 12 px, sigma 48 px.
See `examples/Face Eraser - Example.json` for a ready-to-load workflow.

Gaussian surroundings mode builds a smooth fill exclusively from pixels outside
the removal footprint, preventing removed facial details from contributing to
that fill. This approximates a Nuke-style Gaussian fill; it does not reconstruct
the background. Neutral gray gives a uniform replacement. Increase grow to cover
wispy hair; add missed areas with `extra_mask`. `mask_only` uses only that input
and requires no model. Masks are resized and a single mask broadcasts to a batch.
Feathering extends outward; grow beyond all unwanted detail before feathering.

Full-frame parsing is supplemented by frontal face detection and crop parsing
for smaller heads. Profiles, tiny faces and occlusions can still be missed.
Inspect coverage; an empty mask leaves the image unchanged. The node reduces
identity conditioning but cannot guarantee that every identity cue is removed.
Segmentation runs on CPU, with one temporary model per execution and no VRAM use.
The `FaceHairEraser` node ID is preserved from the standalone version. Use this
pack instead of loading both copies.

Run `python sync_models.py --models-dir /path/to/ComfyUI/models` to install the
pinned, SHA-256 verified face parsing files along with the existing models.
Inference is offline. Existing `models/face_eraser` files are reused after
verification. Model binaries and user reference images are not committed.

**Model terms:** [jonathandinu/face-parsing](https://huggingface.co/jonathandinu/face-parsing)
specifies non-commercial research and educational use. These terms apply
separately to this repository's MIT integration code. Existing isolation/depth
models retain their licenses. See `licenses/FaceParsing-NOTICE.md`.
