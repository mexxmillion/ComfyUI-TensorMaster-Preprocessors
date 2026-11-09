# Tensor Master nodes for ComfyUI

Reference editing, a compact multi-LoRA loader, and synchronized mode switches
for editable ComfyUI workflows. Runs inside ComfyUI; no Tensor Master or Comfy
Commander account is required. Uses the current ComfyUI V3 node API.

## Quick install

In **ComfyUI-Manager**, choose **Install via Git URL** and paste:

```text
https://github.com/mexxmillion/ComfyUI-TensorMaster-Preprocessors
```

Restart ComfyUI and hard-refresh the browser with **Ctrl+Shift+R**. Search for
`TM` in the node picker. This repository can be installed directly by URL;
it does not need a Comfy Registry listing.

For a manual install, run from your ComfyUI directory using its Python environment:

```sh
git clone https://github.com/mexxmillion/ComfyUI-TensorMaster-Preprocessors.git custom_nodes/ComfyUI-TensorMaster-Preprocessors
python -m pip install -r custom_nodes/ComfyUI-TensorMaster-Preprocessors/requirements.txt
```

Windows portable users: run the clone command from the `ComfyUI` folder, then
run this from the containing `ComfyUI_windows_portable` folder:

```powershell
.\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-TensorMaster-Preprocessors\requirements.txt
```

| Node | What it does |
| --- | --- |
| **TM · H3 Reference Manager** | Upload and edit image, video, and audio references, with a full Reference Studio editor. |
| **TM · Power LoRA** | Add any number of installed LoRAs; one strength slider controls model and CLIP together. |
| **TM · State Menu** | Create a named dropdown for your generation modes. |
| **TM · State Switch** | Grow matching inputs and evaluate only the selected branch; drive several switches from one menu. |
| **TM · Face Cover** | Cover detected faces with solid shapes using installed detectors. |

The reference editor, LoRA loader, and state switches need no preprocessing
model downloads. Isolation and depth processing need the optional weights below.
LoRA and generation checkpoints are supplied by the user. Video preparation
needs FFmpeg on PATH; media decoding uses ComfyUI's PyAV dependency.

**Example:** [TM H3 - Three Mode Studio](examples/TM%20H3%20-%20Three%20Mode%20Studio.json)
contains editable four-, six-, and eight-step recipes. Download its JSON and
drop it onto ComfyUI. See the example requirements at the end of this README.
The reference editor was visually checked with Nodes 2.0 in frontend 1.58.5;
the example's subgraph exposure and full generation remain unverified.

## H3 reference manager

Add **TM · H3 Reference Manager** from **Tensor Master / References**. This is a
separate, resizable fork of MiniMaxRefPack's manager; the original node remains
available. The prompt and compact reference tiles share Commander's H3 studio
layout. **Expand** opens a larger studio synced with the node. The gallery reflows
and scrolls inside the node.

- Upload images, videos and audio. Replacing an image keeps its socket, crop,
  depth, face-cover, and isolation settings. Crop uses the same relative area
  on the new image. **Reset edits** clears those settings when you want to start
  fresh. Replacing video or audio resets its source-specific edits.
- Drag image thumbnails to reorder them. Use their checkboxes, Ctrl/Cmd-click or
  Shift-click to select a group, then drag it together. Edits follow the images;
  Picture tags and image sockets follow the new order. Prompt text is not rewritten.
- **Process** opens a side-by-side original/result editor with a persistent action
  bar. Image crop and preparation controls share one page: drag on the original
  image to crop, or use the aspect presets and corner handles. Video/audio
  **Trim** provides in/out fields, playhead controls and range sliders. Images support
  text-guided isolation, subject selection, background compositing and depth.
- **Preview result** queues only that image's preparation and leaves the editor
  open for comparison. **Save reference** applies the settings; Cancel discards
  the draft. Normal execution applies the same preparation automatically.
- Plug text into the **prompt** input, or type into the node. Connected text takes
  precedence, including an empty string. **AI enhance prompt** applies to either
  source: off returns it unchanged; on runs the selected prompt writer.
- Connect **width**, **height** and **length_seconds** from your generation settings
  to give the AI writer the target frame, aspect ratio and time budget. They default
  to unspecified and do not change the downstream generation settings.
- **Resize references to target** is off by default. Enable it to resize images
  and video frames to the exact connected width/height, overriding the image edge
  cap. This stretches to fit without adding a crop. Images are resized before
  preprocessing and again after subject cropping if needed. Audio and reference
  duration stay unchanged. Image previews use the same connected dimensions.
- **Model & options** configures the local OpenAI-compatible endpoint or OpenRouter,
  model ID, reasoning, system prompt and image size cap. A local endpoint needs
  its own vision model ID. Local video is sampled as stills and audio is omitted.
  Enhancement sends the prompt and supported media to the configured provider.
  Prefer `OPENROUTER_API_KEY` over saving a key in a shared workflow.

Wire the matching image/video/audio outputs and **prompt** into the native MiniMax
H3 Reference to Video node. **manifest** lists the actual reference tags. Sources
and settings save with the workflow; processed thumbnails live in ComfyUI's temp
directory and can be regenerated. Preprocessing uses the pack's existing local
models; the manager does not download weights.

Reference loading, tagging and prompt writing are derived from Hearmeman24's MIT
MiniMaxRefPack. Its notice is retained in `licenses/MiniMaxRefPack-LICENSE`.

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

## Solid face cover

**TM · Face Cover (solid ellipse)** accepts an IMAGE batch and outputs the covered
images plus a binary cover mask. The default RetinaFace backend reuses the installed,
enabled **ComfyUI-AutoCropFaces** node and its local weights, running on CPU. OpenCV
frontal/profile detection remains selectable. Choose ellipse or circle, coverage
width/height and a hex fill color (default `#808080`). Coverage has a hard, opaque
edge. RGB/RGBA inputs retain their dimensions, dtype, device and alpha channel.
Increase **scale_factor** in the standalone node or **Scale factor** in the manager to
cover hair too; `1` preserves existing coverage, while `1.5`–`2` enlarges it.
No downloads or segmentation model are needed. Inspect results for missed faces.

**number_of_faces** limits covers per image (`0` = all remaining), starting at the
zero-based **start_index**. Choose confidence, left-to-right or largest-first order.
RetinaFace confidence and max faces per image control detection/selection. The
vertical **shift_factor** is centered at `0.5`; higher values move the cover upward.
Existing saved frontal-detector settings stay unchanged; select **retinaface** to upgrade them.

In the reference manager, open **Process → Prepare → Cover faces**. The same
controls, previews and saved settings are available there. Detection uses the RGB
reference; solid covers are applied after depth processing and before isolated
subject cropping, so the cover stays aligned with other edits.

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

For optional isolation, depth, and face-parsing weights, run from this node
pack's directory using ComfyUI's Python environment:

```sh
python sync_models.py --models-dir /path/to/ComfyUI/models
```

This explicitly downloads the pinned artifacts in `models.lock.json`. Read the
individual model licenses before use; the face-parsing weights have separate
non-commercial terms. Weights are not included in this repository.
Add `ComfyUI-TensorMaster-Preprocessors` to an existing secure launcher's node
allowlist. Keep all existing allowlist entries. Restart only when the worker is idle.

Weights/configs are pinned and checksum verified. Preprocessing is offline and
does not download models or execute remote model code. Optional prompt enhancement
sends the prompt and supported media to the selected endpoint only when enabled.
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

### Reduce over-detection

Start with `detected_faces_only = true`, `segmentation_confidence = 0.65`,
`detector_neighbors = 8`, and `grow = 4`. Higher segmentation confidence rejects
uncertain pixels in both full-frame and crop parsing. Crop-only mode skips
full-frame parsing, limiting segmentation to the contextual crops around detected
frontal faces; those crops include hair and some surrounding image. Higher
detector neighbors rejects weaker face boxes. Increase `min_face_size` to ignore
tiny detections (in input image pixels). Detector settings only affect face
crops; full-frame parsing still runs unless crop-only mode is enabled.

`grow` expands the mask and `feather` extends its blending edge, so reduce those
if detection is correct but the removed area is too large. Higher confidence
can miss real hair/face pixels. Enclosed mask holes are filled; inspect the final
preview. Defaults preserve existing workflows, and `extra_mask` is unaffected
by detection thresholds.

Run `python sync_models.py --models-dir /path/to/ComfyUI/models` to install the
pinned, SHA-256 verified face parsing files along with the existing models.
Inference is offline. Existing `models/face_eraser` files are reused after
verification. Model binaries and user reference images are not committed.

**Model terms:** [jonathandinu/face-parsing](https://huggingface.co/jonathandinu/face-parsing)
specifies non-commercial research and educational use. These terms apply
separately to this repository's MIT integration code. Existing isolation/depth
models retain their licenses. See `licenses/FaceParsing-NOTICE.md`.

### H3 Reference Manager in Nodes 2.0

The H3 Reference Manager uses native widget visibility for Nodes 2.0 and a
container-sized DOM panel. The editor sits beside the output socket column
instead of below it, keeping both the classic and Nodes 2.0 layouts compact.
Previously saved tall nodes can be resized shorter after refreshing.
**Open Reference Studio** opens its full editor in a
dialog; the existing **Expand** button opens the same editor. Workflow values,
reference sockets, and the node ID are unchanged. After updating the frontend
files, hard-refresh ComfyUI to load the new panel code.

### State menu and synchronized switches

Add **TM · State Menu** and use **Edit states…** to name your modes, for example
`Turbo 4-step`, `Turbo 8-step`, and `Flex 6-step`. Connect its `state` output to
several **TM · State Switch** nodes. Wire each mode's checkpoint MODEL, CLIP,
VAE, steps, CFG, or sampler value into the matching numbered input on its own
switch. Inputs grow as you connect them; each switch routes one type of value.
Only the selected branch is requested. Output nodes such as Save Image on other
branches are still independent execution targets.

State numbers stay fixed when names change or entries are removed. The menu
editor shows these numbers; keep them aligned across switches. An unconnected
selected input reports an error. The native Autogrow interface supports up to
100 numbered states in this version of ComfyUI.

Keep the menu inside your subgraph and promote its `selected` dropdown to the
outer controls. Edit branch nodes inside the subgraph to experiment with each
mode. The dropdown uses a native combo widget with dynamic options; actual
Nodes 2.0 and subgraph UI verification remains pending.

### Three-mode H3 studio example

Open `examples/TM H3 - Three Mode Studio.json`. The nine outer nodes contain
the mode dropdown, shared LoRA editor, resolution, duration, seed, three editable
subgraphs, and video output. No reference files or credentials are included.

| Mode | Model | Acceleration | Sampling |
| --- | --- | --- | --- |
| Turbo - 4 steps | Community Ref2VA INT8 | Ref2V Turbo 4-step v0.1, strength 1 | Euler, simple schedule, video/audio shifts 12/3 |
| Prism / Pika - 6 steps | Eros × Singularity × PikaFlex Prism2 INT8 HQ | Built into the checkpoint | LCM, fixed Pika sigmas, no extra shifts |
| Turbo - 8 steps | Singularity Ref2VA v2 INT8 | Ref2V Turbo 8-step v1.0, strength 1 | Euler, simple schedule, video/audio shifts 12/3 |

Open **Model & Recipe** to edit any checkpoint, acceleration LoRA, sampler,
or sigma schedule. One shared state selects MODEL, SAMPLER, and SIGMAS together.
Open **References & Prompt** to use the complete Reference Manager panel;
all nine image, three video, three video soundtrack, and three audio outputs
are wired internally to native H3 conditioning. Open **Sample & Decode** to
change memory options, sampling, or VAE decoding.

The mode dropdown stays outside the subgraphs in this example. The Reference
Manager's custom panel stays inside its workspace; this example does not
promote that panel to the outer node. Both remain editable without a combined
generator node. Defaults are portrait 0.4 MP, five seconds, seed 42, and the
six-step recipe. Shared LoRAs start empty. The flattened prompt passed backend
validation for all three modes; generation and Nodes 2.0 visual checks remain
pending. Required packs: this pack, KJNodes, VideoHelperSuite, and
comfyui-unload-model, plus the installed H3 models named in the graph.
