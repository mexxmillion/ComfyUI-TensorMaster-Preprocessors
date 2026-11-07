"""Reference media loading adapted from MiniMaxRefPack (MIT)."""

from __future__ import annotations

import math
import mimetypes

from . import logs


def resample_indices(n_src: int, src_fps: float, target_fps: int = 24) -> list[int]:
    """Which source frame each output frame reads from, resampling to `target_fps`.

    Preserves the clip's real-world duration: output frame count is
    round(n_src * target_fps / src_fps), not n_src. Output frame i reads source frame
    round(i * src_fps / target_fps), clamped to [0, n_src-1]. Core never touches
    framerate (CU/comfy_extras/nodes_minimax_h3.py:246-252 only resizes/trims by frame
    count) - this resample is ours alone.
    """
    if n_src <= 0:
        return []
    if src_fps <= 0:
        raise ValueError(f"source fps must be positive, got {src_fps!r}")
    n_out = max(1, round(n_src * target_fps / src_fps))
    return [min(max(round(i * src_fps / target_fps), 0), n_src - 1) for i in range(n_out)]


def _crop_box(crop, width: int, height: int) -> tuple[int, int, int, int]:
    """Fraction rect [x, y, w, h] -> integer (left, top, right, bottom) pixel box.

    THE one fraction->pixel rule - the loaders and the thumbnails all come through
    here, so a tile's thumb and the emitted tensor can never disagree. Each edge is
    floor(fraction * dimension + 0.5): half-up, not Python's round(), whose
    round-half-to-even would move an edge by a pixel depending on parity. Then clamp
    to the frame so a rect that rounds past an edge still keeps at least one pixel
    on each axis - a zero-width crop can never come out of here.
    """
    x, y, w, h = crop
    left = min(max(int(math.floor(x * width + 0.5)), 0), width - 1)
    top = min(max(int(math.floor(y * height + 0.5)), 0), height - 1)
    right = min(max(int(math.floor((x + w) * width + 0.5)), left + 1), width)
    bottom = min(max(int(math.floor((y + h) * height + 0.5)), top + 1), height)
    return left, top, right, bottom


def _slice_audio(audio: dict, trim) -> dict:
    """Slice a {"waveform","sample_rate"} dict to [start, end) seconds. Returns a NEW
    dict - callers may still hold the unsliced original."""
    sr = audio["sample_rate"]
    start, end = trim
    return {"waveform": audio["waveform"][..., int(start * sr):int(end * sr)].clone(), "sample_rate": sr}




def _video_from_file_cls():
    """Indirection point so tests can substitute a fake without comfy_api installed."""
    from comfy_api.latest._input_impl.video_types import VideoFromFile

    return VideoFromFile


def _audio_load_fn():
    """Indirection point so tests can substitute a fake without comfy_extras installed."""
    from comfy_extras.nodes_audio import load as _load

    return _load


def load_image(path: str, crop=None, max_edge: int = 0):
    """[1, H, W, 3] float32 in 0..1. Plain PIL decode - references are single stills
    (not animated), so we skip the ImageSequence handling CU/nodes.py:1734 LoadImage
    needs for animated webp.

    `crop` is a [x, y, w, h] fraction rect (refs.Reference.crop), applied AFTER the
    EXIF transpose so the fractions refer to the image as the editor showed it.

    `max_edge` caps the LONG edge (0 = off), applied AFTER the crop so it measures what
    is actually emitted. Core sizes reference images off the SHORT edge - `scale =
    min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))` at CU/comfy_extras/nodes_minimax_h3.py:301
    with REF_IMAGE_SHORT_EDGE = 2048 (:29) - so at ref_image_size="max" a wide sheet
    reaches the VAE enormous: 5000x2550 lands at 4000x2048, ~32k reference tokens that
    ride every sampling step. thumbnail() shrinks in place and never enlarges, so a
    reference already under the cap is emitted untouched.
    """
    import os

    import numpy as np
    import torch
    from PIL import Image, ImageOps

    with logs.timed("load_image", file=os.path.basename(path), crop=crop,
                    cap=max_edge or None) as fields:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        fields["src"] = f"{img.width}x{img.height}"
        if crop is not None:
            img = img.crop(_crop_box(crop, img.width, img.height))
        if max_edge:
            img.thumbnail((max_edge, max_edge), Image.LANCZOS)
        fields["out"] = f"{img.width}x{img.height}"
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr)[None, ...]


def load_audio(path: str, trim=None) -> dict:
    """{"waveform": [1, C, L], "sample_rate": int} - CU/comfy_extras/nodes_audio.py:333
    returns (waveform, sample_rate); the AUDIO socket shape wraps it with a batch dim
    (:380-381). `trim` = [start, end] seconds, sliced exactly the way a video's
    soundtrack is (_slice_audio)."""
    import os

    with logs.timed("load_audio", file=os.path.basename(path), trim=trim) as fields:
        waveform, sample_rate = _audio_load_fn()(path)
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        if trim is not None:
            audio = _slice_audio(audio, trim)
        fields["sample_rate"] = sample_rate
        return audio


def load_video(path: str, target_fps: int = 24, crop=None, trim=None):
    """(frames [N,H,W,3] resampled to target_fps, audio dict or None).

    CU/comfy_api/latest/_input_impl/video_types.py:118 VideoFromFile.get_components().
    `.audio` is already built as {"waveform","sample_rate"} (video_types.py:445-448),
    the same shape core AUDIO sockets use, so it's passed through unchanged.

    `trim` = [start, end] seconds. Cut on SOURCE frames first - a source frame is kept
    when its timestamp i/src_fps lands in [start, end) - then the usual resample runs
    on that span, so the output preserves the span's real duration. The soundtrack is
    sliced to the SAME window or it would drift out of sync with its frames.

    `crop` = [x, y, w, h] fractions, applied to the resampled frames (_crop_box).

    MiniMax needs >=5 frames per reference video (CU/comfy_extras/nodes_minimax_h3.py:250)
    - the check fires on the RESULT: a trim too short to survive it raises, naming the
    file and the requested window, never silently clamping or padding.
    """
    import os

    with logs.timed("load_video", file=os.path.basename(path), crop=crop, trim=trim) as fields:
        return _decode_video(path, target_fps, crop, trim, fields)


def _decode_video(path, target_fps, crop, trim, fields):
    """The body of load_video. Split out only so the timing/logging wrapper above stays
    a plain `with` block instead of wrapping 30 lines."""
    components = _video_from_file_cls()(path).get_components()
    frames = components.images
    n_src = frames.shape[0]
    src_fps = float(components.frame_rate)
    audio = components.audio
    fields["src_frames"] = n_src
    fields["fps"] = src_fps

    first = 0
    n_sel = n_src
    if trim is not None:
        start, end = trim


        first = max(0, math.ceil(start * src_fps - 1e-9))
        n_sel = max(0, min(n_src, math.ceil(end * src_fps - 1e-9)) - first)

    indices = resample_indices(n_sel, src_fps, target_fps)
    if len(indices) < 5:
        duration = (n_src / src_fps) if src_fps else 0.0
        window = f" trimmed to {trim[0]:.2f}-{trim[1]:.2f}s" if trim is not None else ""
        raise ValueError(
            f"reference video {path!r}{window} has only {len(indices)} frame(s) at {target_fps}fps "
            f"(source: {n_src} frames, {duration:.2f}s) - MiniMax H3 needs at least 5"
        )
    out = frames[[first + i for i in indices]]
    if crop is not None:
        left, top, right, bottom = _crop_box(crop, out.shape[2], out.shape[1])
        out = out[:, top:bottom, left:right, :]
    if trim is not None and audio is not None:
        audio = _slice_audio(audio, trim)
    fields["frames"] = len(indices)
    fields["audio"] = audio is not None
    return out, audio


VLM_VIDEO_MIMES = {
    ".mp4": "video/mp4",
    ".mov": "video/mov",
    ".webm": "video/webm",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
}


def video_clip_bytes(path: str, crop=None, trim=None) -> tuple[bytes, str]:
    """(bytes, mime) of the WHOLE clip, for the VLM's `video_url` part.

    The VLM reads video natively - one 10s 1080p clip cost 660 video tokens plus 250
    audio tokens on google/gemini-3-flash-preview - so it gets the clip, not stills.

    Untouched references take the fast path: the file's own bytes, nothing decoded,
    soundtrack included. A crop or a trim means the file no longer matches what the
    node's sockets emit, so that window is re-encoded (video only - its soundtrack is
    still sent as its own labelled part, exactly as before).
    """
    import os

    with logs.timed("video_bytes", file=os.path.basename(path), crop=crop, trim=trim) as fields:
        mime = VLM_VIDEO_MIMES.get(os.path.splitext(path)[1].lower())
        if crop is None and trim is None and mime is not None:
            with open(path, "rb") as f:
                data = f.read()
            fields["mode"] = "file"
            fields["bytes"] = len(data)
            return data, mime
        data = _transcode_window(path, crop, trim)
        fields["mode"] = "re-encoded"
        fields["bytes"] = len(data)
        return data, "video/mp4"


def _transcode_window(path: str, crop, trim) -> bytes:
    """The cropped/trimmed window as an in-memory mp4, video only.

    Decodes only the window: `container.seek` lands on the keyframe at or before the
    in-point (the same seek-then-skip-to-pts core uses,
    CU/comfy_api/latest/_input_impl/video_types.py:316-325) and decoding stops at the
    out-point, so a 4s window out of a 10 minute file costs 4s of decoding.
    """
    import io

    import av
    import numpy as np

    out_buf = io.BytesIO()
    start, end = (trim if trim is not None else (0.0, None))

    with av.open(path) as src:
        stream = src.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate or 24
        if start:
            src.seek(int(start / stream.time_base), stream=stream)

        out = av.open(out_buf, "w", format="mp4")
        enc = None
        try:
            for frame in src.decode(stream):
                t = float(frame.pts * stream.time_base) if frame.pts is not None else 0.0
                if t < start - 1e-9:
                    continue
                if end is not None and t >= end - 1e-9:
                    break
                arr = frame.to_ndarray(format="rgb24")
                if crop is not None:
                    left, top, right, bottom = _crop_box(crop, arr.shape[1], arr.shape[0])
                    arr = arr[top:bottom, left:right]
                if enc is None:
                    enc = out.add_stream("libx264", rate=rate)

                    enc.width = arr.shape[1] - (arr.shape[1] % 2)
                    enc.height = arr.shape[0] - (arr.shape[0] % 2)
                    enc.pix_fmt = "yuv420p"
                arr = np.ascontiguousarray(arr[: enc.height, : enc.width])
                for packet in enc.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                    out.mux(packet)
            if enc is not None:
                for packet in enc.encode():
                    out.mux(packet)
        finally:
            out.close()

    return out_buf.getvalue()
