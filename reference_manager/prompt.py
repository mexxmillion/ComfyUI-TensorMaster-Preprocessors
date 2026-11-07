"""Prompt enhancement adapted from MiniMaxRefPack (MIT)."""

from __future__ import annotations

import base64
import io
import math
import os
import time
import wave

import numpy as np
import requests
from PIL import Image

from . import endpoint as _endpoint
from . import logs, media


_DEFAULT_ENDPOINT = _endpoint.resolve("openrouter")


_DEGRADED_FRAMES = 6

DEFAULT_MODEL = "google/gemini-3-flash-preview"


_AUDIO_FORMATS = {".wav": "wav", ".mp3": "mp3"}


DEFAULT_REASONING_EFFORT = "medium"
REASONING_EFFORTS = ("none", "low", "medium", "high")

_SYSTEM_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "system_prompt.md")
_REPLACEMENT_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "system_prompt_replacement.md")


_PROMPT_PATHS = {"standard": _SYSTEM_PROMPT_PATH, "replacement": _REPLACEMENT_PROMPT_PATH}


CLASSIFIER_MODEL = "google/gemini-2.5-flash-lite"


_CLASSIFY_SYSTEM = (
    "Classify the user's video-generation request. Answer with exactly one word.\n"
    "REPLACEMENT - they want a specific object, product or character in a reference VIDEO "
    "swapped for the thing shown in a reference IMAGE, with everything else in that video "
    "left exactly as it is.\n"
    "STANDARD - anything else: a new scene, a continuation of a clip, a described shot, or "
    "a scene built from reference imagery.\n"
    "e.g. putting the character from the image into the video is REPLACEMENT; "
    "taking the motion from the video to animate a character is STANDARD.\n"
    "Answer REPLACEMENT or STANDARD."
)


class PromptError(RuntimeError):
    """The OpenRouter call could not produce a prompt."""


def _resolve_api_key(api_key: str) -> str:
    """UI box -> OPENROUTER_API_KEY -> LLM_KEY. Mirrors ComfyUI-Openrouter_node/node.py:42-67."""
    if api_key and api_key.strip():
        return api_key.strip()
    for env_name in ("OPENROUTER_API_KEY", "LLM_KEY"):
        env_key = os.environ.get(env_name)
        if env_key and env_key.strip():
            return env_key.strip()
    raise PromptError(
        "no OpenRouter API key: pass one in the node, or set OPENROUTER_API_KEY or LLM_KEY"
    )


def _key_for(ep, api_key: str) -> str:
    """The credential this endpoint may use, which is not always the one lying around.

    OpenRouter keeps today's behaviour exactly: node box, then OPENROUTER_API_KEY, then
    LLM_KEY, then raise.

    Any other endpoint gets ONLY what was typed into the node, never the environment.
    That asymmetry is the point. `api_base` is a free text field, so inheriting an
    ambient OPENROUTER_API_KEY would mean posting the user's paid credential to whatever
    host happens to be in that box - a typo, a copied workflow, or a hostile example
    graph. An empty string here is normal: local servers ignore the header entirely.
    """
    if ep.requires_key:
        return _resolve_api_key(api_key)
    return (api_key or "").strip()


def _auth_header(key: str) -> dict:
    """No key means no header at all, rather than `Bearer ` with nothing after it.

    llama.cpp and vLLM reject a malformed Authorization header even when they are not
    checking credentials, so an empty Bearer is worse than silence.
    """
    return {"Authorization": f"Bearer {key}"} if key else {}


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


VLM_IMAGE_LONG_EDGE = 1536
VLM_JPEG_QUALITY = 90


def _tensor_to_jpeg_b64(tensor) -> str:
    """A [H,W,3] or [1,H,W,3] float tensor in 0..1 -> base64 JPEG bytes, long edge capped
    at VLM_IMAGE_LONG_EDGE. Downscale only - a small reference is sent as it is."""
    arr = _to_numpy(tensor).astype(np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")

    img.thumbnail((VLM_IMAGE_LONG_EDGE, VLM_IMAGE_LONG_EDGE), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=VLM_JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _waveform_to_wav_b64(waveform, sample_rate: int) -> str:
    """A [1,C,T] or [C,T] float waveform in -1..1 -> base64 PCM16 WAV bytes."""
    arr = _to_numpy(waveform).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr[None, :]
    channels = arr.shape[0]
    pcm = np.clip(arr, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    interleaved = pcm.T.reshape(-1).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(interleaved)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _video_part(data: bytes, mime: str) -> dict:
    """OpenRouter's video content part. Verified live against
    google/gemini-3-flash-preview: a 10.12s 1080p clip inlined this way came back 200 in
    4.5s and billed 660 video tokens + 250 audio tokens - the model reads the motion, the
    cut and the dialogue, none of which six stills carry."""
    return {
        "type": "video_url",
        "video_url": {"url": f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"},
    }


def _image_part(jpeg_b64: str) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64}"}}


def _evenly_spaced(n: int, count: int) -> list[int]:
    """`count` indices spread across n frames, first and last always included.

    Separate from media.resample_indices, which preserves duration at a target fps. This
    one throws duration away on purpose: it picks a fixed number of stills to stand in
    for a clip, so what matters is covering the whole span rather than the rate.
    """
    if n <= 0:
        return []
    if n <= count:
        return list(range(n))
    if count == 1:
        return [0]
    return [round(i * (n - 1) / (count - 1)) for i in range(count)]


def _sampled_frame_b64s(path: str, crop=None, trim=None, count: int = _DEGRADED_FRAMES):
    """A clip reduced to stills, for an endpoint that cannot take video.

    Goes through the same loader the sockets use, so the frames carry the reference's
    crop and trim rather than the untouched original. The soundtrack load_video returns
    is dropped on the floor here by design: this path exists precisely because the
    endpoint takes neither video nor audio.
    """
    frames, _audio = media.load_video(path, target_fps=24, crop=crop, trim=trim)
    n = len(frames)
    picked = _evenly_spaced(n, count)
    logs.log("degraded_video", file=os.path.basename(path), src_frames=n, sent=len(picked))
    return [_tensor_to_jpeg_b64(frames[i]) for i in picked]


def _text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def _read_system_prompt(mode: str = "standard") -> str:
    path = _PROMPT_PATHS.get(mode, _SYSTEM_PROMPT_PATH)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def classify_mode(
    *, direction: str, references, api_key: str, model: str = "", ep=None,
    disable_local_thinking: bool = False,
) -> str:
    """'replacement' or 'standard' for a direction. Never raises.

    Replacement is only reachable with at least one video AND one image to swap between,
    so the cheap structural check runs first and skips the call entirely when the set
    can't support it. Any failure - no key, network, rate limit, a model that answers
    something else - falls back to 'standard', which is the right register for almost
    every job and never produces a catastrophically wrong format.

    This is the node's SECOND egress path and it used to hardcode OpenRouter's URL and an
    OpenRouter-only classifier model, while job_type defaults to "auto" (nodes.py). A run
    the user believed was local therefore still posted their direction text and their key
    to openrouter.ai - and because every failure here is swallowed, it did so silently.
    The endpoint now comes from the caller, and on a non-OpenRouter endpoint the writer's
    own model does the classifying, because a Gemini model id means nothing to a server
    holding one local model.
    """
    ep = ep or _DEFAULT_ENDPOINT
    counts = references.counts()
    if not counts.get("video") or not counts.get("image"):
        return "standard"
    if not (direction and direction.strip()):
        return "standard"


    classifier = model or (CLASSIFIER_MODEL if ep.is_openrouter else "")
    if not classifier:
        return "standard"
    try:
        key = _key_for(ep, api_key)
        payload = {
            "model": classifier,
            "max_tokens": 6,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": _CLASSIFY_SYSTEM},
                {"role": "user", "content": direction},
            ],
        }
        if disable_local_thinking and not ep.is_openrouter:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        resp = requests.post(
            ep.chat_url,
            headers=_auth_header(key),
            json=payload,
            timeout=ep.classify_timeout,
        )
        if resp.status_code != 200:
            return "standard"
        answer = (resp.json()["choices"][0]["message"].get("content") or "").strip().upper()
    except Exception:
        return "standard"
    return "replacement" if "REPLACEMENT" in answer else "standard"


def _file_audio_part(path: str, tag: str, filename: str) -> dict:
    """A standalone audio reference: inline if OpenRouter takes the format, else a text
    line naming the tag, filename and duration. Never crashes on audio."""
    ext = os.path.splitext(filename)[1].lower()
    fmt = _AUDIO_FORMATS.get(ext)
    if fmt is None:
        return _text_part(f"{tag}: audio reference {filename} (format {ext or 'unknown'} not inlined)")
    try:
        with open(path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("ascii")
        return {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}}
    except Exception:
        return _text_part(f"{tag}: audio reference {filename} (could not be read)")


def _video_audio_part(audio: dict, audio_tag: str, filename: str) -> dict:
    """A video's soundtrack, as decoded PCM by media.load_video. Never crashes on audio."""
    try:
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        b64 = _waveform_to_wav_b64(waveform, sample_rate)
        return {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}
    except Exception:
        return _text_part(f"{audio_tag}: soundtrack of {filename} (could not be encoded)")


def _target_format_lines(width: int, height: int, length_seconds: float) -> list[str]:
    """The TARGET FORMAT block's lines. 0/absent means unspecified: a line whose value
    is unknown is omitted rather than sent as a zero, and the frame line needs both
    dimensions (half a frame spec pins no aspect ratio)."""
    lines: list[str] = []
    if width and height:
        g = math.gcd(int(width), int(height))
        lines.append(f"frame: {int(width)} x {int(height)} (aspect ratio {int(width) // g}:{int(height) // g})")
    if length_seconds:
        lines.append(f"duration: {float(length_seconds):.3f} seconds")
    return lines


def _build_content(
    references, input_dir: str, direction: str,
    *, width: int = 0, height: int = 0, length_seconds: float = 0.0, accepts=None,
) -> list[dict]:
    """`references` is a refs.ReferenceSet; untyped here to avoid importing refs.py
    for a type hint alone (nothing else in this module needs it).

    Layout: one manifest text part, then every media part preceded by its own
    `<kind>_reference <Tag>` label line. The manifest alone is not enough - a model
    handed nine bare images in a row has to count positions to work out which one is
    <Picture 7>, and a video's six sampled frames sit in the same flat list as the
    standalone images. The inline labels make each part self-identifying, which is
    what system_prompt.md's per-type handling rules key off.
    """


    tagged = references.assign_tags()
    accepts = _DEFAULT_ENDPOINT.accepts if accepts is None else accepts
    takes_video = "video" in accepts
    takes_audio = "audio" in accepts


    concise = not takes_video

    manifest: list[str] = []
    groups: dict[str, list[str]] = {"Images": [], "Videos": [], "Audio": []}
    files: list[str] = []
    parts: list[dict] = []

    for t in tagged:
        path = os.path.join(input_dir, t.file)
        label = t.tag if t.audio_tag is None else f"{t.tag} {t.audio_tag}"
        if concise:
            files.append(f"{t.tag}={t.file}")
        else:
            manifest.append(f"{label}: {t.file}")


        if t.kind == "image":
            tensor = media.load_image(path, crop=t.ref.crop)
            if concise:
                groups["Images"].append(t.tag)
                parts.append(_text_part(f"{t.tag}:"))
            else:
                parts.append(_text_part(f"image_reference {t.tag} ({t.file}) - the next image:"))
            parts.append(_image_part(_tensor_to_jpeg_b64(tensor)))

        elif t.kind == "video" and not takes_video:


            b64s = _sampled_frame_b64s(path, crop=t.ref.crop, trim=t.ref.trim)
            groups["Videos"].append(f"{t.tag} ({len(b64s)} still frames, no sound)")
            parts.append(_text_part(
                f"{t.tag} - the next {len(b64s)} images are still frames from ONE video, "
                "in playback order. They are not separate pictures. No sound."
            ))
            for b64 in b64s:
                parts.append(_image_part(b64))
            if t.audio_tag is not None:
                groups["Audio"].append(f"{t.audio_tag} (not sent)")

        elif t.kind == "video":


            data, mime = media.video_clip_bytes(path, crop=t.ref.crop, trim=t.ref.trim)
            edited = t.ref.crop is not None or t.ref.trim is not None
            parts.append(
                _text_part(f"video_reference {t.tag} ({t.file}) - the next video, whole:")
            )
            parts.append(_video_part(data, mime))
            if t.audio_tag is not None:
                if not edited:


                    manifest.append(f"  ({t.audio_tag}: the soundtrack inside {t.tag})")
                else:
                    _, audio = media.load_video(
                        path, target_fps=24, crop=t.ref.crop, trim=t.ref.trim
                    )
                    if audio is not None:
                        parts.append(
                            _text_part(
                                f"audio_reference {t.audio_tag} - the soundtrack of {t.tag} "
                                f"({t.file}), the next audio clip:"
                            )
                        )
                        parts.append(_video_audio_part(audio, t.audio_tag, t.file))
                    else:
                        manifest.append(f"  ({t.audio_tag}: {t.file}'s soundtrack could not be read)")

        elif not takes_audio:


            groups["Audio"].append(f"{t.tag} (not sent)")

        else:
            parts.append(_text_part(f"audio_reference {t.tag} ({t.file}) - the next audio clip:"))
            if t.ref.trim is not None:


                parts.append(_video_audio_part(media.load_audio(path, trim=t.ref.trim), t.tag, t.file))
            else:
                parts.append(_file_audio_part(path, t.tag, t.file))

    text = "USER DIRECTION:\n" + direction
    fmt = _target_format_lines(width, height, length_seconds)
    if fmt:
        text += "\n\nTARGET FORMAT:\n" + "\n".join(fmt)


    if concise:
        used = [(name, items) for name, items in groups.items() if items]
        if used:
            width_ = max(len(name) for name, _ in used) + 1
            lines = [f"{(name + ':').ljust(width_)} {', '.join(items)}" for name, items in used]
            text += "\n\nREFERENCES\n" + "\n".join(lines)


            text += "\n\nfiles: " + " · ".join(files)
    elif manifest:
        text += "\n\nReference manifest:\n" + "\n".join(manifest)
    return [_text_part(text)] + parts


BASE64_PLACEHOLDER = "<BASE64_STRING>"


def _part_summary(part: dict) -> tuple[str, str]:
    """(type, one-line description) for the index at the top of the render."""
    kind = part.get("type")
    if kind == "text":
        first = (part.get("text") or "").strip().splitlines()
        head = first[0] if first else "(empty)"
        if len(head) > 62:
            head = head[:59] + "..."
        return "text", head
    if kind == "image_url":
        url = part.get("image_url", {}).get("url", "")
        b64 = url.split(",", 1)[1] if "," in url else ""
        mime = url[5 : url.find(";")] if url.startswith("data:") else "image"
        return "image_url", f"{mime}, ~{_b64_size(b64):,} bytes"
    if kind == "input_audio":
        ia = part.get("input_audio", {})
        return "input_audio", f"{ia.get('format', '?')}, ~{_b64_size(ia.get('data', '')):,} bytes"
    return str(kind), "(unrecognized part type)"


def render_payload(payload: dict, ep=None) -> str:
    """The whole request body as readable text, base64 stubbed out.

    Rendered from the payload dict itself, immediately before it is posted, so the
    debug output can never drift from what actually went over the wire. The API key
    lives in the Authorization header and is NOT part of this dict - keep it that way;
    nothing here may ever be handed the headers.

    Base64 is replaced by BASE64_PLACEHOLDER rather than dropped: the part still shows
    its real `data:<mime>;base64,` prefix, so the reader sees the actual shape that was
    sent instead of a summary of it, without a megabyte of payload in a text socket.

    The index at the top exists because the body is a flat list of parts and the
    interesting question is usually "how many images went, and which reference does each
    one belong to" - which is otherwise only answerable by scrolling through the lot.
    """
    ep = ep or _DEFAULT_ENDPOINT
    lines: list[str] = [f"POST {ep.chat_url}", f"model: {payload.get('model')}"]
    for extra in ("reasoning", "chat_template_kwargs", "temperature", "max_tokens"):
        if extra in payload:
            lines.append(f"{extra}: {payload[extra]}")
    if "reasoning" not in payload:
        lines.append("reasoning: (not sent to this endpoint)")


    for message in payload.get("messages", []):
        content = message.get("content")
        role = str(message.get("role", "?")).upper()
        if isinstance(content, str):
            lines.append(f"\n--- {role} MESSAGE: one text block, {len(content):,} chars ---")
            continue
        parts = list(content or [])
        counts: dict[str, int] = {}
        for p in parts:
            counts[p.get("type")] = counts.get(p.get("type"), 0) + 1
        tally = ", ".join(f"{v}x {k}" for k, v in counts.items())
        lines.append(f"\n--- {role} MESSAGE: {len(parts)} content parts ({tally}) ---")
        for i, p in enumerate(parts, 1):
            kind, desc = _part_summary(p)
            lines.append(f"  #{i:<3} {kind:<12} {desc}")


    messages = list(payload.get("messages", []))
    ordered = [m for m in messages if m.get("role") != "system"]
    ordered += [m for m in messages if m.get("role") == "system"]
    if len(ordered) > 1:
        lines.append("")
        lines.append("(shown user-message-first for reading; on the wire the system message goes first)")

    for message in ordered:
        role = message.get("role")
        content = message.get("content")
        lines.append("")
        if isinstance(content, str):
            lines.append(f"===== {str(role).upper()} MESSAGE =====")
            lines.append(content)
            continue
        parts = list(content or [])
        lines.append(f"===== {str(role).upper()} MESSAGE ({len(parts)} parts) =====")
        for i, part in enumerate(parts, 1):
            kind = part.get("type")
            lines.append("")
            if kind == "text":
                lines.append(f"[{i}/{len(parts)} · text]")
                lines.append(part.get("text", ""))
            elif kind == "image_url":
                url = part.get("image_url", {}).get("url", "")
                b64 = url.split(",", 1)[1] if "," in url else ""
                prefix = url.split(",", 1)[0] if "," in url else url
                mime = url[5 : url.find(";")] if url.startswith("data:") else "image"
                lines.append(f"[{i}/{len(parts)} · image_url · {mime} · ~{_b64_size(b64):,} bytes]")
                lines.append(f"    {prefix},{BASE64_PLACEHOLDER}")
            elif kind == "input_audio":
                ia = part.get("input_audio", {})
                fmt = ia.get("format", "?")
                lines.append(
                    f"[{i}/{len(parts)} · input_audio · {fmt} · ~{_b64_size(ia.get('data', '')):,} bytes]"
                )
                lines.append(f'    {{"format": "{fmt}", "data": "{BASE64_PLACEHOLDER}"}}')
            else:
                lines.append(f"[{i}/{len(parts)} · {kind!r}]")
                lines.append("    (unrecognized part type)")
    return "\n".join(lines)


def _b64_size(b64: str) -> int:
    """Decoded byte count of a base64 string, near enough for a debug line."""
    return max(0, len(b64) * 3 // 4)


def _short_reason(resp: requests.Response) -> str:
    try:
        msg = resp.json().get("error", {}).get("message")
        if msg:
            return str(msg)[:200]
    except ValueError:
        pass
    return (resp.text or "")[:200]


def write_prompt(
    *, references, input_dir: str, direction: str, api_key: str, model: str,
    system_prompt: str = "", width: int = 0, height: int = 0, length_seconds: float = 0.0,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT, job_type: str = "auto",
    classifier_model: str = "", debug: list | None = None,
    provider: str = "openrouter", api_base: str = "",
    disable_local_thinking: bool = False,
) -> str:
    """Pass a list as `debug` to have the rendered payload appended to it. A sink rather
    than a second return value, so the frozen `-> str` contract and every existing caller
    stay untouched, and so the payload survives a raised PromptError.

    `provider`/`api_base` default to today's behaviour, so every existing caller and test
    that omits them posts to OpenRouter exactly as before."""
    ep = _endpoint.resolve(provider, api_base)
    key = _key_for(ep, api_key)


    resolved = job_type if job_type in ("standard", "replacement") else "standard"
    if job_type == "auto":


        classifier = classifier_model or ("" if ep.is_openrouter else model)
        resolved = classify_mode(
            direction=direction, references=references, api_key=api_key,
            model=classifier, ep=ep, disable_local_thinking=disable_local_thinking,
        )


    if not (system_prompt and system_prompt.strip()):
        system_prompt = _read_system_prompt(resolved)
    content = _build_content(
        references, input_dir, direction,
        width=width, height=height, length_seconds=length_seconds, accepts=ep.accepts,
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
    }


    if ep.sends_reasoning and reasoning_effort in REASONING_EFFORTS:
        payload["reasoning"] = {"effort": reasoning_effort}
    if disable_local_thinking and not ep.is_openrouter:
        payload["chat_template_kwargs"] = {"enable_thinking": False}


    if debug is not None:


        routing = f"job_type: {job_type} -> {resolved} (system prompt: {resolved}.md)"
        routing += f"\nendpoint: {ep.describe()}"
        if ep.degrades:
            missing = ", ".join(sorted(_DEFAULT_ENDPOINT.accepts - ep.accepts))
            routing += (
                f"\nDEGRADED: this endpoint does not accept {missing}. Videos were sent "
                f"as {_DEGRADED_FRAMES} still frames and no audio was sent at all."
            )
        debug.append(routing + "\n" + render_payload(payload, ep))

    parts = payload["messages"][1]["content"]
    with logs.timed(
        "chat", endpoint=ep.describe(), model=model, job_type=resolved,
        effort=reasoning_effort if ep.sends_reasoning else None,
        degraded=ep.degrades or None, timeout_s=ep.chat_timeout,
        parts=len(parts), bytes=_payload_bytes(parts),
    ) as fields:
        resp = _post_chat(key, payload, ep)
        fields["status"] = resp.status_code
        _record_usage(resp, fields)
        message = _completion_of(resp, ep)
        fields["chars"] = len(message)
    return message


def _payload_bytes(parts) -> int:
    """Roughly what goes on the wire: the base64 media dominates, the text is noise."""
    total = 0
    for part in parts:
        if part.get("type") == "text":
            total += len(part.get("text", ""))
        elif part.get("type") == "image_url":
            total += len(part["image_url"]["url"])
        elif part.get("type") == "video_url":
            total += len(part["video_url"]["url"])
        elif part.get("type") == "input_audio":
            total += len(part["input_audio"].get("data", ""))
    return total


def _record_usage(resp, fields) -> None:
    """OpenRouter reports tokens and cost per call when it feels like it - log what is
    there, never invent what is not."""
    try:
        usage = resp.json().get("usage") or {}
    except (ValueError, AttributeError):
        return
    for key_name in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cost"):
        if key_name in usage:
            fields[key_name] = usage[key_name]
    details = usage.get("completion_tokens_details") or {}
    if "reasoning_tokens" in details:
        fields["reasoning_tokens"] = details["reasoning_tokens"]


def _post_chat(key, payload, ep=None):
    ep = ep or _DEFAULT_ENDPOINT
    try:
        return requests.post(
            ep.chat_url,
            headers=_auth_header(key),
            json=payload,
            timeout=ep.chat_timeout,
        )
    except requests.Timeout:


        logs.log(
            "chat_timeout", endpoint=ep.describe(), limit_s=ep.chat_timeout, ok=False,
        )
        raise PromptError(
            f"{ep.describe()} did not answer within {ep.chat_timeout}s. A large model on "
            "CPU can exceed this; try a smaller model, or a machine with more to give."
        ) from None
    except requests.RequestException as e:


        raise PromptError(f"{ep.describe()} request failed: {type(e).__name__}") from None


def _completion_of(resp, ep=None) -> str:
    ep = ep or _DEFAULT_ENDPOINT
    who = ep.describe()
    if resp.status_code != 200:
        raise PromptError(f"{who} returned {resp.status_code}: {_short_reason(resp)}")

    try:
        choice = resp.json()["choices"][0]["message"]
        message = choice["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        raise PromptError(f"{who} response was malformed") from None

    if not isinstance(message, str) or not message.strip():


        if isinstance(choice, dict) and (choice.get("reasoning_content") or "").strip():
            raise PromptError(
                f"{who} returned only reasoning and no answer: the model used its whole "
                "output budget thinking. Raise the model's token limit, turn its "
                "reasoning down, or use a non-reasoning model."
            )
        raise PromptError(f"{who} returned an empty completion")

    return message.strip()
