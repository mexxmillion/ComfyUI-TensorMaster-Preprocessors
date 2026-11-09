from importlib import import_module
import json

import nodes

from . import prompt as writer, refs


PROVIDER = 'Local Gemma 4 12B (Director)'
MODEL = 'Gemma 4 12B QAT — Uncensored Balanced (Q4_K_M)'


def write_prompt(direction, outputs, references, system_prompt, width, height, length_seconds):
    director = nodes.NODE_CLASS_MAPPINGS.get('PromptDirector')
    if director is None:
        raise writer.PromptError('Install and enable ComfyUI-PromptDirector to use local Gemma.')
    images = {f'image_{i}': outputs[refs.slot_index(f'image_{i}')] for i in range(1, 10)
              if outputs[refs.slot_index(f'image_{i}')] is not None}
    videos = {f'video_{i}': outputs[refs.slot_index(f'video_{i}')] for i in range(1, 4)
              if outputs[refs.slot_index(f'video_{i}')] is not None}
    audios = {f'audio_{i}': outputs[refs.slot_index(f'audio_{i}')] for i in range(1, 4)
              if outputs[refs.slot_index(f'audio_{i}')] is not None}
    local_models = import_module(director.__module__.rsplit('.', 1)[0] + '.local_models')
    profile = local_models.resolve_local_model(MODEL, bool(images or videos))
    for path in (profile.model_path, profile.mmproj_path):
        if path is not None and not path.is_file():
            raise writer.PromptError('Install the Gemma 4 12B model and vision projector in Prompt Director Settings first.')
    manifest = [tagged.to_dict() for tagged in references.assign_tags()]
    try:
        result = director.execute(
            prompt=direction, mode='MiniMax H3 — Reference Manager', profile='', creativity='Balanced',
            duration=length_seconds, seed=-1, max_attempts=1, width=width, height=height,
            local_llama_thinking=False, prompt_provider='local model', local_model=MODEL,
            reference_images=images, reference_videos=videos, reference_audios=audios,
            reference_context=json.dumps(manifest, ensure_ascii=False),
            custom_system_prompt=system_prompt.strip() or writer._read_system_prompt(), fail_on_error=True,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise writer.PromptError(f'Local Gemma: {error}') from error
    return result.result[0]
