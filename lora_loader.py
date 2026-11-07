import json
import math

import folder_paths
from comfy_api.latest import io
from nodes import LoraLoader


def parse_loras(raw):
    entries = json.loads(raw)
    if not isinstance(entries, list):
        raise ValueError('LoRAs must be a JSON list.')
    for index, entry in enumerate(entries, 1):
        if not isinstance(entry, dict) or not isinstance(entry.get('name'), str):
            raise ValueError(f'LoRA {index}: expected a filename.')
        if not isinstance(entry.get('enabled', True), bool):
            raise ValueError(f'LoRA {index}: enabled must be true or false.')
        for key in ('strength_model', 'strength_clip'):
            value = entry.get(key, 1.0)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'LoRA {index}: {key} must be a finite number.')
    return entries


class TMMultiLora(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='TMMultiLora', display_name='TM · Power LoRA',
            category='Tensor Master/Loaders',
            inputs=[
                io.Model.Input('model'),
                io.String.Input('loras_json', default='[]', multiline=True, dynamic_prompts=False),
                io.Clip.Input('clip', optional=True),
            ],
            outputs=[io.Model.Output(), io.Clip.Output()],
        )

    @classmethod
    def execute(cls, model, loras_json='[]', clip=None):
        entries = parse_loras(loras_json)
        active = [entry for entry in entries if entry.get('enabled', True) and (
            entry.get('strength_model', 1.0) != 0 or (clip is not None and entry.get('strength_clip', 1.0) != 0))]
        available = set(folder_paths.get_filename_list('loras')) if active else set()
        for entry in active:
            if entry['name'] not in available:
                raise ValueError(f"LoRA not found in the LoRA folders: {entry['name']}")
            folder_paths.get_full_path_or_raise('loras', entry['name'])
        loader = LoraLoader()
        for entry in active:
            model, clip = loader.load_lora(model, clip, entry['name'], entry.get('strength_model', 1.0),
                                          entry.get('strength_clip', 1.0) if clip is not None else 0.0)
        return io.NodeOutput(model, clip)
