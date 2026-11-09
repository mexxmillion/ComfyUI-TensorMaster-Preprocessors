import hashlib
import json
import math
from pathlib import Path
from uuid import uuid4

import folder_paths
import comfy.utils
import torch
from PIL import Image
from comfy_api.latest import io

from ..nodes import TMCompositeReference, TMIsolateReference
from ..prepare import TMDepthReference
from ..face_cover import face_cover_mask, apply_face_cover
from . import director, media, prompt as writer, refs


def input_path(name):
    root = Path(folder_paths.get_input_directory()).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'Reference is not a file inside the input directory: {name}')
    return str(path)


def reference_entries(raw):
    data = json.loads(raw or '{"references": []}')
    entries = data.get('references', []) if isinstance(data, dict) else data
    if not isinstance(entries, list) or not all(isinstance(r, dict) for r in entries):
        raise ValueError('References must be a list of reference objects.')
    reference_set = refs.ReferenceSet.from_obj(entries)
    reference_set.validate()
    for entry in entries:
        input_path(entry['file'])
        for field in ('crop', 'trim'):
            if entry.get(field) is not None and not all(math.isfinite(v) for v in entry[field]):
                raise ValueError(f'{field} must contain finite numbers.')
        options = entry.get('preprocess', {})
        if not isinstance(options, dict):
            raise ValueError('Preprocessing settings must be an object.')
        if options and entry['kind'] != 'image':
            raise ValueError('Depth and isolation currently process still images.')
        if set(options) - {'isolate', 'depth', 'invert_depth', 'text', 'confidence', 'selection', 'background', 'crop_subject',
                           'face_cover', 'face_detector', 'face_shape', 'face_color', 'face_scale_x', 'face_scale_y',
                           'face_neighbors', 'face_min_size', 'face_size', 'face_number', 'face_start_index',
                           'face_max_faces', 'face_confidence', 'face_order', 'face_shift'}:
            raise ValueError('Unknown reference preprocessing setting.')
        if options.get('background', 'gray') not in ('gray', 'white', 'black'):
            raise ValueError('Choose a gray, white or black background.')
    return entries, reference_set


def preprocess_image(image, options):
    isolate, depth = options.get('isolate', False), options.get('depth', False)
    cover = None
    if options.get('face_cover', False):
        cover = face_cover_mask(image, options.get('face_detector', 'retinaface'), options.get('face_shape', 'ellipse'),
            options.get('face_scale_x', 1.35), options.get('face_scale_y', 1.5), options.get('face_neighbors', 5),
            options.get('face_min_size', 8), options.get('face_size', 1.0), options.get('face_number', 0),
            options.get('face_start_index', 0), options.get('face_max_faces', 50), options.get('face_confidence', 0.6),
            options.get('face_order', 'confidence'), options.get('face_shift', 0.5))
    if isolate:
        rgba, mask = TMIsolateReference().isolate(image, options.get('text', 'person'),
            options.get('confidence', 0.25), options.get('selection', 'best'))
    if depth:
        image = TMDepthReference().depth(image, options.get('invert_depth', False))[0]
        if isolate:
            rgba = torch.cat((image * mask.unsqueeze(-1), mask.unsqueeze(-1)), dim=-1)
    if cover is not None:
        if isolate:
            rgba = apply_face_cover(rgba, cover, options.get('face_color', '#808080'))
        else:
            image = apply_face_cover(image, cover, options.get('face_color', '#808080'))
    if isolate:
        image = TMCompositeReference().composite(rgba, options.get('background', 'gray'),
                                                options.get('crop_subject', False))[0]
    return image


def resize_reference(image, width, height):
    if image.shape[1:3] == (height, width):
        return image
    return comfy.utils.common_upscale(image.movedim(-1, 1), width, height, 'bilinear', 'disabled').movedim(1, -1)


class TMH3ReferenceManager(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='TMH3ReferenceManager', display_name='TM · H3 Reference Manager',
            category='Tensor Master/References', is_output_node=True,
            inputs=[
                io.String.Input('references_json', default='{"references": []}', multiline=True, dynamic_prompts=False),
                io.String.Input('direction', default='', multiline=True, dynamic_prompts=False),
                io.Boolean.Input('ai_enhance', default=False),
                io.Combo.Input('prompt_provider', options=['local', 'openrouter', director.PROVIDER], default='local'),
                io.String.Input('model', default=writer.DEFAULT_MODEL),
                io.String.Input('api_base', default='http://localhost:1234/v1'),
                io.String.Input('api_key', default=''),
                io.Combo.Input('reasoning_effort', options=list(writer.REASONING_EFFORTS), default='medium'),
                io.String.Input('system_prompt', default='', multiline=True, dynamic_prompts=False),
                io.Int.Input('max_reference_edge', default=2048, min=0, max=8192, step=64),
                io.String.Input('prompt', optional=True, force_input=True,
                    tooltip='Connected text takes precedence over the text box. AI enhance still applies.'),
                io.Int.Input('width', default=0, min=0, max=16384, optional=True, force_input=True,
                    tooltip='Target video width. Informs AI enhancement and optional reference resizing. 0 = unspecified.'),
                io.Int.Input('height', default=0, min=0, max=16384, optional=True, force_input=True,
                    tooltip='Target video height. Informs AI enhancement and optional reference resizing. 0 = unspecified.'),
                io.Float.Input('length_seconds', default=0, min=0, optional=True, force_input=True,
                    tooltip='Target video duration in seconds. Informs AI enhancement; does not trim references.'),
                io.Boolean.Input('resize_references', default=False, optional=True,
                    tooltip='Resize image references and video frames to the exact connected width and height.'),
            ],
            outputs=[io.Image.Output(display_name=f'image_{i}') for i in range(1, 10)]
                + [io.Image.Output(display_name=f'video_{i}') for i in range(1, 4)]
                + [io.Audio.Output(display_name=f'video_audio_{i}') for i in range(1, 4)]
                + [io.Audio.Output(display_name=f'audio_{i}') for i in range(1, 4)]
                + [io.String.Output(display_name='prompt'), io.String.Output(display_name='manifest')],
        )

    @classmethod
    def fingerprint_inputs(cls, references_json='', **kwargs):
        entries, _ = reference_entries(references_json)
        digest = hashlib.sha256()
        for entry in entries:
            with open(input_path(entry['file']), 'rb') as stream:
                digest.update(hashlib.file_digest(stream, 'sha256').digest())
        return digest.hexdigest()

    @classmethod
    def execute(cls, references_json='', direction='', ai_enhance=False, prompt_provider='local',
                model=writer.DEFAULT_MODEL, api_base='http://localhost:1234/v1', api_key='',
                reasoning_effort='medium', system_prompt='', max_reference_edge=2048, prompt=None,
                width=0, height=0, length_seconds=0, resize_references=False):
        entries, reference_set = reference_entries(references_json)
        if resize_references and any(entry['kind'] in ('image', 'video') for entry in entries) and (width <= 0 or height <= 0):
            raise ValueError('Connect a positive target width and height to resize references.')
        outputs = refs.empty_outputs()
        previews = []
        effective = []
        # Preserve grouping exactly as the native H3 reference sockets consume it.
        ordered = [entry for kind in refs.KINDS for entry in entries if entry['kind'] == kind]
        temp = Path(folder_paths.get_temp_directory()) / 'tm-references'
        temp.mkdir(parents=True, exist_ok=True)
        for entry, tagged in zip(ordered, reference_set.assign_tags()):
            ref = tagged.ref
            path = input_path(ref.file)
            effective_ref = dict(entry)
            if ref.kind == 'image':
                image = media.load_image(path, crop=ref.crop, max_edge=0 if resize_references else max_reference_edge)
                if resize_references:
                    image = resize_reference(image, width, height)
                image = preprocess_image(image, entry.get('preprocess', {}))
                if resize_references:
                    image = resize_reference(image, width, height)
                outputs[refs.slot_index(f'image_{tagged.slot}')] = image
                filename = f'{uuid4().hex}.png'
                Image.fromarray((image[0].cpu().numpy().clip(0, 1) * 255).round().astype('uint8')).save(temp / filename)
                previews.append({'source': entry, 'filename': filename, 'subfolder': 'tm-references', 'type': 'temp'})
                # The prompt writer sees exactly the edited reference sent to H3.
                effective_ref = {'kind': 'image', 'file': str(temp / filename)}
            elif ref.kind == 'video':
                frames, audio = media.load_video(path, crop=ref.crop, trim=ref.trim)
                if resize_references:
                    frames = resize_reference(frames, width, height)
                outputs[refs.slot_index(f'video_{tagged.slot}')] = frames
                if ref.use_soundtrack and audio is not None:
                    outputs[refs.slot_index(f'video_audio_{tagged.slot}')] = audio
                else:
                    effective_ref['use_soundtrack'] = False
            else:
                outputs[refs.slot_index(f'audio_{tagged.slot}')] = media.load_audio(path, trim=ref.trim)
            effective.append(effective_ref)
        text = direction if prompt is None else prompt
        if ai_enhance:
            try:
                references = refs.ReferenceSet.from_obj(effective)
                if prompt_provider == director.PROVIDER:
                    text = director.write_prompt(text, outputs, references, system_prompt, width, height, length_seconds)
                else:
                    text = writer.write_prompt(references=references,
                        input_dir=folder_paths.get_input_directory(), direction=text, provider=prompt_provider,
                        model=model, api_base=api_base, api_key=api_key, reasoning_effort=reasoning_effort,
                        system_prompt=system_prompt, width=width, height=height, length_seconds=length_seconds)
            except writer.PromptError as error:
                message = str(error)
                if api_key:
                    message = message.replace(api_key, '***')
                raise ValueError(f'Prompt enhancement failed: {message}') from None
        outputs[refs.slot_index('prompt')] = text
        manifest = [dict(tagged.to_dict(), file=source['file']) for tagged, source in
                    zip(refs.ReferenceSet.from_obj(effective).assign_tags(), ordered)]
        outputs[refs.slot_index('debug')] = json.dumps(manifest, ensure_ascii=False)
        return io.NodeOutput(*outputs, ui={'tm_reference_previews': previews, 'tm_prompt': [text]})
