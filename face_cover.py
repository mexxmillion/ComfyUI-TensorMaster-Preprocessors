from pathlib import Path
from importlib import import_module
import re

import cv2
import numpy as np
import torch
from comfy_api.latest import io


def load_retinaface(confidence, max_faces_per_image):
    registry = import_module('nodes').NODE_CLASS_MAPPINGS
    provider = registry.get('AutoCropFaces')
    if provider is None:
        raise RuntimeError('Enable ComfyUI-AutoCropFaces to use its RetinaFace detector, or select a frontal detector.')
    backend = import_module(provider.__module__).Pytorch_RetinaFace
    return backend(device='cpu', vis_thres=confidence, confidence_threshold=confidence,
                   top_k=max(5000, max_faces_per_image), keep_top_k=max_faces_per_image)


def select_faces(detections, number_of_faces, start_index, max_faces_per_image, order):
    if order == 'left to right':
        detections = sorted(detections, key=lambda face: (face[0], face[1]))
    elif order == 'largest first':
        detections = sorted(detections, key=lambda face: face[2] * face[3], reverse=True)
    elif order == 'confidence':
        detections = sorted(detections, key=lambda face: face[4], reverse=True)
    else:
        raise ValueError('Choose a supported face selection order.')
    detections = detections[:max_faces_per_image]
    return detections[start_index:start_index + number_of_faces] if number_of_faces else detections[start_index:]


def face_cover_mask(image, detector='retinaface', shape='ellipse', scale_x=1.35, scale_y=1.5,
                    neighbors=5, min_face_size=8, size=1.0, number_of_faces=0, start_index=0,
                    max_faces_per_image=50, confidence=0.6, order='confidence', shift_factor=0.5):
    if image.shape[-1] not in (3, 4):
        raise ValueError('Face Cover expects RGB or RGBA images.')
    if detector not in ('retinaface', 'frontal', 'frontal + profile') or shape not in ('ellipse', 'circle'):
        raise ValueError('Choose a supported face detector and cover shape.')
    retinaface = frontal = profile = None
    if detector == 'retinaface':
        retinaface = load_retinaface(confidence, max_faces_per_image)
    else:
        root = Path(cv2.data.haarcascades)
        frontal = cv2.CascadeClassifier(str(root / 'haarcascade_frontalface_default.xml'))
        profile = cv2.CascadeClassifier(str(root / 'haarcascade_profileface.xml')) if detector == 'frontal + profile' else None
        if frontal.empty() or (profile is not None and profile.empty()):
            raise RuntimeError('OpenCV face detector files are missing. Reinstall opencv-python.')
    masks = []
    for frame in image[..., :3].detach().float().cpu().numpy():
        if retinaface is not None:
            bgr = torch.from_numpy(np.ascontiguousarray(frame[..., ::-1].clip(0, 1) * 255))
            detected = retinaface.detect_faces(bgr)
            detections = [(float(x1), float(y1), float(x2 - x1), float(y2 - y1), float(score))
                          for x1, y1, x2, y2, score, *_ in detected
                          if score >= confidence and x2 - x1 >= min_face_size and y2 - y1 >= min_face_size]
        else:
            gray = cv2.cvtColor((frame.clip(0, 1) * 255).round().astype(np.uint8), cv2.COLOR_RGB2GRAY)
            boxes = list(frontal.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=neighbors,
                                                  minSize=(min_face_size, min_face_size)))
            if profile is not None:
                boxes.extend(profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=neighbors,
                                                      minSize=(min_face_size, min_face_size)))
                for x, y, w, h in profile.detectMultiScale(cv2.flip(gray, 1), scaleFactor=1.1,
                        minNeighbors=neighbors, minSize=(min_face_size, min_face_size)):
                    boxes.append((gray.shape[1] - int(x) - int(w), y, w, h))
            detections = [(*map(float, box), 1.0) for box in boxes]
            if len(detections) > 1:
                keep = cv2.dnn.NMSBoxes([list(face[:4]) for face in detections], [1.0] * len(detections), 0, 0.4)
                detections = [detections[int(i)] for i in np.asarray(keep).reshape(-1)]
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for x, y, w, h, _ in select_faces(detections, number_of_faces, start_index, max_faces_per_image, order):
            axes = (max(1, round(int(w) * scale_x * size / 2)), max(1, round(int(h) * scale_y * size / 2)))
            if shape == 'circle':
                axes = (max(axes), max(axes))
            center = (round(x + w / 2), round(y + h / 2 + 2 * axes[1] * (0.5 - shift_factor)))
            if shape == 'circle':
                cv2.circle(mask, center, max(axes), 1, thickness=-1, lineType=cv2.LINE_8)
            else:
                cv2.ellipse(mask, center, axes, 0, 0, 360, 1, thickness=-1, lineType=cv2.LINE_8)
        masks.append(mask)
    return torch.from_numpy(np.stack(masks)).to(device=image.device, dtype=image.dtype)


def apply_face_cover(image, mask, color='#808080'):
    if not isinstance(color, str) or re.fullmatch(r'#[0-9a-fA-F]{6}', color) is None:
        raise ValueError('Face cover color must be a hex color such as #808080.')
    fill = image.new_tensor([int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    rgb = torch.where(mask[..., None] > 0, fill, image[..., :3])
    return torch.cat((rgb, image[..., 3:]), dim=-1) if image.shape[-1] == 4 else rgb


class TMFaceCover(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='TMFaceCover', display_name='TM · Face Cover (solid ellipse)',
            category='Tensor Master/Reference preprocessing',
            description='Cover selected faces with opaque circles or ellipses. RetinaFace reuses the installed Auto Crop Faces node. No downloads.',
            inputs=[
                io.Image.Input('image'),
                io.Combo.Input('detector', options=['retinaface', 'frontal', 'frontal + profile'], default='retinaface'),
                io.Combo.Input('shape', options=['ellipse', 'circle'], default='ellipse'),
                io.String.Input('color', default='#808080'),
                io.Float.Input('scale_x', default=1.35, min=0.1, max=5, step=0.05),
                io.Float.Input('scale_y', default=1.5, min=0.1, max=5, step=0.05),
                io.Int.Input('neighbors', default=5, min=1, max=30),
                io.Int.Input('min_face_size', default=8, min=1, max=2048),
                io.Float.Input('size', display_name='scale_factor', default=1.0, min=0.1, max=5, step=0.1, optional=True,
                    tooltip='Cover size multiplier. Increase to cover hair as well as the face.'),
                io.Int.Input('number_of_faces', default=0, min=0, max=1000, optional=True,
                    tooltip='Faces to cover per image. 0 = all remaining faces from start_index.'),
                io.Int.Input('start_index', default=0, min=0, max=1000, optional=True,
                    tooltip='Zero-based first face in the selected order, independently for each image.'),
                io.Int.Input('max_faces_per_image', default=50, min=1, max=1000, optional=True),
                io.Float.Input('confidence', default=0.6, min=0, max=1, step=0.05, optional=True,
                    tooltip='RetinaFace confidence threshold.'),
                io.Combo.Input('order', options=['confidence', 'left to right', 'largest first'], default='confidence', optional=True),
                io.Float.Input('shift_factor', default=0.5, min=0, max=1, step=0.05, optional=True,
                    tooltip='0.5 centers the cover. Higher values shift it upward toward the hair.'),
            ],
            outputs=[io.Image.Output(display_name='covered_image'), io.Mask.Output(display_name='cover_mask')],
        )

    @classmethod
    def execute(cls, image, detector='retinaface', shape='ellipse', color='#808080', scale_x=1.35,
                scale_y=1.5, neighbors=5, min_face_size=8, size=1.0, number_of_faces=0, start_index=0,
                max_faces_per_image=50, confidence=0.6, order='confidence', shift_factor=0.5):
        mask = face_cover_mask(image, detector, shape, scale_x, scale_y, neighbors, min_face_size, size,
                               number_of_faces, start_index, max_faces_per_image, confidence, order, shift_factor)
        return io.NodeOutput(apply_face_cover(image, mask, color), mask)
