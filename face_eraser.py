from pathlib import Path
import hashlib
import json

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi


def prepare_mask(mask, grow, feather):
    hard = ndi.binary_fill_holes(mask > 0.5)
    if grow:
        hard = ndi.distance_transform_edt(~hard) <= grow if hard.any() else hard
    # Feather outward only: original face pixels always get fully replaced.
    if feather and hard.any():
        outside = ndi.distance_transform_edt(~hard)
        alpha = np.clip(1.0 - outside / feather, 0, 1)
    else:
        alpha = hard.astype(np.float32)
    return alpha.astype(np.float32)


def gauss_fill(image, alpha, sigma, mode, gray):
    if not np.any(alpha):
        return image.copy()
    if mode == "neutral gray":
        fill = np.full_like(image, gray)
    else:
        # Exclude the entire feather footprint, not just its opaque center.
        known = (alpha == 0).astype(np.float32)
        if not known.any():
            fill = np.full_like(image, gray)
        else:
            # Extend clean pixels into the hole before normalized Gaussian fill.
            # This stabilizes the center of holes larger than the Gaussian kernel.
            indices = ndi.distance_transform_edt(known == 0, return_distances=False,
                                                return_indices=True)
            extension = image[tuple(indices)]
            numerator = ndi.gaussian_filter(image * known[..., None],
                                             (sigma, sigma, 0), mode="nearest")
            denominator = ndi.gaussian_filter(known, sigma, mode="nearest")
            fallback = ndi.gaussian_filter(extension, (sigma, sigma, 0), mode="nearest")
            fill = np.divide(numerator, denominator[..., None], out=fallback,
                             where=denominator[..., None] > 1e-6)
    return image * (1 - alpha[..., None]) + fill * alpha[..., None]


class FaceHairEraser:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "fill_mode": (["gaussian surroundings", "neutral gray"],),
            "grow": ("INT", {"default": 16, "min": 0, "max": 512}),
            "feather": ("INT", {"default": 12, "min": 0, "max": 256}),
            "fill_sigma": ("FLOAT", {"default": 48.0, "min": 1.0, "max": 512.0}),
            "include_neck": ("BOOLEAN", {"default": False}),
            "include_hat": ("BOOLEAN", {"default": True}),
            "neutral_value": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
        }, "optional": {
            "extra_mask": ("MASK", {"tooltip": "White adds to the automatic mask."}),
            "mask_only": ("BOOLEAN", {"default": False,
                         "tooltip": "Use extra_mask alone; skips the segmentation model."}),
            "segmentation_confidence": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0,
                "step": 0.05, "tooltip": "Higher rejects uncertain face/hair pixels. 0 preserves the original behavior."}),
            "detected_faces_only": ("BOOLEAN", {"default": False,
                "tooltip": "Only segment crops around detected frontal faces. Reduces false masks elsewhere; may miss profiles."}),
            "detector_neighbors": ("INT", {"default": 5, "min": 1, "max": 30,
                "tooltip": "Higher requires stronger agreement from the face detector, rejecting weak detections."}),
            "min_face_size": ("INT", {"default": 24, "min": 8, "max": 2048,
                "tooltip": "Ignore face detections smaller than this many pixels. Applies to crop detection only."}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("clean_image", "removal_mask", "mask_preview")
    FUNCTION = "erase"
    CATEGORY = "Tensor Master/Reference preprocessing"
    DESCRIPTION = "Remove face, ears, hair and accessories with a detail-free fill. White mask pixels are removed."

    def load_parser(self):
        import folder_paths
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

        root = Path(folder_paths.models_dir)
        manifest = json.loads(Path(__file__).with_name("models.lock.json").read_text())
        for item in manifest["artifacts"]:
            if item["group"] != "face-eraser":
                continue
            path = root / item["local_dir"] / item["file"]
            if not path.is_file():
                raise RuntimeError("Face Eraser weights missing. Run this repository's sync_models.py.")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if path.stat().st_size != item["size"] or digest != item["sha256"]:
                raise RuntimeError(f"Face Eraser checkpoint verification failed: {path}")
        model_path = root / "face_eraser"
        processor = SegformerImageProcessor.from_pretrained(model_path, local_files_only=True)
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_path, local_files_only=True, use_safetensors=True).eval()
        return processor, model

    def segment(self, image, include_neck, include_hat, processor, model,
                segmentation_confidence=0.0, detected_faces_only=False,
                detector_neighbors=5, min_face_size=24):
        rgb = (image.clip(0, 1) * 255).astype(np.uint8)
        mask = (np.zeros(rgb.shape[:2], dtype=np.float32) if detected_faces_only else
                self.parse_crop(rgb, include_neck, include_hat, processor, model, segmentation_confidence))
        # Re-parse detected faces with hair context: a full-frame parser can
        # miss a small face next to a much larger portrait in reference sheets.
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        detector = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
        for x, y, w, h in detector.detectMultiScale(gray, scaleFactor=1.1,
                minNeighbors=detector_neighbors, minSize=(min_face_size, min_face_size)):
            left, top = max(0, x - w), max(0, y - h)
            right, bottom = min(rgb.shape[1], x + 2 * w), min(rgb.shape[0], y + 3 * h)
            parsed = self.parse_crop(rgb[top:bottom, left:right], include_neck, include_hat,
                                     processor, model, segmentation_confidence)
            mask[top:bottom, left:right] = np.maximum(mask[top:bottom, left:right], parsed)
        return mask

    def parse_crop(self, rgb, include_neck, include_hat, processor, model, segmentation_confidence=0.0):
        pil = Image.fromarray(rgb)
        inputs = processor(images=pil, return_tensors="pt")
        logits = model(**inputs).logits
        logits = F.interpolate(logits, size=rgb.shape[:2], mode="bilinear", align_corners=False)
        labels = logits.argmax(1)[0].detach().numpy()
        selected = list(range(1, 14)) + [15]
        if include_hat:
            selected.append(14)
        if include_neck:
            selected.extend([16, 17])
        mask = np.isin(labels, selected)
        if segmentation_confidence > 0:
            confidence = logits.softmax(1).amax(1)[0].detach().numpy()
            mask &= confidence >= segmentation_confidence
        return mask.astype(np.float32)

    def erase(self, image, fill_mode="gaussian surroundings", grow=16, feather=12,
              fill_sigma=48.0, include_neck=False, include_hat=True, neutral_value=0.5,
              extra_mask=None, mask_only=False, segmentation_confidence=0.0,
              detected_faces_only=False, detector_neighbors=5, min_face_size=24):
        if image.shape[-1] != 3:
            raise ValueError("Face Eraser expects an RGB IMAGE; convert RGBA to RGB first.")
        if mask_only and extra_mask is None:
            raise ValueError("Connect extra_mask when mask_only is enabled.")
        if fill_mode not in ("gaussian surroundings", "neutral gray"):
            raise ValueError("Unknown fill mode.")
        extra = None
        if extra_mask is not None:
            extra = extra_mask.detach().float().cpu()
            if extra.ndim == 2:
                extra = extra.unsqueeze(0)
            if extra.shape[0] not in (1, image.shape[0]):
                raise ValueError("Mask batch must contain one mask or match the image batch.")
            extra = F.interpolate(extra[:, None], size=image.shape[1:3], mode="bilinear",
                                  align_corners=False)[:, 0].numpy()
        processor, model = (None, None) if mask_only else self.load_parser()
        results, masks, previews = [], [], []
        for i, frame in enumerate(image.detach().float().cpu().numpy()):
            mask = (np.zeros(frame.shape[:2], dtype=np.float32) if mask_only else
                    self.segment(frame, include_neck, include_hat, processor, model,
                                 segmentation_confidence, detected_faces_only, detector_neighbors, min_face_size))
            if extra is not None:
                mask = np.maximum(mask, extra[0 if len(extra) == 1 else i])
            alpha = prepare_mask(mask, grow, feather)
            results.append(gauss_fill(frame, alpha, fill_sigma, fill_mode, neutral_value))
            masks.append(alpha)
            overlay = frame.copy()
            overlay[..., 0] = 1
            overlay[..., 1:] *= 0.15
            previews.append(frame * (1 - alpha[..., None] * 0.65) + overlay * alpha[..., None] * 0.65)
        return tuple(torch.from_numpy(np.stack(items).astype(np.float32))
                     for items in (results, masks, previews))
