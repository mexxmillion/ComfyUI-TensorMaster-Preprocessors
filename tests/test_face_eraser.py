import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("face_eraser", Path(__file__).resolve().parents[1] / "face_eraser.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
FaceHairEraser, gauss_fill, prepare_mask = module.FaceHairEraser, module.gauss_fill, module.prepare_mask


class FillTests(unittest.TestCase):
    def test_confidence_rejects_uncertain_pixels(self):
        logits = torch.zeros(1, 19, 2, 2)
        logits[0, 1, 0, 0] = 1
        logits[0, 1, 0, 1] = 10
        logits[0, 0, 1, :] = 20
        model = Mock(return_value=SimpleNamespace(logits=logits))
        processor = Mock(return_value={})
        node = FaceHairEraser()
        rgb = np.zeros((2, 2, 3), np.uint8)
        original = node.parse_crop(rgb, False, True, processor, model)
        strict = node.parse_crop(rgb, False, True, processor, model, 0.8)
        np.testing.assert_array_equal(original, [[1, 1], [0, 0]])
        np.testing.assert_array_equal(strict, [[0, 1], [0, 0]])

    def test_crop_only_skips_full_frame_and_forwards_detector_settings(self):
        node = FaceHairEraser()
        detector = Mock()
        detector.detectMultiScale.return_value = ()
        with patch.object(module.cv2, "CascadeClassifier", return_value=detector), \
                patch.object(node, "parse_crop") as parse:
            mask = node.segment(np.zeros((96, 96, 3), np.float32), False, True,
                                None, None, 0.8, True, 12, 64)
        parse.assert_not_called()
        self.assertFalse(mask.any())
        self.assertEqual(detector.detectMultiScale.call_args.kwargs["minNeighbors"], 12)
        self.assertEqual(detector.detectMultiScale.call_args.kwargs["minSize"], (64, 64))

    def test_removed_pixels_cannot_influence_fill(self):
        rng = np.random.default_rng(12)
        original = rng.random((96, 96, 3), dtype=np.float32)
        mask = np.zeros((96, 96), np.float32)
        mask[16:80, 16:80] = 1
        alpha = prepare_mask(mask, 4, 6)
        changed = original.copy()
        changed[alpha > 0] = 1 - changed[alpha > 0]
        first = gauss_fill(original, alpha, 8, "gaussian surroundings", 0.5)
        second = gauss_fill(changed, alpha, 8, "gaussian surroundings", 0.5)
        np.testing.assert_allclose(first[alpha == 1], second[alpha == 1])
        np.testing.assert_array_equal(first[alpha == 0], original[alpha == 0])
        self.assertTrue(np.isfinite(first).all())

    def test_empty_full_and_enclosed_holes(self):
        image = np.ones((32, 32, 3), np.float32)
        empty = prepare_mask(np.zeros((32, 32)), 8, 8)
        np.testing.assert_array_equal(gauss_fill(image, empty, 2, "gaussian surroundings", 0.5), image)
        full = prepare_mask(np.ones((32, 32)), 8, 8)
        np.testing.assert_allclose(gauss_fill(image, full, 2, "gaussian surroundings", 0.5), 0.5)
        ring = np.ones((32, 32))
        ring[10:20, 10:20] = 0
        self.assertEqual(prepare_mask(ring, 0, 0)[15, 15], 1)

    def test_batch_mask_broadcast_and_resize(self):
        images = torch.rand(2, 64, 96, 3)
        mask = torch.zeros(1, 32, 48)
        mask[:, 8:24, 12:36] = 1
        clean, masks, preview = FaceHairEraser().erase(images, extra_mask=mask, mask_only=True)
        self.assertEqual(clean.shape, images.shape)
        self.assertEqual(preview.shape, images.shape)
        self.assertEqual(masks.shape, (2, 64, 96))
        self.assertTrue(torch.isfinite(clean).all())


if __name__ == "__main__":
    unittest.main()
