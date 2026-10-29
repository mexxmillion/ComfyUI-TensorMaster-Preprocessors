import unittest

import numpy as np
import torch

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("face_eraser", Path(__file__).resolve().parents[1] / "face_eraser.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
FaceHairEraser, gauss_fill, prepare_mask = module.FaceHairEraser, module.gauss_fill, module.prepare_mask


class FillTests(unittest.TestCase):
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
