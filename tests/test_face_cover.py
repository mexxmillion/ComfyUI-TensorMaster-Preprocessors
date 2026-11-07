import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

spec = importlib.util.spec_from_file_location('face_cover_test', Path(__file__).resolve().parents[1] / 'face_cover.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FaceCoverTests(unittest.TestCase):
    def detector(self, boxes):
        detector = Mock()
        detector.empty.return_value = False
        detector.detectMultiScale.return_value = boxes
        return detector

    def test_solid_fill_preserves_uncovered_pixels_alpha_and_dtype(self):
        image = torch.rand((2, 64, 80, 4), dtype=torch.float64)
        detector = self.detector(())
        detector.detectMultiScale.side_effect = [np.array([[20, 10, 20, 30]]), ()]
        with patch.object(module.cv2, 'CascadeClassifier', return_value=detector):
            result = module.TMFaceCover.execute(image, detector='frontal', color='#804020', neighbors=8, min_face_size=16)
        covered, mask = result.result
        self.assertEqual(covered.dtype, image.dtype)
        self.assertEqual(covered.device, image.device)
        self.assertEqual(tuple(mask.shape), (2, 64, 80))
        self.assertEqual(set(mask.unique().tolist()), {0.0, 1.0})
        expected = torch.tensor([128, 64, 32], dtype=image.dtype) / 255
        self.assertTrue(torch.equal(covered[0, 25, 30, :3], expected))
        self.assertTrue(torch.equal(covered[..., 3], image[..., 3]))
        self.assertTrue(torch.equal(covered[..., :3][mask == 0], image[..., :3][mask == 0]))
        self.assertTrue(torch.equal(covered[1], image[1]))
        self.assertEqual(detector.detectMultiScale.call_args.kwargs['minNeighbors'], 8)
        self.assertEqual(detector.detectMultiScale.call_args.kwargs['minSize'], (16, 16))

    def test_circle_and_ellipse_have_different_coverage(self):
        image = torch.zeros((1, 80, 80, 3))
        with patch.object(module.cv2, 'CascadeClassifier', return_value=self.detector([[30, 20, 20, 40]])):
            ellipse = module.face_cover_mask(image, detector='frontal', shape='ellipse', scale_x=1, scale_y=1)
            circle = module.face_cover_mask(image, detector='frontal', shape='circle', scale_x=1, scale_y=1)
        self.assertEqual(ellipse[0, 40, 55], 0)
        self.assertEqual(circle[0, 40, 55], 1)
        self.assertGreater(circle.sum(), ellipse.sum())

    def test_size_multiplier_expands_cover_into_hair_region(self):
        image = torch.zeros((1, 100, 100, 3))
        for shape in ('ellipse', 'circle'):
            with self.subTest(shape=shape), patch.object(module.cv2, 'CascadeClassifier', return_value=self.detector([[40, 40, 20, 20]])):
                original = module.TMFaceCover.execute(image, detector='frontal', shape=shape, scale_x=1, scale_y=1).result[1]
                enlarged = module.TMFaceCover.execute(image, detector='frontal', shape=shape, scale_x=1, scale_y=1, size=2).result[1]
            self.assertEqual(original[0, 33, 50], 0)
            self.assertEqual(enlarged[0, 33, 50], 1)
            self.assertTrue((enlarged >= original).all())

    def test_flipped_profile_detection_maps_back_to_original(self):
        frontal, profile = self.detector(()), self.detector(())
        profile.detectMultiScale.side_effect = [(), [[5, 10, 12, 16]]]
        with patch.object(module.cv2, 'CascadeClassifier', side_effect=[frontal, profile]):
            mask = module.face_cover_mask(torch.zeros((1, 50, 80, 3)), detector='frontal + profile', scale_x=1, scale_y=1)
        self.assertEqual(mask[0, 18, 69], 1)
        self.assertEqual(mask[0, 18, 11], 0)

    def test_cover_at_frame_edge_is_clipped_without_feathering(self):
        image = torch.zeros((1, 20, 20, 3), dtype=torch.float16)
        with patch.object(module.cv2, 'CascadeClassifier', return_value=self.detector([[0, 0, 10, 10]])):
            covered, mask = module.TMFaceCover.execute(image, detector='frontal', scale_x=3, scale_y=3).result
        self.assertEqual(mask[0, 0, 0], 1)
        self.assertEqual(covered.dtype, torch.float16)
        self.assertTrue(torch.isfinite(covered).all())

    def test_color_rejects_invalid_hex(self):
        with self.assertRaisesRegex(ValueError, 'hex color'):
            module.apply_face_cover(torch.zeros((1, 2, 2, 3)), torch.ones((1, 2, 2)), 'gray')

    def test_retinaface_confidence_count_and_start_index_are_per_image(self):
        backend = Mock()
        backend.detect_faces.return_value = np.array([
            [60, 20, 80, 40, .99], [10, 20, 30, 40, .90], [35, 20, 55, 40, .40],
        ])
        image = torch.zeros((2, 80, 100, 3))
        with patch.object(module, 'load_retinaface', return_value=backend) as load:
            mask = module.face_cover_mask(image, min_face_size=8, confidence=.6, number_of_faces=1,
                                          start_index=1, order='left to right', scale_x=1, scale_y=1)
        load.assert_called_once_with(.6, 50)
        self.assertEqual(backend.detect_faces.call_count, 2)
        self.assertTrue((mask[:, 30, 70] == 1).all())
        self.assertTrue((mask[:, 30, 20] == 0).all())
        self.assertTrue((mask[:, 30, 45] == 0).all())

    def test_selection_orders_and_out_of_range_start(self):
        faces = [(40, 0, 10, 10, .99), (10, 0, 20, 20, .8), (70, 0, 30, 30, .9)]
        self.assertEqual(module.select_faces(faces, 1, 0, 50, 'confidence'), [faces[0]])
        self.assertEqual(module.select_faces(faces, 1, 0, 50, 'largest first'), [faces[2]])
        self.assertEqual(module.select_faces(faces, 0, 1, 50, 'left to right'), [faces[0], faces[2]])
        self.assertEqual(module.select_faces(faces, 1, 9, 50, 'confidence'), [])

    def test_vertical_shift_moves_cover_toward_hair(self):
        with patch.object(module.cv2, 'CascadeClassifier', return_value=self.detector([[40, 40, 20, 20]])):
            centered = module.face_cover_mask(torch.zeros((1, 100, 100, 3)), detector='frontal', scale_x=1, scale_y=1)
            raised = module.face_cover_mask(torch.zeros((1, 100, 100, 3)), detector='frontal', scale_x=1,
                                            scale_y=1, shift_factor=1)
        self.assertEqual(centered[0, 32, 50], 0)
        self.assertEqual(raised[0, 32, 50], 1)


if __name__ == '__main__':
    unittest.main()
