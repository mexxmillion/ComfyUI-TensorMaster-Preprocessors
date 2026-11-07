import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('tm_reference_test', ROOT / '__init__.py',
                                             submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
manager = sys.modules['tm_reference_test.reference_manager.node']
Node = manager.TMH3ReferenceManager


class ReferenceManagerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.inputs = self.root / 'input'
        self.inputs.mkdir()
        for method, value in [('get_input_directory', self.inputs), ('get_temp_directory', self.root / 'temp')]:
            patcher = patch.object(manager.folder_paths, method, return_value=str(value))
            patcher.start()
            self.addCleanup(patcher.stop)

    def image(self, name='image.png', color='red'):
        Image.new('RGB', (20, 10), color).save(self.inputs / name)
        return {'kind': 'image', 'file': name}

    def run_node(self, entries=(), **kwargs):
        return Node.execute(references_json=json.dumps({'references': list(entries)}), **kwargs)

    def test_connected_prompt_respects_enhancement_switch(self):
        for connected in (None, '', '  connected <Picture 1>\n'):
            for enhance in (False, True):
                with self.subTest(connected=connected, enhance=enhance):
                    with patch.object(manager.writer, 'write_prompt', return_value='enhanced') as writer:
                        result = self.run_node(direction='typed', prompt=connected, ai_enhance=enhance)
                    source = 'typed' if connected is None else connected
                    self.assertEqual(result[18], 'enhanced' if enhance else source)
                    self.assertEqual(writer.call_count, int(enhance))
                    if enhance:
                        self.assertEqual(writer.call_args.kwargs['direction'], source)

    def test_crop_preprocessors_preview_and_writer_use_same_image(self):
        entry = self.image()
        entry.update(crop=[0.5, 0, 0.5, 1], preprocess={'depth': True})
        depth = torch.full((1, 10, 10, 3), 0.4)
        with patch.object(manager.TMDepthReference, 'depth', return_value=(depth,)) as infer, \
             patch.object(manager.writer, 'write_prompt', return_value='done') as writer:
            result = self.run_node([entry], ai_enhance=True)
        self.assertEqual(tuple(infer.call_args.args[0].shape), (1, 10, 10, 3))
        self.assertTrue(torch.equal(result[0], depth))
        rendered = Path(writer.call_args.kwargs['references'].references[0].file)
        self.assertTrue(rendered.is_file())
        np.testing.assert_array_equal(np.array(Image.open(rendered)), np.full((10, 10, 3), 102))
        self.assertEqual(result.ui['tm_reference_previews'][0]['source'], entry)
        self.assertEqual(json.loads(result[19])[0]['file'], 'image.png')

    def test_real_writer_payload_uses_connected_prompt(self):
        with patch.object(manager.writer.requests, 'post') as post:
            post.return_value.status_code = 200
            post.return_value.json.return_value = {'choices': [{'message': {'content': 'enhanced'}}]}
            result = self.run_node([self.image()], direction='ignore this', prompt='connected direction',
                                   ai_enhance=True, prompt_provider='local', model='local-vision',
                                   width=864, height=480, length_seconds=6.5)
        self.assertEqual(result[18], 'enhanced')
        self.assertEqual(post.call_count, 1)
        payload = post.call_args.kwargs['json']
        self.assertEqual(payload['model'], 'local-vision')
        content = payload['messages'][1]['content']
        self.assertIn('connected direction', content[0]['text'])
        self.assertIn('frame: 864 x 480 (aspect ratio 9:5)', content[0]['text'])
        self.assertIn('duration: 6.500 seconds', content[0]['text'])
        self.assertNotIn('ignore this', content[0]['text'])
        self.assertEqual(content[-1]['type'], 'image_url')

    def test_target_dimensions_are_advice_until_resize_is_enabled(self):
        entry = self.image()
        untouched = self.run_node([entry], width=13, height=7, length_seconds=4, direction=' unchanged ')
        self.assertEqual(tuple(untouched[0].shape), (1, 10, 20, 3))
        self.assertEqual(untouched[18], ' unchanged ')
        resized = self.run_node([entry], width=13, height=7, resize_references=True)
        self.assertEqual(tuple(resized[0].shape), (1, 7, 13, 3))
        preview = resized.ui['tm_reference_previews'][0]
        with Image.open(self.root / 'temp' / preview['subfolder'] / preview['filename']) as stored:
            self.assertEqual(stored.size, (13, 7))

    def test_resizing_reduces_preprocessor_input_and_restores_cropped_output(self):
        entry = self.image()
        entry['preprocess'] = {'isolate': True, 'crop_subject': True}
        rgba = torch.ones((1, 7, 13, 4))
        with patch.object(manager.TMIsolateReference, 'isolate', return_value=(rgba, rgba[..., 3])) as isolate, \
             patch.object(manager.TMCompositeReference, 'composite', return_value=(torch.ones((1, 3, 5, 3)),)):
            result = self.run_node([entry], width=13, height=7, resize_references=True)
        self.assertEqual(tuple(isolate.call_args.args[0].shape), (1, 7, 13, 3))
        self.assertEqual(tuple(result[0].shape), (1, 7, 13, 3))

    def test_video_resize_preserves_frame_count_and_soundtrack(self):
        path = self.inputs / 'video.mp4'
        path.touch()
        audio = {'sample_rate': 24000, 'waveform': torch.ones((1, 1, 24000))}
        with patch.object(manager.media, 'load_video', return_value=(torch.rand((5, 4, 6, 3)), audio)):
            result = self.run_node([{'kind': 'video', 'file': 'video.mp4'}], width=13, height=7, resize_references=True)
        self.assertEqual(tuple(result[9].shape), (5, 7, 13, 3))
        self.assertIs(result[12], audio)

    def test_resizing_requires_a_real_target_for_visual_references(self):
        with self.assertRaisesRegex(ValueError, 'target width and height'):
            self.run_node([self.image()], resize_references=True)
        self.assertEqual(len(self.run_node(resize_references=True).result), 20)

    def test_isolation_and_depth_share_foreground_mask(self):
        entry = self.image()
        entry['preprocess'] = {'isolate': True, 'depth': True, 'background': 'white'}
        mask = torch.zeros((1, 10, 20))
        mask[:, :, :10] = 1
        rgba = torch.cat((torch.zeros((1, 10, 20, 3)), mask.unsqueeze(-1)), -1)
        with patch.object(manager.TMIsolateReference, 'isolate', return_value=(rgba, mask)), \
             patch.object(manager.TMDepthReference, 'depth', return_value=(torch.full((1, 10, 20, 3), 0.25),)):
            result = self.run_node([entry])
        self.assertTrue((result[0][:, :, :10] == 0.25).all())
        self.assertTrue((result[0][:, :, 10:] == 1).all())

    def test_face_cover_detects_rgb_and_overlays_solid_color_after_depth(self):
        entry = self.image()
        entry['preprocess'] = {'face_cover': True, 'depth': True, 'face_color': '#808080'}
        mask = torch.zeros((1, 10, 20))
        mask[:, 2:6, 4:8] = 1
        with patch.object(manager, 'face_cover_mask', return_value=mask) as detect, \
             patch.object(manager.TMDepthReference, 'depth', return_value=(torch.full((1, 10, 20, 3), 0.25),)):
            result = self.run_node([entry])
        self.assertTrue((detect.call_args.args[0][..., 0] == 1).all())
        self.assertTrue((detect.call_args.args[0][..., 1:] == 0).all())
        self.assertTrue((result[0][mask > 0] == 128 / 255).all())
        self.assertTrue((result[0][mask == 0] == 0.25).all())

    def test_input_paths_cannot_escape_input_directory(self):
        Image.new('RGB', (2, 2)).save(self.root / 'outside.png')
        for name in ('../outside.png', str(self.root / 'outside.png')):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'inside the input directory'):
                self.run_node([{'kind': 'image', 'file': name}])

    def test_replacing_file_invalidates_execution_cache(self):
        entry = self.image()
        raw = json.dumps([entry])
        before = Node.fingerprint_inputs(references_json=raw)
        self.image(color='blue')
        self.assertNotEqual(before, Node.fingerprint_inputs(references_json=raw))

    def test_video_trim_resamples_to_24fps_and_omits_absent_soundtrack(self):
        import av
        path = self.inputs / 'clip.mp4'
        with av.open(str(path), 'w') as container:
            stream = container.add_stream('mpeg4', rate=12)
            stream.width, stream.height = 32, 16
            stream.pix_fmt = 'yuv420p'
            for i in range(24):
                frame = av.VideoFrame.from_ndarray(np.full((16, 32, 3), i * 10, dtype=np.uint8), format='rgb24')
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        result = self.run_node([{'kind': 'video', 'file': 'clip.mp4', 'trim': [0.5, 1.5], 'crop': [0, 0, 0.5, 1]}])
        self.assertEqual(tuple(result[9].shape), (24, 16, 16, 3))
        self.assertIsNone(result[12])
        self.assertNotIn('audio_tag', json.loads(result[19])[0])


if __name__ == '__main__':
    unittest.main()
