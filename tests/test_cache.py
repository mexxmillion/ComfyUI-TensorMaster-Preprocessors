import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('tm_preprocessors', ROOT / '__init__.py', submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from tm_preprocessors.prepare import TMPrepareReference, TMDepthReference
from tm_preprocessors.nodes import TMIsolateReference


class CacheTests(unittest.TestCase):
    def test_staged_isolation_and_depth_skip_both_models_and_produce_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input/h3_mobile'
            cache = source / 'preprocessed-cache'
            cache.mkdir(parents=True)
            Image.new('RGB', (20, 20), 'red').save(source / 'source.png')
            rgba = Image.new('RGBA', (20, 20), (0, 0, 0, 0))
            for y in range(5, 15):
                for x in range(5, 15):
                    rgba.putpixel((x, y), (255, 0, 0, 255))
            rgba.save(cache / ('a' * 64 + '-rgba.png'))
            Image.new('RGB', (20, 20), (80, 80, 80)).save(cache / ('b' * 64 + '-depth.png'))
            paths = SimpleNamespace(get_input_directory=lambda: str(root / 'input'),
                                    get_output_directory=lambda: str(root / 'output'))
            with patch.dict(sys.modules, {'folder_paths': paths}), \
                 patch.object(TMIsolateReference, 'isolate', side_effect=AssertionError('isolation model should not load')), \
                 patch.object(TMDepthReference, 'depth', side_effect=AssertionError('depth model should not load')):
                for color, value in [('white', 1.0), ('gray', 127 / 255), ('black', 0.0)]:
                    result = TMPrepareReference().prepare('h3_mobile/source.png', depth=True, background=color,
                        crop=False, cache_key='c' * 64, isolation_key='a' * 64, depth_key='b' * 64,
                        cached_isolation='h3_mobile/preprocessed-cache/' + 'a' * 64 + '-rgba.png',
                        cached_depth='h3_mobile/preprocessed-cache/' + 'b' * 64 + '-depth.png')
                    rgb, full, mask = result['result']
                    self.assertEqual(tuple(rgb.shape), (1, 20, 20, 3))
                    self.assertTrue(torch.allclose(rgb[0, 0, 0], torch.tensor([value] * 3)))
                    self.assertTrue(torch.allclose(rgb[0, 10, 10], torch.tensor([80 / 255] * 3)))
                    self.assertEqual(mask[0, 0, 0].item(), 0)
                    self.assertEqual(tuple(full.shape), (1, 20, 20, 4))
                    self.assertEqual(len(result['ui']['reference_cache']), 1)

    def test_local_disk_cache_skips_inference_on_second_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input/h3_mobile'
            source.mkdir(parents=True)
            Image.new('RGB', (20, 20), 'red').save(source / 'source.png')
            paths = SimpleNamespace(get_input_directory=lambda: str(root / 'input'),
                                    get_output_directory=lambda: str(root / 'output'))
            rgba = torch.ones((1, 20, 20, 4))
            with patch.dict(sys.modules, {'folder_paths': paths}), patch.object(TMIsolateReference, 'isolate', return_value=(rgba, rgba[..., 3])) as inference:
                arguments = dict(source_file='h3_mobile/source.png', cache_key='c' * 64,
                                 isolation_key='a' * 64, depth_key='b' * 64)
                first = TMPrepareReference().prepare(**arguments)['result'][0]
                second = TMPrepareReference().prepare(**arguments)['result'][0]
                self.assertEqual(inference.call_count, 1)
                self.assertTrue(torch.equal(first, second))


if __name__ == '__main__':
    unittest.main()
