import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import call, patch


spec = importlib.util.spec_from_file_location('tm_lora_test', Path(__file__).resolve().parents[1] / 'lora_loader.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MultiLoraTests(unittest.TestCase):
    def test_order_strengths_and_disabled_entries(self):
        entries = [
            dict(name='a.safetensors', strength_model=-0.5, strength_clip=0.2),
            dict(name='missing.safetensors', enabled=False),
            dict(name='zero.safetensors', strength_model=0, strength_clip=0),
            dict(name='b.safetensors', strength_model=1.5, strength_clip=0),
        ]
        with patch.object(module.folder_paths, 'get_filename_list', return_value=['a.safetensors', 'b.safetensors']), \
             patch.object(module.folder_paths, 'get_full_path_or_raise'), patch.object(module, 'LoraLoader') as loader:
            loader.return_value.load_lora.side_effect = [('model1', 'clip1'), ('model2', 'clip2')]
            output = module.TMMultiLora.execute('model', json.dumps(entries), 'clip')
            self.assertEqual(output.result, ('model2', 'clip2'))
            self.assertEqual(loader.return_value.load_lora.call_args_list, [
                call('model', 'clip', 'a.safetensors', -0.5, 0.2),
                call('model1', 'clip1', 'b.safetensors', 1.5, 0),
            ])

    def test_model_only_and_empty_stack(self):
        with patch.object(module.folder_paths, 'get_filename_list', return_value=['a']), \
             patch.object(module.folder_paths, 'get_full_path_or_raise'), patch.object(module, 'LoraLoader') as loader:
            loader.return_value.load_lora.return_value = ('patched', None)
            self.assertEqual(module.TMMultiLora.execute('model').result, ('model', None))
            module.TMMultiLora.execute('model', '[{"name":"a"}]')
            loader.return_value.load_lora.assert_called_once_with('model', None, 'a', 1, 0)

    def test_unlisted_paths_are_rejected_before_loading(self):
        for name in ('../outside.safetensors', 'C:\\outside.safetensors', 'missing.safetensors'):
            with self.subTest(name=name), patch.object(module.folder_paths, 'get_filename_list', return_value=['ok']), \
                 patch.object(module, 'LoraLoader') as loader:
                with self.assertRaisesRegex(ValueError, 'LoRA not found'):
                    module.TMMultiLora.execute('model', json.dumps([dict(name=name)]))
                loader.assert_not_called()

    def test_invalid_data_has_clear_errors(self):
        for raw in ('{}', '[null]', '[{"name":"a","strength_model":"bad"}]',
                    '[{"name":"a","strength_clip":NaN}]', '[{"name":"a","enabled":"false"}]'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                module.parse_loras(raw)


if __name__ == '__main__':
    unittest.main()
