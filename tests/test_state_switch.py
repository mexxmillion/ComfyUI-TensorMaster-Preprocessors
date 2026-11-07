import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import execution
import nodes
from comfy_api.latest import io


spec = importlib.util.spec_from_file_location('tm_state_test', Path(__file__).resolve().parents[1] / 'state_switch.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
STATES = json.dumps([{'id': 0, 'label': 'Turbo 4-step'}, {'id': 1, 'label': 'Turbo 8-step'}, {'id': 2, 'label': 'Flex 6-step'}])


class StateTests(unittest.TestCase):
    def test_menu_uses_stable_ids_after_rename_and_removal(self):
        self.assertEqual(module.TMStateMenu.execute('Flex 6-step', STATES).result, (2,))
        self.assertEqual(module.TMStateMenu.execute('Quality', '[{"id":2,"label":"Quality"}]').result, (2,))
        with self.assertRaisesRegex(ValueError, 'not in the menu'):
            module.TMStateMenu.execute('missing', STATES)

    def test_invalid_menu(self):
        for raw in ('[]', '{}', '[null]', '[{"id":true,"label":"a"}]',
                    '[{"id":0,"label":"a"},{"id":0,"label":"b"}]',
                    '[{"id":0,"label":"a"},{"id":1,"label":"a"}]'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                module.parse_states(raw)

    def test_missing_selected_socket(self):
        with self.assertRaisesRegex(ValueError, 'State 2 is not connected'):
            module.TMStateSwitch.check_lazy_status(2, {'state0': (None, 'values.state0')})


class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_menu_synchronizes_switches_and_only_runs_selected_branches(self):
        executed = []

        class Source(io.ComfyNode):
            @classmethod
            def define_schema(cls):
                return io.Schema(node_id='TMTestSource', inputs=[io.Int.Input('value')], outputs=[io.Int.Output()])

            @classmethod
            def execute(cls, value):
                executed.append(value)
                return io.NodeOutput(value)

        class Sink(io.ComfyNode):
            @classmethod
            def define_schema(cls):
                return io.Schema(node_id='TMTestSink', inputs=[io.Int.Input('steps'), io.Int.Input('checkpoint')],
                                 outputs=[], is_output_node=True)

            @classmethod
            def execute(cls, steps, checkpoint):
                return io.NodeOutput(ui={'selected': [steps, checkpoint]})

        mapping = {'TMStateMenu': module.TMStateMenu, 'TMStateSwitch': module.TMStateSwitch,
                   'TMTestSource': Source, 'TMTestSink': Sink}
        with patch.dict(nodes.NODE_CLASS_MAPPINGS, mapping):
            for label, expected_steps, expected_checkpoint in [('Turbo 4-step', 4, 100), ('Turbo 8-step', 8, 101), ('Flex 6-step', 6, 102)]:
                executed.clear()
                prompt = {'mode': {'class_type': 'TMStateMenu', 'inputs': {'selected': label, 'states_json': STATES}}}
                for branch, numbers in [('steps', [4, 8, 6]), ('checkpoint', [100, 101, 102])]:
                    inputs = {'state': ['mode', 0]}
                    for index, number in enumerate(numbers):
                        source = f'{branch}{index}'
                        prompt[source] = {'class_type': 'TMTestSource', 'inputs': {'value': number}}
                        inputs[f'values.state{index}'] = [source, 0]
                    prompt[branch] = {'class_type': 'TMStateSwitch', 'inputs': inputs}
                prompt['sink'] = {'class_type': 'TMTestSink', 'inputs': {'steps': ['steps', 0], 'checkpoint': ['checkpoint', 0]}}
                valid, _, outputs, errors = await execution.validate_prompt('state-test', prompt, None)
                self.assertTrue(valid, errors)
                server = SimpleNamespace(client_id=None, send_sync=Mock())
                executor = execution.PromptExecutor(server, cache_type=execution.CacheType.NONE,
                                                    cache_args={'ram': 0, 'ram_inactive': 0})
                await executor.execute_async(prompt, 'state-test', execute_outputs=outputs)
                self.assertTrue(executor.success, executor.status_messages)
                self.assertEqual(sorted(executed), sorted([expected_steps, expected_checkpoint]))
                self.assertEqual(executor.history_result['outputs']['sink']['selected'], [expected_steps, expected_checkpoint])


if __name__ == '__main__':
    unittest.main()
