import json

from comfy_api.latest import io


DEFAULT_STATES = '[{"id":0,"label":"State 0"},{"id":1,"label":"State 1"},{"id":2,"label":"State 2"}]'


def parse_states(raw):
    states = json.loads(raw)
    if not isinstance(states, list) or not states:
        raise ValueError('The state menu needs at least one state.')
    ids, labels = set(), set()
    for state in states:
        if not isinstance(state, dict):
            raise ValueError('Each state needs an id and label.')
        state_id, label = state.get('id'), state.get('label')
        if type(state_id) is not int or not 0 <= state_id < 100 or state_id in ids:
            raise ValueError('State ids must be unique integers from 0 to 99.')
        if not isinstance(label, str) or not label.strip() or label in labels:
            raise ValueError('State labels must be nonempty and unique.')
        ids.add(state_id)
        labels.add(label)
    return states


class TMStateMenu(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='TMStateMenu', display_name='TM · State Menu', category='Tensor Master/Logic',
            description='Name your modes and connect this state to multiple State Switch nodes to change them together.',
            inputs=[
                io.String.Input('selected', default='State 0', dynamic_prompts=False,
                                extra_dict={'widgetType': 'TM_STATE_MENU'}),
                io.String.Input('states_json', default=DEFAULT_STATES, multiline=True, dynamic_prompts=False),
            ],
            outputs=[io.Int.Output(display_name='state')],
        )

    @classmethod
    def execute(cls, selected='State 0', states_json=DEFAULT_STATES):
        for state in parse_states(states_json):
            if state['label'] == selected:
                return io.NodeOutput(state['id'])
        raise ValueError(f'Selected state is not in the menu: {selected}')


class TMStateSwitch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        match = io.MatchType.Template('state_value')
        return io.Schema(
            node_id='TMStateSwitch', display_name='TM · State Switch', category='Tensor Master/Logic',
            description='Route state0, state1, etc. Only the selected connected branch is evaluated. Connect one State Menu to several switches to synchronize them.',
            inputs=[
                io.Int.Input('state', default=0, min=0, max=99),
                io.Autogrow.Input('values', template=io.Autogrow.TemplatePrefix(
                    io.MatchType.Input('value', template=match, lazy=True, optional=True),
                    prefix='state', min=1, max=100)),
            ],
            outputs=[io.MatchType.Output(template=match, display_name='value')],
        )

    @classmethod
    def check_lazy_status(cls, state, values):
        key = f'state{state}'
        if key not in values:
            raise ValueError(f'State {state} is not connected. Connect values.{key}.')
        value, input_name = values[key]
        return [input_name] if value is None else []

    @classmethod
    def execute(cls, state, values):
        key = f'state{state}'
        if key not in values:
            raise ValueError(f'State {state} is not connected. Connect values.{key}.')
        return io.NodeOutput(values[key])
