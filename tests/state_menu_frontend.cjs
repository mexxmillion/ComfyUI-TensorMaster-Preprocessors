const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');

class Element {
    constructor(tag) { this.tag = tag; this.children = []; this.style = {}; this.listeners = {}; }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    setAttribute() {}
    addEventListener(name, callback) { this.listeners[name] = callback; }
    showModal() {}
    close() { this.listeners.close?.(); }
    remove() {}
}

let extension;
const document = { createElement: tag => new Element(tag), body: new Element('body'),
    head: new Element('head'), getElementById: () => null };
const context = { document, URL, console, window: { comfyAPI: { widgetInputs: {
    setWidgetConfig(input, config) { input.config = config; },
} } }, app: { registerExtension(value) { extension = value; }, graph: {
    beforeChange() {}, afterChange() {}, setDirtyCanvas() {},
} } };
const source = fs.readFileSync(path.join(__dirname, '../web/state_menu.js'), 'utf8')
    .replace(/^import .*;$/gm, '').replace(/export (const|function)/g, '$1')
    .replaceAll('import.meta.url', JSON.stringify('http://localhost/state_menu.js'));
vm.createContext(context);
vm.runInContext(source, context);

class Node {
    constructor() { this.widgets = []; }
    addWidget(type, name, value, callback, options = {}) {
        const widget = { type, name, value, callback, options };
        this.widgets.push(widget);
        return widget;
    }
    setSize() {}
}

(async () => {
    await extension.beforeRegisterNodeDef(Node, { name: 'TMStateMenu' });
    const node = new Node();
    const menu = extension.getCustomWidgets().TM_STATE_MENU(node, 'selected', ['STRING', { default: 'State 0' }]).widget;
    const storage = node.addWidget('text', 'states_json', '[{"id":0,"label":"State 0"},{"id":2,"label":"State 2"}]');
    node.onNodeCreated();
    const input = { widget: { name: 'selected' } };
    node.onInputAdded(input);
    assert.equal(input.config[0], 'COMBO');
    assert.deepEqual(Array.from(input.config[1].options()), ['State 0', 'State 2']);

    menu.value = 'State 2';
    node.widgets.find(widget => widget.name === 'Edit states…').callback();
    const dialog = document.body.children.at(-1);
    const rows = dialog.children.find(child => child.className === 'tm-state-rows');
    const name = rows.children[1].children[1];
    name.value = 'Flex 6-step'; name.oninput();
    dialog.children.at(-1).children[1].onclick();
    assert.equal(menu.value, 'Flex 6-step');
    assert.equal(JSON.parse(storage.value)[1].id, 2);
    assert.deepEqual(Array.from(menu.options.values()), ['State 0', 'Flex 6-step']);
    assert.deepEqual(Array.from(input.config[1].options()), ['State 0', 'Flex 6-step']);

    // A restored workflow must supply its saved names to the promoted dropdown.
    storage.value = '[{"id":7,"label":"Restored mode"}]';
    menu.value = 'Restored mode';
    assert.deepEqual(Array.from(input.config[1].options()), ['Restored mode']);
    assert.equal(vm.runInContext('nextStateId([{id:0},{id:2}])', context), 1);
    assert.throws(() => vm.runInContext('readStates(\'[{"id":0,"label":"a"},{"id":1,"label":"a"}]\')', context));
    console.log('State menu editing, stable ids, restored names, and promoted combo config passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
