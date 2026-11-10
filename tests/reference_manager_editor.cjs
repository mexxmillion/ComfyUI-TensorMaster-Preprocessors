const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');

class Element {
    constructor(tag) {
        this.tag = tag; this.children = []; this.style = {}; this.dataset = {}; this.attributes = {};
        this.listeners = {}; this.clientWidth = 400; this.clientHeight = 600;
        this.naturalWidth = 400; this.naturalHeight = 600;
        const classes = new Set();
        this.classList = {
            add: (...names) => names.forEach(name => classes.add(name)),
            remove: (...names) => names.forEach(name => classes.delete(name)),
            contains: name => classes.has(name),
            toggle: (name, enabled) => enabled ? classes.add(name) : classes.delete(name),
        };
    }
    append(...children) { children.forEach(child => { child.parentElement = this; this.children.push(child); }); }
    setAttribute(name, value) { this.attributes[name] = value; }
    addEventListener(name, listener) { (this.listeners[name] ??= []).push(listener); }
    getBoundingClientRect() { return { left: 0, top: 0, width: 400, height: 600 }; }
    setPointerCapture() {}
    querySelectorAll() { return []; }
    showModal() { this.isConnected = true; }
    close() { this.listeners.close?.forEach(listener => listener()); }
    remove() { this.isConnected = false; }
}

const document = { createElement: tag => new Element(tag), body: new Element('body') };
const context = { document, URLSearchParams, app: { registerExtension() {}, graph: { change() {}, setDirtyCanvas() {} } },
    api: { apiURL: value => value }, ResizeObserver: class { observe() {} disconnect() {} } };
const source = fs.readFileSync(path.join(__dirname, '../web/reference_manager.js'), 'utf8')
    .replace(/^import .*;$/gm, '').replace(/export function/g, 'function')
    .replaceAll('import.meta.url', JSON.stringify('http://localhost/reference_manager.js'));
vm.createContext(context);
vm.runInContext(source, context);
vm.runInContext(`
const node = {
    widgets:[{name:'references_json',value:''}],
    _tmRefs:[{kind:'image',file:'new.png',crop:[0,0,.5,1],preprocess:{depth:true,face_cover:true,face_size:1.5}}],
    _tmPreviews:new Map(), _tmSelected:new Set(), _tmSurfaces:new Set(), _tmDialogs:new Set(),
};
editReference(node,0);
globalThis.testNode = node;
`, context);

const modal = document.body.children[0];
const walk = element => [element, ...element.children.flatMap(walk)];
const elements = walk(modal);
assert.equal(elements.filter(element => element.attributes.role === 'tab').length, 0);
const panels = elements.filter(element => element.className === 'tmrm-tool-panel');
assert.equal(panels.length, 2);
assert.ok(panels.every(panel => !panel.hidden), 'crop and preparation must be visible together');
const stage = elements.find(element => element.className === 'tmrm-media-stage');
assert.ok(stage.classList.contains('is-cropping'));
const media = stage.children.find(element => element.tag === 'img');
media.listeners.load.forEach(listener => listener());
stage.onpointerdown({ clientX: 40, clientY: 60, target: media, pointerId: 1, preventDefault() {} });
stage.onpointermove({ clientX: 320, clientY: 420 });
stage.onpointerup();
const postAi = elements.filter(element => element.tag === 'label' && element.children.some(child => child.className === 'tmrm-switch-copy' && child.children[0]?.textContent === 'Post AI'));
assert.equal(postAi.length, 2, 'depth and face cover need separate Post AI controls');
postAi[0].children[0].checked = true;
postAi[1].children[0].checked = false;
elements.find(element => element.tag === 'button' && element.textContent === 'Save reference').onclick();
const saved = JSON.parse(context.testNode.widgets[0].value).references[0];
assert.ok(saved.preprocess.depth && saved.preprocess.face_cover);
assert.equal(saved.preprocess.face_size, 1.5);
assert.equal(saved.preprocess.depth_post_ai, true);
assert.equal(saved.preprocess.face_cover_post_ai, false);
saved.crop.forEach((value, index) => assert.ok(Math.abs(value - [0.1, 0.1, 0.7, 0.6][index]) < 1e-9));
assert.equal(context.testNode._tmDialogs.size, 0);
console.log('Image crop and preparation share one page; dragging and saving preserve depth and face-cover settings.');

vm.runInContext(`
const settingsNode = {
    widgets: Object.entries({prompt_provider:'Local Gemma 4 12B (Director)', model:'existing-model', api_base:'http://localhost:1234/v1',
        api_key:'', reasoning_effort:'medium', max_reference_edge:2048, system_prompt:''}).map(([name,value])=>({name,value})),
    _tmDialogs:new Set(), _tmPreviews:new Map(), _tmSurfaces:new Set(),
};
openSettings(settingsNode);
globalThis.settingsNode = settingsNode;
`, context);
const settings = document.body.children.at(-1);
const settingsElements = walk(settings);
const setting = label => settingsElements.find(element => element.tag === 'label' && element.children[0]?.textContent === label);
assert.ok(setting('Model ID').hidden && setting('Local API base').hidden && setting('API key').hidden && setting('Reasoning').hidden);
assert.ok(!setting('Image edge limit · 0 for original').hidden);
const provider = setting('Provider').children[1];
provider.value = 'openrouter'; provider.onchange();
assert.ok(!setting('API key').hidden);
provider.value = 'Local Gemma 4 12B (Director)'; provider.onchange();
settingsElements.find(element => element.tag === 'button' && element.textContent === 'Save settings').onclick();
assert.equal(context.settingsNode.widgets.find(widget => widget.name === 'prompt_provider').value, 'Local Gemma 4 12B (Director)');
console.log('Managed Gemma settings hide unused endpoint fields and save the selected provider.');
