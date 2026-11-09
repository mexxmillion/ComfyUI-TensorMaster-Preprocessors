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
elements.find(element => element.tag === 'button' && element.textContent === 'Save reference').onclick();
const saved = JSON.parse(context.testNode.widgets[0].value).references[0];
assert.ok(saved.preprocess.depth && saved.preprocess.face_cover);
assert.equal(saved.preprocess.face_size, 1.5);
saved.crop.forEach((value, index) => assert.ok(Math.abs(value - [0.1, 0.1, 0.7, 0.6][index]) < 1e-9));
assert.equal(context.testNode._tmDialogs.size, 0);
console.log('Image crop and preparation share one page; dragging and saving preserve depth and face-cover settings.');
