const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../web/reference_manager.js'), 'utf8')
    .replace(/^import .*;$/gm, '')
    .replace(/export function/g, 'function')
    .replaceAll('import.meta.url', JSON.stringify('http://localhost/reference_manager.js'));
const context = { app: { registerExtension() {} }, api: { apiURL: value => value }, URLSearchParams };
vm.createContext(context);
vm.runInContext(source, context);
vm.runInContext(`
const refs = [
    {kind: 'image', file: 'a', crop: [0, 0, .5, 1], preprocess: {depth: true, face_cover:true, face_size:1.5}},
    {kind: 'image', file: 'b'},
    {kind: 'video', file: 'v', use_soundtrack: false},
    {kind: 'audio', file: 's'},
];
const replaced = replaceReference(refs, 0, 'new');
if (replaced[0].file !== 'new' || replaced[1] !== refs[1] || JSON.stringify(replaced[0].crop) !== JSON.stringify(refs[0].crop) || JSON.stringify(replaced[0].preprocess) !== JSON.stringify(refs[0].preprocess)) throw Error('image replacement lost its slot, crop, depth, or face-cover settings');
if (refs[0].file !== 'a' || previewKey(replaced[0]) === previewKey(refs[0])) throw Error('replacement mutated the source or retained its preview key');
const replacedVideo = replaceReference([{kind:'video',file:'old',trim:[1,2],crop:[0,0,.5,1],use_soundtrack:false}],0,'new');
if (replacedVideo[0].trim || replacedVideo[0].crop || replacedVideo[0].use_soundtrack !== false) throw Error('video replacement behavior changed');
if (referenceTags(refs).get(1) !== '<Picture 2>' || referenceTags(refs).get(3) !== '<Audio 1>') throw Error('incorrect tags');
if (!viewUrl('tm-references/file.png').includes('subfolder=tm-references')) throw Error('incorrect nested input URL');
if (parseReferences(JSON.stringify({references: refs})).length !== 4) throw Error('state roundtrip failed');
const reverseCrop = cropFromDrag([.9,.8],[.1,.2]);
if (reverseCrop.some((v,i) => Math.abs(v - [.1,.2,.8,.6][i]) > 1e-9)) throw Error('reverse crop drag failed');
const clippedCrop = cropFromDrag([-.4,.2],[1.3,.9]);
if (clippedCrop.some((v,i) => Math.abs(v - [0,.2,1,.7][i]) > 1e-9)) throw Error('crop bounds failed');
const c = {kind:'image',file:'c'}, d = {kind:'image',file:'d'};
const mixed = [refs[0],refs[2],refs[1],refs[3],c,d];
const group = new Set([refs[0],c]);
const grouped = moveImages(mixed,group,d,true);
if (grouped.map(r=>r.file).join(',') !== 'b,v,d,s,a,c') throw Error('group reorder or non-image placement failed');
if (grouped[4] !== refs[0] || grouped[4].preprocess.depth !== true || grouped[4].crop !== refs[0].crop) throw Error('reorder lost preprocessing or source identity');
const single = moveImages(mixed,new Set([d]),refs[0]);
if (single.map(r=>r.file).join(',') !== 'd,v,a,s,b,c') throw Error('single reorder failed');
if (referenceTags(grouped).get(4) !== '<Picture 3>') throw Error('reordered tags failed');
if (moveImages(mixed,group,refs[0]) !== mixed) throw Error('drop on selected image should be a no-op');
if (moveImages(mixed,group,null).map(r=>r.file).join(',') !== 'b,v,d,s,a,c') throw Error('append selected group failed');
`, context);
console.log('Reference replacement, tags, source URLs, state roundtrip, crop drags and image group reordering passed.');

// Remote HTTP pages expose getRandomValues but not randomUUID.
context.crypto = { getRandomValues: array => require('node:crypto').webcrypto.getRandomValues(array) };
context.FormData = FormData;
context.File = File;
const uploadedNames = [];
context.api.fetchApi = async (url, options) => {
    if (url !== '/upload/image') throw Error('unexpected upload URL');
    const file = options.body.get('image');
    if (!/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}-reference.png$/.test(file.name)) throw Error('invalid upload filename');
    if (options.body.get('subfolder') !== 'tm-references') throw Error('upload folder changed');
    uploadedNames.push(file.name);
    return { ok: true, json: async () => ({ subfolder: 'tm-references', name: file.name }) };
};
vm.runInContext(`(async () => {
    const file = new File(['image bytes'], 'reference.png', {type: 'image/png'});
    const first = await upload(file, 'image');
    const second = await upload(file, 'image');
    if (first.file === second.file || first.kind !== 'image') throw Error('uploads must have distinct paths');
    if (displayFilename(first.file) !== 'reference.png') throw Error('display filename changed');
})()`, context).then(() => {
    if (uploadedNames.length !== 2) throw Error('uploads did not complete');
    console.log('Uploads without crypto.randomUUID passed.');
}).catch(error => { console.error(error); process.exitCode = 1; });

const layoutContext = {
    app: { registerExtension(extension) { this.extension = extension; } },
    api: { addEventListener() {}, removeEventListener() {} }, URL, URLSearchParams,
    document: { getElementById: () => true },
};
vm.createContext(layoutContext);
vm.runInContext(source, layoutContext);
vm.runInContext(`
buildSurface = node => {
    const surface = {root:{querySelectorAll:()=>[]}};
    node._tmSurfaces.add(surface);
    return surface;
};
renderSurface = () => {};
openStudio = node => { node.openedStudio = true; };
class TestNode {
    constructor() {
        this.widgets = ['references_json','direction','ai_enhance','max_reference_edge','prompt_provider','model','api_base','api_key','reasoning_effort','system_prompt','resize_references']
            .map(name => ({name, value: name === 'references_json' ? '{"references":[{"kind":"image","file":"saved.png"}]}' : '', options:{}, element:{style:{}}}));
    }
    addWidget(type,name,value,callback,options) { this.widgets.push({type,name,value,callback,options}); }
    addDOMWidget(name,type,root,options) { this.panel = {name,type,root,options}; }
    setSize(size) { this.size = size; }
}
app.extension.beforeRegisterNodeDef(TestNode, {name:'TMH3ReferenceManager'});
const node = new TestNode();
const saved = node.widgets.map(widget => widget.value);
node.onNodeCreated();
for (const field of node.widgets.slice(0,saved.length)) {
    if (field.options.surfaces.vueNode !== 'never' || field.options.surfaces.canvas !== 'never' || field.options.surfaces.panel !== 'never') throw Error('internal field exposed on a rendering surface');
}
if (node.widgets_start_y !== 95 || node.panel.options.getHeight) throw Error('compact canvas offset or native panel sizing changed');
if (node.panel.options.getMinHeight() !== 450) throw Error('panel lost native minimum height');
const launch = node.widgets.find(widget => widget.name === 'Open Reference Studio');
if (!launch || launch.options.serialize !== false) throw Error('missing nonserialized studio button');
launch.callback();
if (!node.openedStudio) throw Error('studio button did not open the owner editor');
if (JSON.stringify(saved) !== JSON.stringify(node.widgets.slice(0,saved.length).map(widget => widget.value))) throw Error('layout changed serialized state');
node.onConfigure();
if (node._tmRefs[0].file !== 'saved.png') throw Error('restoring the node lost references');
node.onRemoved();
if (node._tmSurfaces.size) throw Error('removed node retained its surfaces');
`, layoutContext);
console.log('Nodes 2.0 visibility, native panel sizing, studio button, restore, and cleanup passed.');

const previewContext = { app: { registerExtension() {} }, api: {}, URLSearchParams };
vm.createContext(previewContext);
vm.runInContext(source, previewContext);
vm.runInContext(`(async () => {
    const node = { _tmSurfaces: new Set() };
    let rejected;
    const pending = { promptId: 'preview-id', reject: error => { rejected = error.message; } };
    node._tmPending = pending;
    api.fetchApi = async () => ({ ok: true, json: async () => ({ queue_running: [[0, 'preview-id']], queue_pending: [] }) });
    await checkPreviewQueue(node);
    if (node._tmPending !== pending) throw Error('running preview was unlocked');
    api.fetchApi = async () => ({ ok: true, json: async () => ({ queue_running: [], queue_pending: [[0, 'preview-id']] }) });
    await checkPreviewQueue(node);
    if (node._tmPending !== pending) throw Error('queued preview was unlocked');
    api.fetchApi = async () => ({ ok: true, json: async () => ({ queue_running: [[0, 'other-job']], queue_pending: [] }) });
    await checkPreviewQueue(node);
    if (node._tmPending || !rejected.includes('stopped')) throw Error('deleted preview stayed locked');
    node._tmPending = pending;
    api.fetchApi = async () => { throw Error('offline'); };
    await checkPreviewQueue(node);
    if (node._tmPending || !rejected.includes('disconnected')) throw Error('disconnect stayed locked');
    const calls = [];
    api.fetchApi = async (url, options) => { calls.push([url, JSON.parse(options.body)]); return { ok: true }; };
    await cancelPreview(pending);
    if (calls.length !== 2 || calls[0][0] !== '/queue' || calls[0][1].delete[0] !== 'preview-id' || calls[1][0] !== '/interrupt' || calls[1][1].prompt_id !== 'preview-id') throw Error('cancel must target only the preview');
    calls.length = 0;
    await cancelPreview({ promptId: null });
    if (calls.length) throw Error('cancel without ID must not interrupt globally');
    node._tmPending = { promptId: 'new-preview' };
    failPreview(node, pending, 'late event');
    if (node._tmPending.promptId !== 'new-preview') throw Error('late event cleared a newer preview');
})()`, previewContext).then(() => {
    console.log('Preview cancellation, queue deletion, disconnect, active queue protection and stale-event protection passed.');
}).catch(error => { console.error(error); process.exitCode = 1; });
