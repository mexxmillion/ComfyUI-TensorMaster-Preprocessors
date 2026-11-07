import { app } from "../../scripts/app.js";

export const DEFAULT_STATES = [0, 1, 2].map(id => ({ id, label: `State ${id}` }));

export function readStates(raw) {
    const states = JSON.parse(raw);
    if (!Array.isArray(states) || !states.length) throw new Error("Add at least one state.");
    const ids = new Set(), labels = new Set();
    for (const state of states) {
        if (!state || !Number.isInteger(state.id) || state.id < 0 || state.id > 99 || ids.has(state.id)) {
            throw new Error("State ids must be unique integers from 0 to 99.");
        }
        if (typeof state.label !== "string" || !state.label.trim() || labels.has(state.label)) {
            throw new Error("Give each state a unique, nonempty name.");
        }
        ids.add(state.id); labels.add(state.label);
    }
    return states;
}

export function nextStateId(states) {
    const used = new Set(states.map(state => state.id));
    for (let id = 0; id < 100; id++) if (!used.has(id)) return id;
    return null;
}

function el(tag, text) {
    const element = document.createElement(tag);
    if (text !== undefined) element.textContent = text;
    return element;
}

function button(text, action) {
    const control = el("button", text);
    control.type = "button";
    control.onclick = action;
    return control;
}

function menuStates(node) {
    const raw = node.widgets?.find(widget => widget.name === "states_json")?.value;
    return raw === undefined ? DEFAULT_STATES : readStates(raw);
}

function openEditor(node) {
    const selected = node.widgets.find(widget => widget.name === "selected");
    const storage = node.widgets.find(widget => widget.name === "states_json");
    let states;
    try { states = menuStates(node).map(state => ({ ...state })); }
    catch (error) { app.extensionManager.toast.add({ severity: "error", summary: "State menu", detail: error.message }); return; }
    const selectedId = states.find(state => state.label === selected.value)?.id;
    const dialog = el("dialog");
    dialog.className = "tm-state-dialog";
    dialog.setAttribute("aria-label", "Edit state menu");
    dialog.append(el("h2", "State menu"), el("p", "Names appear in the dropdown. State numbers match the inputs on every connected switch."));
    const rows = el("div"); rows.className = "tm-state-rows";
    const status = el("p"); status.setAttribute("role", "status");
    const add = button("+ Add state", () => {
        const id = nextStateId(states);
        if (id === null) return;
        let label = `State ${id}`;
        while (states.some(state => state.label === label)) label += " (new)";
        states.push({ id, label }); render();
    });
    const footer = el("footer");
    footer.append(button("Cancel", () => dialog.close()), button("Save menu", () => {
        try { readStates(JSON.stringify(states)); }
        catch (error) { status.textContent = error.message; return; }
        app.graph?.beforeChange();
        storage.value = JSON.stringify(states);
        selected.value = (states.find(state => state.id === selectedId) ?? states[0]).label;
        storage.callback?.(storage.value);
        selected.callback?.(selected.value);
        app.graph?.afterChange();
        app.graph?.setDirtyCanvas(true, true);
        dialog.close();
    }));
    function render() {
        rows.replaceChildren();
        for (const [index, state] of states.entries()) {
            const row = el("div"); row.className = "tm-state-row";
            const label = el("input"); label.value = state.label;
            label.setAttribute("aria-label", `Name for state ${state.id}`);
            label.oninput = () => { state.label = label.value; };
            const remove = button("Remove", () => { states.splice(index, 1); render(); });
            remove.disabled = states.length === 1;
            row.append(el("span", `State ${state.id}`), label, remove);
            rows.append(row);
        }
        add.disabled = nextStateId(states) === null;
    }
    render();
    dialog.append(rows, add, status, footer);
    document.body.append(dialog);
    node._tmStateDialogs.add(dialog);
    dialog.addEventListener("close", () => { node._tmStateDialogs.delete(dialog); dialog.remove(); }, { once: true });
    dialog.showModal();
}

app.registerExtension({
    name: "TensorMaster.StateMenu",
    getCustomWidgets() {
        return {
            TM_STATE_MENU(node, name, inputData) {
                const values = () => menuStates(node).map(state => state.label);
                const widget = node.addWidget("combo", name, inputData[1].default, () => {}, {
                    values,
                });
                const added = node.onInputAdded;
                node.onInputAdded = function (input) {
                    added?.apply(this, arguments);
                    if (input.widget?.name === name) {
                        window.comfyAPI.widgetInputs.setWidgetConfig(input, ['COMBO', { options: values }]);
                    }
                };
                return { widget };
            },
        };
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "TMStateMenu") return;
        if (!document.getElementById("tm-state-css")) {
            const link = el("link");
            link.id = "tm-state-css"; link.rel = "stylesheet";
            link.href = new URL("./state_menu.css", import.meta.url).href;
            document.head.append(link);
        }
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            created?.apply(this, arguments);
            const node = this;
            node._tmStateDialogs = new Set();
            const storage = node.widgets.find(widget => widget.name === "states_json");
            storage.hidden = true;
            storage.computeSize = () => [0, -4];
            if (storage.element) storage.element.style.display = "none";
            node.addWidget("button", "Edit states…", null, () => openEditor(node), { serialize: false });
            const removed = node.onRemoved;
            node.onRemoved = function () {
                for (const dialog of node._tmStateDialogs) dialog.close();
                removed?.apply(this, arguments);
            };
            node.setSize([280, 110]);
        };
    },
});
