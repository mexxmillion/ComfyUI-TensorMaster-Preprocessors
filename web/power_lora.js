import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function el(tag, className = "", text) {
    const element = document.createElement(tag);
    element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
}

function button(text, label, action) {
    const control = el("button", "", text);
    control.type = "button";
    control.title = label;
    control.setAttribute("aria-label", label);
    control.onclick = action;
    return control;
}

function buildEditor(node, state, expanded = true) {
    const root = el("div", "tmpl-root");
    const header = el("header");
    const count = el("span", "tmpl-count");
    const list = el("div", "tmpl-list");
    const status = el("div", "tmpl-status");
    status.setAttribute("role", "status");
    const picker = el("details", "tmpl-picker");
    picker.append(el("summary", "", "+ Add LoRA"));
    const search = el("input");
    search.type = "search";
    search.placeholder = "Search installed LoRAs…";
    search.setAttribute("aria-label", "Search installed LoRAs");
    const results = el("div", "tmpl-results");
    const refresh = button("↻", "Refresh installed LoRAs", () => loadFiles());
    const searchbar = el("div", "tmpl-search");
    searchbar.append(search, refresh);
    picker.append(searchbar, results);
    const title = el("strong", "", "LoRA stack");
    header.append(title, count, button(expanded ? "−" : "+", "Collapse or expand LoRA list", event => {
        expanded = !expanded;
        list.hidden = !expanded;
        event.currentTarget.textContent = expanded ? "−" : "+";
    }));
    root.append(header, list, picker, status);
    list.hidden = !expanded;
    root.addEventListener("pointerdown", event => event.stopPropagation());
    root.addEventListener("keydown", event => event.stopPropagation());
    root.addEventListener("wheel", event => event.stopPropagation());

    function commit(change) {
        app.graph?.beforeChange();
        change();
        state.widget.value = JSON.stringify(state.entries);
        state.widget.callback?.(state.widget.value);
        app.graph?.afterChange();
        app.graph?.setDirtyCanvas(true, true);
        state.render();
    }

    function renderFiles() {
        const query = search.value.toLowerCase();
        results.replaceChildren();
        const matches = state.files.filter(name => name.toLowerCase().includes(query));
        for (const name of matches.slice(0, 100)) {
            results.append(button(name, `Add ${name}`, () => {
                commit(() => state.entries.push({ name, enabled: true, strength_model: 1, strength_clip: 1 }));
                picker.open = false;
            }));
        }
        if (matches.length > 100) results.append(el("p", "", `${matches.length} matches · refine your search`));
        else if (!matches.length) results.append(el("p", "", "No matching LoRAs"));
    }

    async function loadFiles() {
        refresh.disabled = true;
        try {
            const response = await api.fetchApi("/models/loras");
            if (!response.ok) throw new Error(`Could not list LoRAs (${response.status}).`);
            state.files = await response.json();
            status.textContent = "";
            renderFiles();
        } catch (error) { status.textContent = error.message; }
        finally { refresh.disabled = false; }
    }
    search.oninput = renderFiles;
    picker.ontoggle = () => { if (picker.open) { void loadFiles(); search.focus(); } };

    function render() {
        list.replaceChildren();
        count.textContent = `${state.entries.filter(entry => entry.enabled !== false).length} / ${state.entries.length} enabled`;
        status.textContent = state.error || "Applied from top to bottom";
        picker.hidden = !!state.error;
        if (!state.entries.length) list.append(el("p", "tmpl-empty", "Add LoRAs to build your stack."));
        state.entries.forEach((entry, index) => {
            const row = el("div", `tmpl-row${entry.enabled === false ? " is-disabled" : ""}`);
            const top = el("div", "tmpl-row-top");
            const enabled = el("input");
            enabled.type = "checkbox";
            enabled.checked = entry.enabled !== false;
            enabled.setAttribute("aria-label", `Enable ${entry.name}`);
            enabled.onchange = () => commit(() => { entry.enabled = enabled.checked; });
            const name = el("span", "tmpl-name", entry.name.replaceAll("\\", "/").split("/").pop());
            name.title = entry.name;
            const up = button("↑", `Move ${entry.name} up`, () => commit(() => {
                [state.entries[index - 1], state.entries[index]] = [entry, state.entries[index - 1]];
            }));
            up.disabled = index === 0;
            const down = button("↓", `Move ${entry.name} down`, () => commit(() => {
                [state.entries[index + 1], state.entries[index]] = [entry, state.entries[index + 1]];
            }));
            down.disabled = index === state.entries.length - 1;
            top.append(enabled, name, up, down, button("×", `Remove ${entry.name}`, () => commit(() => state.entries.splice(index, 1))));
            const strengths = el("div", "tmpl-strengths");
            const field = el("label", "", "Strength");
            const strength = entry.strength_model ?? 1;
            const slider = el("input");
            slider.type = "range";
            slider.min = Math.min(-2, strength);
            slider.max = Math.max(2, strength);
            slider.step = "0.01";
            slider.value = strength;
            slider.setAttribute("aria-label", `Strength for ${entry.name}`);
            const number = el("input");
            number.type = "number";
            number.step = "0.01";
            number.value = strength;
            number.setAttribute("aria-label", `Strength value for ${entry.name}`);
            const setStrength = value => commit(() => {
                entry.strength_model = value;
                entry.strength_clip = value;
            });
            slider.oninput = () => { number.value = slider.value; };
            slider.onchange = () => setStrength(slider.valueAsNumber);
            number.onchange = () => {
                if (!number.value || !Number.isFinite(number.valueAsNumber)) {
                    number.value = strength;
                    return;
                }
                setStrength(number.valueAsNumber);
            };
            field.append(slider, number);
            strengths.append(field);
            row.append(top, strengths);
            list.append(row);
        });
    }
    state.views.add(render);
    render();
    return { root, render, dispose: () => state.views.delete(render) };
}

app.registerExtension({
    name: "TensorMaster.PowerLoRA",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "TMMultiLora") return;
        if (!document.getElementById("tmpl-css")) {
            const link = el("link");
            link.id = "tmpl-css"; link.rel = "stylesheet";
            link.href = new URL("./power_lora.css", import.meta.url).href;
            document.head.append(link);
        }
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            created?.apply(this, arguments);
            const node = this;
            const widget = node.widgets.find(widget => widget.name === "loras_json");
            const state = { widget, entries: [], files: [], views: new Set(), dialogs: new Set(), error: "" };
            state.render = () => { for (const render of state.views) render(); };
            function restore() {
                try {
                    const entries = JSON.parse(widget.value);
                    if (!Array.isArray(entries) || entries.some(entry => !entry || typeof entry.name !== "string")) {
                        throw new Error("Invalid LoRA list. Check loras_json.");
                    }
                    state.entries = entries;
                    state.error = "";
                } catch (error) { state.entries = []; state.error = error.message; }
                state.render();
            }
            restore();
            widget.hidden = true;
            widget.computeSize = () => [0, -4];
            if (widget.element) widget.element.style.display = "none";
            node.addWidget("button", "Edit LoRAs", null, () => {
                const dialog = el("dialog", "tmpl-dialog");
                const editor = buildEditor(node, state);
                const close = button("Done", "Close LoRA editor", () => dialog.close());
                dialog.append(editor.root, close);
                document.body.append(dialog);
                state.dialogs.add(dialog);
                dialog.addEventListener("close", () => { editor.dispose(); state.dialogs.delete(dialog); dialog.remove(); }, { once: true });
                dialog.showModal();
            }, { serialize: false });
            const editor = buildEditor(node, state);
            node.addDOMWidget("tm_lora_stack", "custom", editor.root, {
                serialize: false, hideOnZoom: false,
                getMinHeight: () => 180, getHeight: () => Math.max(180, node.size[1] - 130),
            });
            node.setSize([380, 390]);
            const configured = node.onConfigure;
            node.onConfigure = function () { configured?.apply(this, arguments); restore(); };
            const removed = node.onRemoved;
            node.onRemoved = function () {
                for (const dialog of state.dialogs) dialog.close();
                editor.dispose();
                removed?.apply(this, arguments);
            };
        };
    },
});
