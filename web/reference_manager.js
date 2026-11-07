import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NAME = "TMH3ReferenceManager";
const CAPS = { image: 9, video: 3, audio: 3 };
const widget = (node, name) => node.widgets?.find(w => w.name === name);
const viewUrl = (file, type = "input", subfolder = "") => {
    const parts = file.replaceAll("\\", "/").split("/");
    const filename = parts.pop();
    return api.apiURL(`/view?${new URLSearchParams({ filename, type, subfolder: [subfolder, ...parts].filter(Boolean).join("/") })}`);
};
const previewKey = ref => JSON.stringify(ref);

export function parseReferences(raw) {
    const data = JSON.parse(raw || '{"references":[]}');
    const list = Array.isArray(data) ? data : data.references;
    if (!Array.isArray(list)) throw new Error("References must be a list.");
    return list;
}

export function replaceReference(refs, index, file) {
    // Replacement preserves the socket; source-specific edits start fresh.
    return refs.map((ref, i) => i === index ? {
        kind: ref.kind, file,
        ...(ref.kind === "video" ? { use_soundtrack: ref.use_soundtrack !== false } : {}),
    } : ref);
}

export function moveImages(refs, selected, target, after = false) {
    const images = refs.filter(ref => ref.kind === "image");
    const moving = images.filter(ref => selected.has(ref));
    if (!moving.length || (target !== null && (selected.has(target) || !images.includes(target)))) return refs;
    const ordered = images.filter(ref => !selected.has(ref));
    ordered.splice(target === null ? ordered.length : ordered.indexOf(target) + Number(after), 0, ...moving);
    let index = 0;
    return refs.map(ref => ref.kind === "image" ? ordered[index++] : ref);
}

export function referenceTags(refs) {
    let audio = 0;
    const tags = new Map();
    for (const [kind, label] of [["image", "Picture"], ["video", "Video"], ["audio", "Audio"]]) {
        let slot = 0;
        refs.forEach((ref, index) => {
            if (ref.kind !== kind) return;
            let tag = `<${label} ${kind === "audio" ? ++audio : ++slot}>`;
            if (kind === "video" && ref.use_soundtrack !== false) tag += ` <Audio ${++audio}>`;
            tags.set(index, tag);
        });
    }
    return tags;
}

function el(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
}

function button(label, action) {
    const b = el("button", "tmrm-button", label);
    b.type = "button";
    b.onclick = action;
    return b;
}

function field(parent, label, input) {
    const row = el("label", "tmrm-field");
    row.append(el("span", "", label), input);
    parent.append(row);
    return input;
}

function input(type, value) {
    const control = el("input");
    control.type = type;
    if (type === "checkbox") control.checked = !!value;
    else control.value = value;
    return control;
}

function select(values, value) {
    const control = el("select");
    for (const name of values) {
        const option = el("option", "", name);
        option.value = name;
        control.append(option);
    }
    control.value = value;
    return control;
}

function setStatus(node, text, error = false) {
    node._tmStatus = text;
    for (const surface of node._tmSurfaces) {
        surface.status.textContent = text;
        surface.status.classList.toggle("is-error", error);
    }
}

function dialog(node, title, subtitle = "", className = "") {
    const modal = el("dialog", `tmrm-dialog ${className}`);
    modal.setAttribute("aria-label", title);
    const header = el("header", "tmrm-dialog-header");
    const heading = el("div");
    heading.append(el("div", "tmrm-eyebrow", subtitle), el("h2", "", title));
    const close = button("×", () => modal.close());
    close.classList.add("tmrm-icon-button");
    close.setAttribute("aria-label", "Close editor");
    header.append(heading, close);
    const body = el("div", "tmrm-dialog-body");
    const footer = el("footer", "tmrm-dialog-footer");
    modal.append(header, body, footer);
    node._tmDialogs.add(modal);
    modal.addEventListener("close", () => {
        modal.querySelectorAll("video,audio").forEach(media => { media.pause(); media.removeAttribute("src"); media.load(); });
        node._tmDialogs.delete(modal);
        modal.remove();
    }, { once: true });
    modal.addEventListener("click", event => {
        if (event.target !== modal) return;
        const rect = modal.getBoundingClientRect();
        if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) modal.close();
    });
    document.body.append(modal);
    return { modal, body, footer };
}

function primary(label, action) {
    const control = button(label, action);
    control.classList.add("tmrm-primary");
    return control;
}

function toggle(label, checked, description = "") {
    const row = el("label", "tmrm-switch");
    const control = input("checkbox", checked);
    const text = el("span", "tmrm-switch-copy");
    text.append(el("strong", "", label));
    if (description) text.append(el("small", "", description));
    row.append(control, text);
    return { row, control };
}

function section(title, subtitle = "") {
    const panel = el("section", "tmrm-panel");
    const heading = el("div", "tmrm-section-heading");
    heading.append(el("h3", "", title));
    if (subtitle) heading.append(el("span", "tmrm-hint", subtitle));
    panel.append(heading);
    return { panel, heading };
}

async function upload(file, kind) {
    const body = new FormData();
    const bytes = crypto.getRandomValues(new Uint8Array(16));
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = Array.from(bytes, value => value.toString(16).padStart(2, "0")).join("");
    const id = `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
    body.append("image", file, `${id}-${file.name}`);
    body.append("subfolder", "tm-references");
    const response = await api.fetchApi("/upload/image", { method: "POST", body });
    if (!response.ok) throw new Error(await response.text());
    const result = await response.json();
    return { kind, file: [result.subfolder, result.name].filter(Boolean).join("/"),
        ...(kind === "video" ? { use_soundtrack: true } : {}) };
}

function chooseFiles(node, kind, multiple, action) {
    const picker = input("file", "");
    picker.accept = kind ? `${kind}/*` : "image/*,video/*,audio/*";
    picker.multiple = multiple;
    picker.onchange = async () => {
        try { if (picker.files.length) await action(Array.from(picker.files)); }
        catch (error) { setStatus(node, error.message, true); }
    };
    picker.click();
}

async function addFiles(node, files, kind = null) {
    for (const file of files) {
        const type = kind || file.type.split("/")[0];
        if (!(type in CAPS)) { setStatus(node, `Unsupported file: ${file.name}`, true); continue; }
        if (node._tmRefs.filter(ref => ref.kind === type).length >= CAPS[type]) {
            setStatus(node, `All ${CAPS[type]} ${type} sockets are in use. Replace an existing reference.`, true);
            return;
        }
        setStatus(node, `Uploading ${file.name}…`);
        node._tmRefs.push(await upload(file, type));
        commit(node);
    }
    setStatus(node, "References saved in this workflow.");
}

function openSettings(node) {
    const { modal, body, footer } = dialog(node, "AI model & options", "PROMPT ENHANCEMENT", "tmrm-settings-dialog");
    const grid = el("div", "tmrm-settings-grid");
    const controls = [];
    for (const [name, label, type] of [
        ["prompt_provider", "Provider", "provider"], ["model", "Model ID", "text"],
        ["api_base", "Local API base", "text"], ["api_key", "API key", "password"],
        ["reasoning_effort", "Reasoning", "reasoning"], ["max_reference_edge", "Image edge limit · 0 for original", "number"],
        ["system_prompt", "System prompt override", "textarea"],
    ]) {
        const w = widget(node, name);
        const control = type === "provider" ? select(["local", "openrouter"], w.value)
            : type === "reasoning" ? select(["none", "low", "medium", "high"], w.value)
            : type === "textarea" ? el("textarea") : input(type, w.value);
        if (type === "textarea") control.value = w.value;
        if (name === "api_key") { control.autocomplete = "off"; control.placeholder = "Or use OPENROUTER_API_KEY"; }
        if (name === "max_reference_edge") { control.min = 0; control.max = 8192; control.step = 64; }
        field(grid, label, control);
        if (name === "system_prompt") control.parentElement.classList.add("tmrm-span-all");
        controls.push([w, control]);
    }
    body.append(grid, el("p", "tmrm-hint", "Enhancement sends your prompt and references to this provider. Local servers receive images and video stills; audio is omitted. Keys entered here save in the workflow."));
    footer.append(el("span", "tmrm-hint", "Connected prompts use these settings too."), button("Cancel", () => modal.close()),
        primary("Save settings", () => {
            const oldEdge = widget(node, "max_reference_edge").value;
            controls.forEach(([w, control]) => { w.value = control.type === "number" ? Number(control.value) : control.value; });
            if (oldEdge !== widget(node, "max_reference_edge").value) node._tmPreviews.clear();
            app.graph?.change();
            render(node);
            modal.close();
        }));
    modal.showModal();
}

export function cropFromDrag(start, end) {
    const clamp = value => Math.max(0, Math.min(1, value));
    const [x1, y1] = start.map(clamp), [x2, y2] = end.map(clamp);
    return [Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1)];
}

function editedLabel(ref) {
    return [ref.crop && "Crop", ref.trim && `${ref.trim[0]}–${ref.trim[1]}s`,
        ref.preprocess?.isolate && "Isolate", ref.preprocess?.depth && "Depth", ref.preprocess?.face_cover && "Face cover"].filter(Boolean).join(" · ");
}

function displayFilename(file) {
    return file.split("/").pop().replace(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}-/i, "");
}

function editReference(node, index, initialTab = null) {
    const source = node._tmRefs[index];
    if (!source) return;
    const tag = referenceTags(node._tmRefs).get(index);
    const { modal, body, footer } = dialog(node, `Prepare ${tag}`, displayFilename(source.file), "tmrm-editor");
    const previewArea = el("div", "tmrm-editor-previews");
    const previewsHeading = el("div", "tmrm-preview-heading");
    previewsHeading.append(el("span", "", "Review your reference"), el("span", "tmrm-hint", "Original files stay untouched"));
    const figures = el("div", "tmrm-compare");
    const originalFigure = el("figure"), resultFigure = el("figure");
    const originalCaption = el("figcaption", "", "Original");
    const resultCaption = el("figcaption", "", "Generation reference");
    const originalViewport = el("div", "tmrm-viewport");
    const resultViewport = el("div", "tmrm-viewport");
    const stage = el("div", "tmrm-media-stage");
    const media = el(source.kind === "image" ? "img" : source.kind);
    media.src = viewUrl(source.file);
    if (source.kind !== "image") { media.controls = true; media.preload = "metadata"; }
    stage.append(media);
    originalViewport.append(stage);
    const resultImage = source.kind === "image" ? el("img", "tmrm-result-image") : null;
    const placeholder = el("div", "tmrm-result-placeholder");
    placeholder.append(el("span", "tmrm-placeholder-mark", "◈"), el("strong", "", "Your prepared reference"),
        el("span", "", "Choose your tools, then preview the result here."));
    if (resultImage) { resultImage.hidden = true; resultViewport.append(resultImage); }
    resultViewport.append(placeholder);
    originalFigure.append(originalCaption, originalViewport);
    resultFigure.append(resultCaption, resultViewport);
    figures.append(originalFigure, resultFigure);
    previewArea.append(previewsHeading, figures);
    const previewNote = el("div", "tmrm-preview-note", source.kind === "image"
        ? "Preview processes only this image. Your H3 generation will not run."
        : "Trim keeps the selected time range. Crop is applied when the reference runs.");
    previewArea.append(previewNote);
    const sidebar = el("aside", "tmrm-editor-sidebar");
    body.append(previewArea, sidebar);
    let crop = source.crop?.slice() || null;
    let width = 1, height = 1, version = 0;
    const status = el("span", "tmrm-editor-status", "Settings save with the reference.");
    status.setAttribute("role", "status");
    const previewBtn = button(source.kind === "image" ? "Preview result" : "Play trim", async () => {});
    const saveBtn = primary("Save reference", () => {});
    footer.append(status, button("Cancel", () => modal.close()), previewBtn, saveBtn);
    modal.addEventListener("close", () => {
        const pending = node._tmPending;
        if (pending?.owner === modal) {
            pending.cancelled = true;
            failPreview(node, pending, "Preview cancelled.");
            void cancelPreview(pending).catch(error => setStatus(node, error.message, true));
        }
    }, { once: true });
    const tabs = el("div", "tmrm-tabs");
    tabs.setAttribute("role", "tablist");
    tabs.setAttribute("aria-label", "Reference tools");
    const panels = new Map();
    let activeTab;
    const activate = name => {
        activeTab = name;
        for (const [key, value] of panels) {
            value.panel.hidden = key !== name;
            value.button.setAttribute("aria-selected", String(key === name));
        }
        stage.classList.toggle("is-cropping", name === "Crop");
        originalCaption.textContent = name === "Crop" ? "Original · drag to crop" : "Original";
    };
    for (const name of source.kind === "image" ? ["Prepare", "Crop"] : source.kind === "video" ? ["Trim", "Crop"] : ["Trim"]) {
        const tab = button(name, () => activate(name));
        tab.setAttribute("role", "tab");
        const panel = el("div", "tmrm-tool-panel");
        panel.setAttribute("role", "tabpanel");
        panels.set(name, { button: tab, panel });
        tabs.append(tab);
    }
    sidebar.append(tabs, ...Array.from(panels.values(), value => value.panel));
    const changed = () => {
        version++;
        status.textContent = "Unsaved changes";
        status.classList.remove("is-error");
        resultCaption.textContent = resultImage && !resultImage.hidden ? "Previous preview · update to refresh" : "Generation reference";
    };
    let rectangle;
    const cropInputs = [];
    const updateCrop = () => {
        const rect = crop || [0, 0, 1, 1];
        if (rectangle) Object.assign(rectangle.style, { left: `${rect[0] * 100}%`, top: `${rect[1] * 100}%`, width: `${rect[2] * 100}%`, height: `${rect[3] * 100}%` });
        cropInputs.forEach((control, i) => { control.value = +(rect[i] * 100).toFixed(2); });
    };
    const fitMedia = () => {
        if (source.kind === "audio") return;
        const scale = Math.min((originalViewport.clientWidth - 24) / width, (originalViewport.clientHeight - 24) / height);
        if (scale > 0) Object.assign(stage.style, { width: `${width * scale}px`, height: `${height * scale}px` });
    };
    const resize = new ResizeObserver(fitMedia);
    resize.observe(originalViewport);
    const loaded = () => {
        width = source.kind === "image" ? media.naturalWidth : media.videoWidth;
        height = source.kind === "image" ? media.naturalHeight : media.videoHeight;
        fitMedia();
    };
    media.addEventListener(source.kind === "image" ? "load" : "loadedmetadata", loaded);
    modal.addEventListener("close", () => resize.disconnect(), { once: true });
    if (source.kind !== "audio") {
        const cropPanel = panels.get("Crop").panel;
        cropPanel.append(el("h3", "", "Frame your reference"), el("p", "tmrm-hint", "Drag a new frame on the original. Move the selection or drag a corner to adjust it."));
        const presets = el("div", "tmrm-aspects");
        for (const [label, ratio] of [["Original", null], ["1:1", 1], ["16:9", 16 / 9], ["9:16", 9 / 16], ["4:3", 4 / 3]]) {
            presets.append(button(label, () => {
                if (ratio === null) crop = null;
                else {
                    const w = Math.min(1, ratio * height / width), h = Math.min(1, width / (ratio * height));
                    crop = [(1 - w) / 2, (1 - h) / 2, w, h];
                }
                updateCrop(); changed();
            }));
        }
        cropPanel.append(presets, button("Reset crop", () => { crop = null; updateCrop(); changed(); }));
        const precise = el("details", "tmrm-details");
        precise.append(el("summary", "", "Precise crop · percentages"));
        const fields = el("div", "tmrm-two-columns");
        for (const label of ["Left %", "Top %", "Width %", "Height %"]) {
            const control = input("number", 0);
            control.min = 0; control.max = 100; control.step = 0.1;
            control.onchange = () => { crop = cropInputs.map(c => Number(c.value) / 100); updateCrop(); changed(); };
            cropInputs.push(control);
            field(fields, label, control);
        }
        precise.append(fields); cropPanel.append(precise);
        rectangle = el("div", "tmrm-crop");
        for (const corner of ["nw", "ne", "sw", "se"]) {
            const handle = el("span", `tmrm-crop-handle ${corner}`);
            handle.dataset.corner = corner;
            rectangle.append(handle);
        }
        stage.append(rectangle);
        let drag;
        const point = event => {
            const bounds = stage.getBoundingClientRect();
            return [(event.clientX - bounds.left) / bounds.width, (event.clientY - bounds.top) / bounds.height];
        };
        stage.onpointerdown = event => {
            if (activeTab !== "Crop") return;
            event.preventDefault();
            const p = point(event), rect = crop || [0, 0, 1, 1];
            const corner = event.target.dataset.corner;
            drag = { start: p, rect: rect.slice(), mode: corner || (crop && event.target === rectangle ? "move" : "new") };
            stage.setPointerCapture(event.pointerId);
        };
        stage.onpointermove = event => {
            if (!drag) return;
            const p = point(event), [x, y, w, h] = drag.rect;
            if (drag.mode === "move") {
                crop = [Math.max(0, Math.min(1 - w, x + p[0] - drag.start[0])), Math.max(0, Math.min(1 - h, y + p[1] - drag.start[1])), w, h];
            } else {
                const opposite = drag.mode === "new" ? drag.start : [drag.mode.includes("w") ? x + w : x, drag.mode.includes("n") ? y + h : y];
                const rect = cropFromDrag(opposite, p);
                if (rect[2] > 0.002 && rect[3] > 0.002) crop = rect;
            }
            updateCrop(); changed();
        };
        stage.onpointerup = stage.onpointercancel = () => { drag = null; };
        updateCrop();
    }
    const controls = {};
    if (source.kind === "image") {
        const prepare = panels.get("Prepare").panel;
        const options = source.preprocess || {};
        const isolate = toggle("Isolate subject", options.isolate, "Keep the subject. Replace its background.");
        controls.isolate = isolate.control;
        prepare.append(isolate.row);
        const isolation = el("div", "tmrm-tool-options");
        controls.text = field(isolation, "What should we keep?", input("text", options.text || "person"));
        controls.text.placeholder = "person, black shirt, product…";
        controls.selection = field(isolation, "Subject", select(["best", "all", ...Array.from({ length: 20 }, (_, i) => String(i + 1))], options.selection || "best"));
        const output = el("div", "tmrm-two-columns");
        controls.background = field(output, "Background", select(["gray", "white", "black"], options.background || "gray"));
        controls.confidence = field(output, "Confidence", input("number", options.confidence ?? 0.25));
        controls.confidence.min = 0; controls.confidence.max = 1; controls.confidence.step = 0.05;
        isolation.append(output);
        const subjectCrop = toggle("Crop to subject", options.crop_subject);
        controls.crop_subject = subjectCrop.control;
        isolation.append(subjectCrop.row);
        prepare.append(isolation);
        const depth = toggle("Depth map", options.depth, "Use scene depth as the reference.");
        controls.depth = depth.control;
        const inversion = toggle("Invert depth", options.invert_depth, "Swap near and far.");
        controls.invert_depth = inversion.control;
        const depthOptions = el("div", "tmrm-tool-options");
        depthOptions.append(inversion.row);
        prepare.append(depth.row, depthOptions);
        const faceCover = toggle("Cover faces", options.face_cover, "Solid circles or ellipses over detected faces.");
        controls.face_cover = faceCover.control;
        const faceOptions = el("div", "tmrm-tool-options");
        controls.face_detector = field(faceOptions, "Face detection", select(["retinaface", "frontal", "frontal + profile"], options.face_detector || "retinaface"));
        const selection = el("div", "tmrm-two-columns");
        controls.face_number = field(selection, "Number of faces · 0 = all", input("number", options.face_number ?? 0));
        controls.face_start_index = field(selection, "Start index · first = 0", input("number", options.face_start_index ?? 0));
        for (const control of [controls.face_number, controls.face_start_index]) { control.min = 0; control.max = 1000; control.step = 1; }
        faceOptions.append(selection);
        controls.face_order = field(faceOptions, "Face order", select(["confidence", "left to right", "largest first"], options.face_order || "confidence"));
        const appearance = el("div", "tmrm-two-columns");
        controls.face_shape = field(appearance, "Shape", select(["ellipse", "circle"], options.face_shape || "ellipse"));
        controls.face_color = field(appearance, "Solid color", input("color", options.face_color || "#808080"));
        faceOptions.append(appearance);
        controls.face_size = field(faceOptions, "Scale factor · cover face / hair", input("number", options.face_size ?? 1));
        controls.face_size.min = 0.1; controls.face_size.max = 5; controls.face_size.step = 0.1;
        const coverage = el("div", "tmrm-two-columns");
        const detectionSettings = el("details", "tmrm-details");
        detectionSettings.append(el("summary", "", "Detection sensitivity"));
        const detectionFields = el("div", "tmrm-two-columns");
        for (const [name, label, value, min, max, step] of [
            ["face_scale_x", "Width coverage", 1.35, 0.1, 5, 0.05],
            ["face_scale_y", "Height coverage", 1.5, 0.1, 5, 0.05],
            ["face_neighbors", "Detection agreement", 5, 1, 30, 1],
            ["face_min_size", "Minimum face · px", 24, 1, 2048, 1],
            ["face_shift", "Vertical shift · 0.5 = center", 0.5, 0, 1, 0.05],
            ["face_confidence", "RetinaFace confidence", 0.6, 0, 1, 0.05],
            ["face_max_faces", "Max faces per image", 50, 1, 1000, 1],
        ]) {
            const control = input("number", options[name] ?? value);
            control.min = min; control.max = max; control.step = step;
            controls[name] = field(["face_neighbors", "face_min_size", "face_confidence", "face_max_faces"].includes(name) ? detectionFields : coverage, label, control);
        }
        detectionSettings.append(detectionFields);
        faceOptions.append(coverage, detectionSettings, el("p", "tmrm-hint", "Review the preview for missed faces. Coverage controls enlarge the shape around each detection."));
        prepare.append(faceCover.row, faceOptions);
        const syncTools = () => {
            isolation.hidden = !isolate.control.checked;
            depthOptions.hidden = !depth.control.checked;
            faceOptions.hidden = !faceCover.control.checked;
            controls.face_neighbors.parentElement.hidden = controls.face_detector.value === "retinaface";
            controls.face_confidence.parentElement.hidden = controls.face_detector.value !== "retinaface";
        };
        for (const control of Object.values(controls)) control.addEventListener("input", () => { syncTools(); changed(); });
        syncTools();
    }
    let trimStart, trimEnd, useSoundtrack, resultMedia;
    if (source.kind !== "image") {
        const trim = panels.get("Trim").panel;
        trim.append(el("h3", "", "Choose your range"), el("p", "tmrm-hint", "Play the original and set your in and out points."));
        const times = el("div", "tmrm-two-columns");
        trimStart = field(times, "In · seconds", input("number", source.trim?.[0] || 0));
        trimEnd = field(times, "Out · seconds", input("number", source.trim?.[1] ?? ""));
        for (const control of [trimStart, trimEnd]) { control.min = 0; control.step = 0.01; control.oninput = changed; }
        trim.append(times, button("Set in at playhead", () => { trimStart.value = +media.currentTime.toFixed(2); trimStart.dispatchEvent(new Event("input")); }),
            button("Set out at playhead", () => { trimEnd.value = +media.currentTime.toFixed(2); trimEnd.dispatchEvent(new Event("input")); }));
        const range = el("div", "tmrm-trim-ranges");
        for (const [name, target] of [["In point", trimStart], ["Out point", trimEnd]]) {
            const slider = input("range", target.value || 0);
            slider.min = 0; slider.max = 1; slider.step = 0.01;
            field(range, name, slider);
            media.addEventListener("loadedmetadata", () => { slider.max = media.duration; slider.value = target.value || (target === trimEnd ? media.duration : 0); });
            slider.oninput = () => { target.value = slider.value; changed(); };
            target.addEventListener("input", () => { slider.value = target.value || (target === trimEnd ? media.duration : 0); });
        }
        trim.append(range, button("Reset trim", () => { trimStart.value = 0; trimEnd.value = "";
            trimStart.dispatchEvent(new Event("input")); trimEnd.dispatchEvent(new Event("input")); }));
        if (source.kind === "video") {
            const sound = toggle("Use soundtrack", source.use_soundtrack !== false, "Output this video's audio as a reference.");
            useSoundtrack = sound.control;
            sound.control.onchange = changed;
            trim.append(sound.row);
        }
        resultMedia = el(source.kind);
        resultMedia.src = viewUrl(source.file);
        resultMedia.preload = "metadata";
        resultMedia.controls = source.kind === "audio";
        resultViewport.prepend(resultMedia);
        resultMedia.hidden = true;
        resultMedia.ontimeupdate = () => {
            if (trimEnd.value && resultMedia.currentTime >= Number(trimEnd.value)) resultMedia.pause();
        };
        resultMedia.onpause = resultMedia.onended = () => { previewBtn.textContent = "Play trim"; };
        resultMedia.onplay = () => { previewBtn.textContent = "Pause preview"; };
    }
    const draft = () => {
        const entry = { ...source };
        if (crop && (!crop.every(Number.isFinite) || crop[0] < 0 || crop[1] < 0 || crop[2] <= 0 || crop[3] <= 0 || crop[0] + crop[2] > 1.000001 || crop[1] + crop[3] > 1.000001)) throw new Error("Keep the crop inside the image with positive width and height.");
        if (crop) entry.crop = crop.slice(); else delete entry.crop;
        if (trimEnd?.value) {
            const range = [Number(trimStart.value), Number(trimEnd.value)];
            if (!range.every(Number.isFinite) || range[0] < 0 || range[1] <= range[0]) throw new Error("Out must be later than in.");
            entry.trim = range;
        } else delete entry.trim;
        if (useSoundtrack) entry.use_soundtrack = useSoundtrack.checked;
        if (source.kind === "image") {
            if (controls.isolate.checked || controls.depth.checked || controls.face_cover.checked) entry.preprocess = Object.fromEntries(Object.entries(controls).map(([name, control]) =>
                [name, control.type === "checkbox" ? control.checked : control.type === "number" ? Number(control.value) : control.value]));
            else delete entry.preprocess;
        }
        return entry;
    };
    const showResult = preview => {
        if (!modal.isConnected) return;
        resultImage.src = viewUrl(preview.filename, preview.type, preview.subfolder);
        resultImage.hidden = false;
        placeholder.hidden = true;
        resultCaption.textContent = "Generation reference · preview ready";
    };
    const existing = node._tmPreviews.get(previewKey(source));
    if (existing && resultImage) showResult(existing);
    previewBtn.onclick = async () => {
        try {
            const entry = draft(), submittedVersion = version;
            if (source.kind !== "image") {
                if (!resultMedia.paused) { resultMedia.pause(); previewBtn.textContent = "Play trim"; return; }
                resultMedia.hidden = false; placeholder.hidden = true;
                resultMedia.currentTime = Number(trimStart.value);
                // Match the crop's aspect and fit the selected area in the result viewport.
                if (source.kind === "video") {
                    const [x, y, w, h] = crop || [0, 0, 1, 1];
                    const scale = Math.min(resultViewport.clientWidth / (width * w), resultViewport.clientHeight / (height * h));
                    Object.assign(resultMedia.style, { position: "absolute", width: `${width * scale}px`, height: `${height * scale}px`, maxWidth: "none", maxHeight: "none",
                        left: `${(resultViewport.clientWidth - width * w * scale) / 2 - x * width * scale}px`, top: `${(resultViewport.clientHeight - height * h * scale) / 2 - y * height * scale}px` });
                    resultMedia.style.clipPath = `inset(${y * 100}% ${(1 - x - w) * 100}% ${(1 - y - h) * 100}% ${x * 100}%)`;
                }
                await resultMedia.play();
                previewBtn.textContent = "Pause preview";
                return;
            }
            previewBtn.disabled = true;
            previewBtn.textContent = "Processing…";
            status.textContent = "Queued · preparing this reference…";
            status.classList.remove("is-error");
            const preview = await queuePreview(node, entry, modal);
            if (!modal.isConnected) return;
            if (submittedVersion === version) { showResult(preview); status.textContent = "Preview ready. Save when you're happy with it."; }
            else status.textContent = "Settings changed during processing. Preview again to update.";
        } catch (error) {
            if (modal.isConnected) { status.textContent = error.message; status.classList.add("is-error"); }
        } finally {
            if (source.kind === "image") { previewBtn.disabled = false; previewBtn.textContent = "Preview result"; }
        }
    };
    saveBtn.onclick = () => {
        try {
            if (node._tmRefs[index] !== source) throw new Error("Reference changed. Reopen its editor.");
            node._tmRefs[index] = draft();
            commit(node);
            setStatus(node, "Reference settings saved.");
            modal.close();
        } catch (error) { status.textContent = error.message; status.classList.add("is-error"); }
    };
    activate(initialTab || (source.kind === "image" ? "Prepare" : "Trim"));
    modal.showModal();
}

function failPreview(node, pending, message) {
    if (!pending || node._tmPending !== pending) return;
    node._tmPending = null;
    pending.reject(new Error(message));
    setStatus(node, message, true);
}

async function checkPreviewQueue(node) {
    const pending = node._tmPending;
    if (!pending?.promptId) return;
    try {
        const response = await api.fetchApi("/queue");
        if (!response.ok) throw new Error("Could not check the preview queue.");
        const queue = await response.json();
        if (![...queue.queue_running, ...queue.queue_pending].some(item => item[1] === pending.promptId)) {
            failPreview(node, pending, "Preview stopped. Preview again to retry.");
        }
    } catch (error) {
        failPreview(node, pending, "ComfyUI disconnected. Reconnect, then preview again.");
    }
}

async function cancelPreview(pending) {
    if (!pending.promptId) return;
    for (const [url, body] of [["/queue", { delete: [pending.promptId] }], ["/interrupt", { prompt_id: pending.promptId }]]) {
        const response = await api.fetchApi(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
        if (!response.ok) throw new Error("Could not cancel the preview. Check the ComfyUI connection.");
    }
}

async function queuePreview(node, ref, owner) {
    if (node._tmPending) await checkPreviewQueue(node);
    if (node._tmPending) throw new Error("A reference is already processing. Wait for its preview.");
    const inputs = Object.fromEntries(["model", "prompt_provider", "api_base", "reasoning_effort", "system_prompt", "max_reference_edge", "resize_references"]
        .map(name => [name, widget(node, name).value]));
    let pending;
    const result = new Promise((resolve, reject) => { pending = { key: previewKey(ref), resolve, reject, promptId: null, owner }; });
    result.catch(() => {});
    node._tmPending = pending;
    setStatus(node, "Reference preview queued…");
    try {
        const dependencies = {};
        if (inputs.resize_references) {
            const graph = await app.graphToPrompt();
            const current = graph.output[String(node.id)];
            const isLink = value => Array.isArray(value) && value.length === 2 && typeof value[1] === "number" && graph.output[String(value[0])];
            const include = id => {
                if (String(id) === String(node.id)) throw new Error("Target dimensions cannot depend on this reference manager.");
                if (dependencies[id]) return;
                dependencies[id] = graph.output[id];
                for (const value of Object.values(dependencies[id].inputs)) {
                    if (isLink(value)) include(value[0]);
                }
            };
            for (const name of ["width", "height"]) {
                if (current.inputs[name] === undefined) continue;
                inputs[name] = current.inputs[name];
                if (isLink(inputs[name])) include(inputs[name][0]);
            }
        }
        const response = await api.queuePrompt(0, { output: { ...dependencies, [String(node.id)]: { class_type: NAME, inputs: {
            ...inputs, api_key: "", references_json: JSON.stringify({ references: [ref] }), direction: "", ai_enhance: false,
        } } }, workflow: { nodes: [], links: [] } });
        pending.promptId = response.prompt_id;
        if (pending.cancelled) await cancelPreview(pending);
    } catch (error) {
        if (node._tmPending === pending) node._tmPending = null;
        pending.reject(error);
        setStatus(node, "Could not queue this preview.", true);
    }
    return result;
}

function commit(node) {
    widget(node, "references_json").value = JSON.stringify({ references: node._tmRefs });
    const active = new Set(node._tmRefs.map(previewKey));
    for (const ref of node._tmSelected) if (!node._tmRefs.includes(ref)) node._tmSelected.delete(ref);
    for (const key of node._tmPreviews.keys()) if (!active.has(key) && key !== node._tmPending?.key) node._tmPreviews.delete(key);
    render(node);
    app.graph?.change();
    app.graph?.setDirtyCanvas(true, true);
}

function resetReference(node, index) {
    const { crop, trim, preprocess, ...original } = node._tmRefs[index];
    node._tmRefs[index] = original;
    commit(node);
}

function renderSurface(node, surface) {
    const { gallery, filter } = surface;
    gallery.querySelectorAll("video,audio").forEach(media => media.pause());
    gallery.replaceChildren();
    const tags = referenceTags(node._tmRefs);
    surface.referencesHeading.textContent = `02  References${node._tmSelected.size ? ` · ${node._tmSelected.size} selected` : ""}`;
    for (const [kind, tab] of surface.filters) {
        const count = kind === "all" ? node._tmRefs.length : node._tmRefs.filter(ref => ref.kind === kind).length;
        tab.textContent = `${kind === "all" ? "All" : kind === "image" ? "Images" : kind === "video" ? "Video" : "Audio"} ${count}`;
        tab.setAttribute("aria-selected", String(filter === kind));
    }
    for (const [index, ref] of node._tmRefs.entries()) {
        if (filter !== "all" && ref.kind !== filter) continue;
        const card = el("article", "tmrm-card");
        card.classList.toggle("is-selected", node._tmSelected.has(ref));
        const preview = node._tmPreviews.get(previewKey(ref));
        const thumb = button("", event => {
            if (ref.kind === "image" && (event.ctrlKey || event.metaKey || event.shiftKey)) {
                if (event.shiftKey && node._tmSelectionAnchor && node._tmRefs.includes(node._tmSelectionAnchor)) {
                    const start = node._tmRefs.indexOf(node._tmSelectionAnchor);
                    for (const item of node._tmRefs.slice(Math.min(start, index), Math.max(start, index) + 1)) {
                        if (item.kind === "image") node._tmSelected.add(item);
                    }
                } else if (node._tmSelected.has(ref)) node._tmSelected.delete(ref);
                else node._tmSelected.add(ref);
                node._tmSelectionAnchor = ref;
                render(node);
            } else editReference(node, index);
        });
        thumb.className = "tmrm-thumbnail";
        thumb.setAttribute("aria-label", `Prepare ${tags.get(index)}`);
        const media = el(ref.kind === "image" ? "img" : ref.kind, "tmrm-preview");
        media.draggable = false;
        media.src = preview ? viewUrl(preview.filename, preview.type, preview.subfolder) : viewUrl(ref.file);
        if (ref.kind !== "image") { media.preload = "metadata"; media.muted = true; }
        media.onerror = () => { card.classList.add("has-missing-media"); thumb.title = "Source missing or unsupported. Replace this reference."; };
        const badge = el("span", "tmrm-tag", tags.get(index).replaceAll("<", "").replaceAll(">", ""));
        thumb.append(media, badge);
        if (ref.kind === "image") {
            thumb.draggable = true;
            thumb.title = "Drag to reorder. Select several images to move them together.";
            const selection = input("checkbox", node._tmSelected.has(ref));
            selection.className = "tmrm-select-image";
            selection.setAttribute("aria-label", `Select ${tags.get(index)}`);
            selection.onchange = () => {
                if (selection.checked) node._tmSelected.add(ref); else node._tmSelected.delete(ref);
                node._tmSelectionAnchor = ref;
                render(node);
            };
            card.append(selection);
            card.ondragstart = event => {
                if (!node._tmSelected.has(ref)) { node._tmSelected.clear(); node._tmSelected.add(ref); }
                node._tmDrag = new Set(node._tmSelected);
                event.dataTransfer.effectAllowed = "move";
                event.dataTransfer.setData("application/x-tm-reference-order", "images");
                // Keep the dragged DOM in place until the browser finishes the drag.
                for (const view of node._tmSurfaces) view.referencesHeading.textContent = `02  References · ${node._tmSelected.size} selected`;
            };
            card.ondragover = event => {
                if (!node._tmDrag || node._tmDrag.has(ref)) return;
                event.preventDefault(); event.stopPropagation();
                event.dataTransfer.dropEffect = "move";
                const rect = card.getBoundingClientRect();
                const after = event.clientX > rect.left + rect.width / 2;
                card.classList.toggle("tmrm-drop-before", !after);
                card.classList.toggle("tmrm-drop-after", after);
            };
            card.ondragleave = event => {
                if (!card.contains(event.relatedTarget)) card.classList.remove("tmrm-drop-before", "tmrm-drop-after");
            };
            card.ondrop = event => {
                if (!node._tmDrag) return;
                event.preventDefault(); event.stopPropagation();
                const rect = card.getBoundingClientRect();
                node._tmRefs = moveImages(node._tmRefs, node._tmDrag, ref, event.clientX > rect.left + rect.width / 2);
                node._tmDrag = null;
                commit(node);
                setStatus(node, "Image order saved. Picture tags and image sockets follow the new order.");
            };
            card.ondragend = () => {
                node._tmDrag = null;
                for (const view of node._tmSurfaces) {
                    view.root.classList.remove("is-dragging");
                    view.gallery.querySelectorAll(".tmrm-drop-before,.tmrm-drop-after").forEach(item => item.classList.remove("tmrm-drop-before", "tmrm-drop-after"));
                }
                render(node);
            };
        }
        const remove = button("×", () => { node._tmRefs.splice(index, 1); commit(node); });
        remove.classList.add("tmrm-remove");
        remove.setAttribute("aria-label", `Remove ${tags.get(index)}`);
        const actions = el("div", "tmrm-card-actions");
        actions.append(button("Process", () => editReference(node, index)), button("Replace", () => chooseFiles(node, ref.kind, false, async files => {
            setStatus(node, "Uploading replacement…");
            const uploaded = await upload(files[0], ref.kind);
            if (node._tmRefs[index] !== ref) throw new Error("Reference changed during upload. Try again.");
            node._tmRefs = replaceReference(node._tmRefs, index, uploaded.file);
            commit(node); setStatus(node, "Reference replaced. Its socket stays the same.");
        })));
        card.append(thumb, remove, actions);
        const edits = editedLabel(ref);
        if (edits) {
            const state = el("div", "tmrm-card-state");
            state.append(el("span", "", edits), button("Reset", () => resetReference(node, index)));
            card.append(state);
        }
        card.title = ref.file;
        gallery.append(card);
    }
    const add = button("", () => chooseFiles(node, filter === "all" ? null : filter, true, files => addFiles(node, files, filter === "all" ? null : filter)));
    add.className = "tmrm-add-tile";
    add.append(el("span", "tmrm-add-symbol", "+"), el("strong", "", node._tmRefs.length ? "Add reference" : "Add your references"),
        el("small", "", "Drop files or browse"));
    gallery.append(add);
    const connected = node.inputs?.findIndex(value => value.name === "prompt");
    const linked = connected >= 0 && node.isInputConnected(connected);
    surface.source.textContent = linked ? "Connected prompt" : "Your prompt";
    surface.source.classList.toggle("is-connected", linked);
    surface.direction.disabled = linked;
    if (surface.direction.value !== widget(node, "direction").value) surface.direction.value = widget(node, "direction").value || "";
    surface.enhance.checked = widget(node, "ai_enhance").value;
    surface.resize.checked = widget(node, "resize_references").value;
    surface.promptNote.textContent = linked ? "The connected text is used. AI enhancement still follows your switch." : "Use the text box or connect a prompt to the node's input.";
    surface.result.hidden = !node._tmPromptResult;
    surface.resultText.value = node._tmPromptResult || "";
    surface.status.textContent = node._tmStatus;
}

function render(node) {
    for (const surface of node._tmSurfaces) renderSurface(node, surface);
}

function openStudio(node) {
    const { modal, body, footer } = dialog(node, "H3 Reference Studio", "TENSOR MASTER · MINIMAX", "tmrm-studio-dialog");
    const surface = buildSurface(node, true);
    body.append(surface.root);
    footer.append(el("span", "tmrm-hint", "Changes sync with your node and workflow."), primary("Done", () => modal.close()));
    modal.addEventListener("close", () => node._tmSurfaces.delete(surface), { once: true });
    modal.showModal();
}

function buildSurface(node, expanded = false) {
    const root = el("div", `tmrm-root${expanded ? " tmrm-expanded" : ""}`);
    const top = el("div", "tmrm-topbar");
    const title = el("div", "tmrm-brand");
    title.append(el("span", "tmrm-brand-mark", "H3"), el("span", "", "Reference Studio"));
    top.append(title);
    if (!expanded) top.append(button("Expand ↗", () => openStudio(node)));
    const content = el("div", "tmrm-content");
    const prompt = section("01  Describe your video");
    const source = el("span", "tmrm-source-pill", "Your prompt");
    prompt.heading.append(source);
    const direction = el("textarea", "tmrm-direction");
    direction.placeholder = "Describe the scene, action and camera. Use <Picture 1> or <Video 1> to name a reference…";
    direction.setAttribute("aria-label", "Your video prompt");
    direction.oninput = () => {
        widget(node, "direction").value = direction.value;
        for (const surface of node._tmSurfaces) if (surface.direction !== direction) surface.direction.value = direction.value;
        app.graph?.change();
    };
    const promptNote = el("p", "tmrm-hint tmrm-prompt-note");
    const enhancement = toggle("AI enhance prompt", widget(node, "ai_enhance").value, "Use the H3 prompt writer");
    enhancement.control.onchange = () => { widget(node, "ai_enhance").value = enhancement.control.checked; app.graph?.change(); render(node); };
    const options = el("div", "tmrm-prompt-options");
    options.append(enhancement.row, button("Model & options", () => openSettings(node)));
    const resizing = toggle("Resize references to target", widget(node, "resize_references").value,
        "Exact width × height from the node inputs");
    resizing.row.classList.add("tmrm-resize-option");
    resizing.control.onchange = () => {
        widget(node, "resize_references").value = resizing.control.checked;
        node._tmPreviews.clear();
        app.graph?.change(); render(node);
    };
    const result = el("details", "tmrm-details tmrm-prompt-result");
    result.append(el("summary", "", "Last output prompt"));
    const resultText = el("textarea"); resultText.readOnly = true;
    resultText.setAttribute("aria-label", "Last output prompt"); result.append(resultText);
    prompt.panel.append(direction, options, resizing.row, promptNote, result);
    const references = section("02  References");
    const referencesHeading = references.heading.querySelector("h3");
    references.heading.append(button("Upload +", () => chooseFiles(node, null, true, files => addFiles(node, files))));
    const filters = el("div", "tmrm-tabs tmrm-filter-tabs");
    filters.setAttribute("role", "tablist"); filters.setAttribute("aria-label", "Reference type");
    const gallery = el("div", "tmrm-gallery");
    gallery.ondragover = event => {
        event.preventDefault();
        if (node._tmDrag) event.dataTransfer.dropEffect = "move";
        else root.classList.add("is-dragging");
    };
    gallery.ondragleave = () => root.classList.remove("is-dragging");
    gallery.ondrop = async event => {
        event.preventDefault(); event.stopPropagation(); root.classList.remove("is-dragging");
        if (node._tmDrag) {
            node._tmRefs = moveImages(node._tmRefs, node._tmDrag, null);
            node._tmDrag = null;
            commit(node);
            setStatus(node, "Image order saved. Picture tags and image sockets follow the new order.");
            return;
        }
        try { await addFiles(node, Array.from(event.dataTransfer.files)); }
        catch (error) { setStatus(node, error.message, true); }
    };
    references.panel.append(filters, gallery);
    const status = el("div", "tmrm-status", node._tmStatus);
    status.setAttribute("role", "status");
    content.append(prompt.panel, references.panel);
    root.append(top, content, status);
    root.addEventListener("wheel", event => event.stopPropagation());
    root.addEventListener("pointerdown", event => event.stopPropagation());
    root.addEventListener("keydown", event => event.stopPropagation());
    const surface = { root, gallery, status, direction, referencesHeading, enhance: enhancement.control, resize: resizing.control, promptNote, source, result, resultText, filters: new Map(), filter: "all" };
    for (const kind of ["all", "image", "video", "audio"]) {
        const tab = button("", () => { surface.filter = kind; renderSurface(node, surface); });
        tab.setAttribute("role", "tab"); filters.append(tab); surface.filters.set(kind, tab);
    }
    node._tmSurfaces.add(surface);
    renderSurface(node, surface);
    return surface;
}

app.registerExtension({
    name: "TensorMaster.H3ReferenceManager",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NAME) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            created?.apply(this, arguments);
            const node = this;
            if (!document.getElementById("tmrm-css")) {
                const link = el("link"); link.id = "tmrm-css"; link.rel = "stylesheet";
                link.href = new URL("./reference_manager.css", import.meta.url).href;
                document.head.append(link);
            }
            node._tmRefs = parseReferences(widget(node, "references_json").value);
            node._tmPreviews = new Map();
            node._tmSelected = new Set();
            node._tmSelectionAnchor = null;
            node._tmDrag = null;
            node._tmSurfaces = new Set();
            node._tmDialogs = new Set();
            node._tmStatus = "Ready · images, video & audio";
            node._tmPromptResult = "";
            node._tmPending = null;
            for (const name of ["references_json", "direction", "ai_enhance", "max_reference_edge", "prompt_provider", "model", "api_base", "api_key", "reasoning_effort", "system_prompt", "resize_references"]) {
                const w = widget(node, name);
                w.options = { ...w.options, surfaces: { canvas: "never", vueNode: "never", panel: "never" } };
                w.hidden = true; w.computeSize = () => [0, -4];
                if (w.element) w.element.style.display = "none";
            }
            const surface = buildSurface(node);
            node.addWidget("button", "Open Reference Studio", null, () => openStudio(node), { serialize: false });
            node.addDOMWidget("tm_reference_manager", "custom", surface.root, {
                serialize: false, hideOnZoom: false,
                getMinHeight: () => 450,
            });
            node.resizable = true;
            node.setSize([650, 650]);
            const configured = node.onConfigure;
            node.onConfigure = function () {
                configured?.apply(this, arguments);
                try { node._tmRefs = parseReferences(widget(node, "references_json").value); node._tmSelected.clear(); node._tmSelectionAnchor = null; render(node); }
                catch (error) { setStatus(node, error.message, true); }
            };
            const connected = node.onConnectionsChange;
            node.onConnectionsChange = function () { connected?.apply(this, arguments); render(node); };
            const executed = node.onExecuted;
            node.onExecuted = function (message) {
                executed?.apply(this, arguments);
                const previews = message.tm_reference_previews || [];
                const pending = node._tmPending;
                const matching = pending && previews.find(preview => previewKey(preview.source) === pending.key);
                for (const preview of previews) node._tmPreviews.set(previewKey(preview.source), preview);
                if (matching) { node._tmPending = null; pending.resolve(matching); }
                else if (message.tm_prompt) node._tmPromptResult = message.tm_prompt[0] || "";
                setStatus(node, "Reference preparation complete.");
                render(node);
            };
            const executionError = event => {
                const pending = node._tmPending;
                if (pending ? event.detail.prompt_id !== pending.promptId : String(event.detail.node_id) !== String(node.id)) return;
                const message = event.detail.exception_message || "Reference preparation failed. Check the execution error.";
                if (pending) failPreview(node, pending, message);
                else setStatus(node, message, true);
            };
            const interrupted = event => {
                const pending = node._tmPending;
                if (!pending || event.detail.prompt_id !== pending.promptId) return;
                node._tmPending = null; pending.reject(new Error("Preview interrupted."));
                setStatus(node, "Preview interrupted.");
            };
            const disconnected = () => failPreview(node, node._tmPending, "ComfyUI disconnected. Reconnect, then preview again.");
            const queueChanged = () => { void checkPreviewQueue(node); };
            api.addEventListener("reconnecting", disconnected);
            api.addEventListener("status", queueChanged);
            api.addEventListener("execution_error", executionError);
            api.addEventListener("execution_interrupted", interrupted);
            const removed = node.onRemoved;
            node.onRemoved = function () {
                api.removeEventListener("reconnecting", disconnected);
                api.removeEventListener("status", queueChanged);
                api.removeEventListener("execution_error", executionError);
                api.removeEventListener("execution_interrupted", interrupted);
                node._tmPending?.reject(new Error("Reference manager removed.")); node._tmPending = null;
                for (const modal of node._tmDialogs) modal.close();
                for (const view of node._tmSurfaces) view.root.querySelectorAll("video,audio").forEach(media => { media.pause(); media.removeAttribute("src"); media.load(); });
                node._tmSurfaces.clear(); node._tmPreviews.clear();
                removed?.apply(this, arguments);
            };
        };
    },
});
