import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";

const COLORMAPS = {
  viridis: ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"],
  plasma: ["#0d0887", "#7e03a8", "#cc4778", "#f89540", "#f0f921"],
  coolwarm: ["#3b4cc0", "#8db0fe", "#dddddd", "#f4987a", "#b40426"],
  jet: ["#000080", "#0000ff", "#00ffff", "#ffff00", "#ff0000", "#800000"],
  gray: ["#000000", "#ffffff"],
};

function sample(name, t) {
  const stops = (COLORMAPS[name] || COLORMAPS.viridis).map((c) => new THREE.Color(c));
  const x = Math.min(Math.max(t, 0), 1) * (stops.length - 1);
  const i = Math.min(Math.floor(x), stops.length - 2);
  return stops[i].clone().lerp(stops[i + 1], x - i);
}

function unpack(a) {
  if (!a || !a.b64) return a;
  const bin = atob(a.b64);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return a.dtype === "uint32" ? new Uint32Array(buf.buffer) : new Float32Array(buf.buffer);
}

function range(layer, vals) {
  let lo = Infinity, hi = -Infinity;
  for (const v of vals) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  return [layer.vmin ?? lo, layer.vmax ?? hi];
}

function colorize(layer, vals, lo, hi) {
  const out = new Float32Array(vals.length * 3);
  for (let i = 0; i < vals.length; i++) {
    const c = sample(layer.colormap, hi > lo ? (vals[i] - lo) / (hi - lo) : 0);
    out.set([c.r, c.g, c.b], 3 * i);
  }
  return out;
}

// [roughness, metalness, environment reflection]; "off" is unlit
const LIGHTING = {
  default: [0.55, 0.05, 0], metallic: [0.4, 0.6, 0.8], plastic: [0.45, 0, 0.3],
  shiny: [0.2, 0.1, 0.6], glossy: [0.15, 0, 0.5], ambient: [1, 0, 0],
};

let envTexture = null;
function environment(renderer) {
  if (envTexture === null) {
    try {
      const pmrem = new THREE.PMREMGenerator(renderer);
      envTexture = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    } catch (e) { envTexture = false; }
  }
  return envTexture || null;
}

function surfaceMaterial(layer, opts, renderer) {
  if (layer.lighting === "off") return new THREE.MeshBasicMaterial(opts);
  const [roughness, metalness, env] = LIGHTING[layer.lighting] || LIGHTING.default;
  const m = new THREE.MeshStandardMaterial({ ...opts, roughness, metalness });
  if (env > 0 && (m.envMap = environment(renderer))) m.envMapIntensity = env;
  return m;
}

function loadTexture(layer, pending, redraw) {
  const tex = new THREE.Texture();
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  const img = new Image();
  pending.push(new Promise((resolve) => {
    img.onload = () => { tex.image = img; tex.needsUpdate = true; redraw(); resolve(); };
    img.onerror = resolve;
  }));
  img.src = layer.texture;
  return tex;
}

function makeMesh(layer, pos, idx, vals, lo, hi, renderer, pending, redraw) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  const opts = { side: THREE.DoubleSide, transparent: (layer.opacity ?? 1) < 1, opacity: layer.opacity ?? 1 };
  const uv = unpack(layer.uv);
  if (layer.texture && uv) {
    g.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
    opts.map = loadTexture(layer, pending, redraw);
  } else if (vals) {
    g.setAttribute("color", new THREE.BufferAttribute(colorize(layer, vals, lo, hi), 3));
    opts.vertexColors = true;
  } else opts.color = layer.color;
  g.computeVertexNormals();
  return new THREE.Mesh(g, surfaceMaterial(layer, opts, renderer));
}

function makeLines(layer, pos, idx, materials) {
  const flat = new Float32Array(idx.length * 3);
  for (let i = 0; i < idx.length; i++) flat.set(pos.subarray(3 * idx[i], 3 * idx[i] + 3), 3 * i);
  const g = new LineSegmentsGeometry();
  g.setPositions(flat);
  const m = new LineMaterial({
    color: layer.color, linewidth: layer.width ?? 2,
    transparent: (layer.opacity ?? 1) < 1, opacity: layer.opacity ?? 1,
  });
  materials.push(m);
  return new LineSegments2(g, m);
}

function makePoints(layer, pos, vals, lo, hi) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  const opts = { size: layer.size ?? 6, sizeAttenuation: false };
  if (vals) {
    g.setAttribute("color", new THREE.BufferAttribute(colorize(layer, vals, lo, hi), 3));
    opts.vertexColors = true;
  } else opts.color = layer.color;
  return new THREE.Points(g, new THREE.PointsMaterial(opts));
}

function mergeGeo(a, b) {
  const g = new THREE.BufferGeometry();
  const na = a.toNonIndexed(), nb = b.toNonIndexed();
  for (const k of ["position", "normal"]) {
    const arr = new Float32Array(na.attributes[k].array.length + nb.attributes[k].array.length);
    arr.set(na.attributes[k].array); arr.set(nb.attributes[k].array, na.attributes[k].array.length);
    g.setAttribute(k, new THREE.BufferAttribute(arr, 3));
  }
  return g;
}

function makeArrows(layer, pos, vec, diag) {
  const n = pos.length / 3;
  let maxLen = 0;
  const len = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    len[i] = Math.hypot(vec[3 * i], vec[3 * i + 1], vec[3 * i + 2]);
    maxLen = Math.max(maxLen, len[i]);
  }
  const scale = layer.scale ?? (maxLen > 0 ? (0.05 * diag) / maxLen : 1);
  // unit arrow along +y: shaft + head, total length 1
  const shaft = new THREE.CylinderGeometry(0.03, 0.03, 0.75, 8).translate(0, 0.375, 0);
  const head = new THREE.ConeGeometry(0.08, 0.25, 12).translate(0, 0.875, 0);
  const mesh = new THREE.InstancedMesh(mergeGeo(shaft, head), new THREE.MeshStandardMaterial({ roughness: 0.5 }), n);
  const shift = layer.to_origin ? -1 : 0;
  const [lo, hi] = range(layer, len);
  const up = new THREE.Vector3(0, 1, 0), d = new THREE.Vector3(), q = new THREE.Quaternion();
  const mat = new THREE.Matrix4(), s = new THREE.Vector3(), p = new THREE.Vector3();
  for (let i = 0; i < n; i++) {
    d.set(vec[3 * i], vec[3 * i + 1], vec[3 * i + 2]);
    if (len[i] > 0) q.setFromUnitVectors(up, d.normalize()); else q.identity();
    const L = len[i] * scale;
    s.set(L, L, L);  // L == 0 collapses the instance
    p.set(pos[3 * i] + shift * vec[3 * i] * scale, pos[3 * i + 1] + shift * vec[3 * i + 1] * scale, pos[3 * i + 2] + shift * vec[3 * i + 2] * scale);
    mesh.setMatrixAt(i, mat.compose(p, q, s));
    mesh.setColorAt(i, hi > lo ? sample(layer.colormap, (len[i] - lo) / (hi - lo)) : new THREE.Color(layer.color));
  }
  mesh.instanceMatrix.needsUpdate = true;
  return { object: mesh, lo, hi };
}

function colorbarEl(layer, lo, hi) {
  const bar = document.createElement("div");
  bar.style.cssText = "position:absolute;right:12px;top:12px;font:11px sans-serif;color:#333;text-align:right;pointer-events:none";
  const stops = (COLORMAPS[layer.colormap] || COLORMAPS.viridis).slice().reverse().join(",");
  bar.innerHTML = `<div>${hi.toPrecision(3)}</div><div style="width:14px;height:140px;margin:4px 0 4px auto;background:linear-gradient(${stops});border:1px solid #888"></div><div>${lo.toPrecision(3)}</div>`;
  return bar;
}

function axesGizmo(renderer, camera, target) {
  const scene = new THREE.Scene();
  scene.add(new THREE.AxesHelper(1));
  const cam = new THREE.OrthographicCamera(-1.4, 1.4, 1.4, -1.4, 0.1, 10);
  const label = (t, c, p) => {
    const cv = document.createElement("canvas"); cv.width = cv.height = 64;
    const ctx = cv.getContext("2d"); ctx.fillStyle = c; ctx.font = "bold 44px sans-serif";
    ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(t, 32, 34);
    const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(cv), depthTest: false }));
    s.position.copy(p).multiplyScalar(1.25); s.scale.setScalar(0.5); scene.add(s);
  };
  label("x", "#d62728", new THREE.Vector3(1, 0, 0));
  label("y", "#2ca02c", new THREE.Vector3(0, 1, 0));
  label("z", "#1f77b4", new THREE.Vector3(0, 0, 1));
  return () => {
    const size = 110, pad = 8;
    cam.position.copy(camera.position).sub(target).normalize().multiplyScalar(4);
    cam.up.copy(camera.up); cam.lookAt(0, 0, 0);
    const prev = renderer.autoClear;
    renderer.autoClear = false; renderer.clearDepth();
    renderer.setScissorTest(true);
    renderer.setViewport(pad, pad, size, size); renderer.setScissor(pad, pad, size, size);
    renderer.render(scene, cam);
    renderer.setScissorTest(false);
    const el = renderer.domElement;
    renderer.setViewport(0, 0, el.clientWidth, el.clientHeight);
    renderer.autoClear = prev;
  };
}

function button(text, css, onclick) {
  const b = document.createElement("button");
  b.textContent = text; b.className = "spv-ui";
  b.style.cssText = `position:absolute;font:11px sans-serif;padding:2px 6px;cursor:pointer;${css}`;
  b.onclick = onclick;
  return b;
}

async function renderGrid(el, scene) {
  const { ncols, link, titles } = scene.options;
  const n = scene.panels.length;
  el.style.display = "grid";
  el.style.gridAutoRows = "1fr";
  let enlarged = -1;      // index of the panel filling the grid, or -1
  // without ncols: the column count that gives panels the most room
  // (as 4:3 boxes) for the current size of the window
  const columns = () => {
    if (enlarged >= 0) return 1;
    if (ncols) return ncols;
    const W = el.clientWidth || 640, H = el.clientHeight || 480;
    let best = 1, bestArea = -1;
    for (let c = 1; c <= n; c++) {
      const w = W / c, h = H / Math.ceil(n / c), bw = Math.min(w, (h * 4) / 3);
      if (bw * bw > bestArea + 1e-6) { bestArea = bw * bw; best = c; }
    }
    return best;
  };
  const arrange = () => { el.style.gridTemplateColumns = `repeat(${columns()}, 1fr)`; };
  const layout = new ResizeObserver(arrange);
  layout.observe(el);
  arrange();
  const handles = [], cells = [], linked = [];
  let syncing = false;
  const cameraOf = (h) => ({ position: h.camera.position.toArray(), target: h.controls.target.toArray(), up: h.camera.up.toArray() });
  const enlarge = (i) => {
    enlarged = enlarged === i ? -1 : i;
    cells.forEach((c, j) => { c.style.display = enlarged < 0 || enlarged === j ? "" : "none"; });
    arrange();
    requestAnimationFrame(() => handles.forEach((h) => h.draw()));
  };
  for (const [i, panel] of scene.panels.entries()) {
    const cell = document.createElement("div");
    cell.style.cssText = "position:relative;min-width:0;min-height:0;border:1px solid #ddd;--spv-top:24px";
    el.appendChild(cell);
    cells.push(cell);
    linked.push(!!link);
    const h = await renderScene(cell, panel);
    const bar = document.createElement("div");
    bar.style.cssText = "position:absolute;left:0;right:0;top:0;height:22px;line-height:22px;text-align:center;font:600 12px/22px sans-serif;color:#333;background:#f3f3f3cc;border-bottom:1px solid #ddd;z-index:1;user-select:none";
    bar.textContent = (titles && titles[i]) || "";
    bar.ondblclick = () => enlarge(i);
    const tool = (text, tip, onclick) => {
      const b = document.createElement("span");
      b.textContent = text; b.title = tip;
      b.style.cssText = "position:absolute;top:0;cursor:pointer;padding:0 6px;font:12px/22px sans-serif";
      b.onclick = (e) => { e.stopPropagation(); onclick(b); };
      bar.appendChild(b);
      return b;
    };
    tool("\u2922", "Enlarge / restore this panel (double click the header)", () => enlarge(i)).style.right = "0";
    const lk = tool("", "Link this panel's camera to the other linked panels", () => {
      linked[i] = !linked[i];
      paint();
      if (linked[i]) {
        const src = handles.find((o, j) => j !== i && linked[j]);
        if (src) h.setCamera(cameraOf(src));
      }
    });
    lk.style.right = "24px";
    const paint = () => { lk.textContent = linked[i] ? "\u26d3 linked" : "\u26d3 free"; lk.style.opacity = linked[i] ? "1" : "0.45"; };
    paint();
    cell.appendChild(bar);
    handles.push(h);
    h.controls.addEventListener("change", () => {
      if (syncing || !linked[i]) return;
      syncing = true;
      const c = cameraOf(h);
      handles.forEach((o, j) => { if (j !== i && linked[j]) o.setCamera(c); });
      syncing = false;
    });
  }
  return {
    handles,
    setCamera: (c) => handles.forEach((h) => h.setCamera(c)),
    setScene: (s) => s.panels.forEach((p, i) => handles[i].setScene(p)),
    setUI: (on) => handles.forEach((h) => h.setUI(on)),
    draw: () => handles.forEach((h) => h.draw()),
    ready: () => Promise.all(handles.map((h) => h.ready())),
    dispose: () => { layout.disconnect(); handles.forEach((h) => h.dispose()); },
  };
}

export async function renderScene(el, scene) {
  if (scene.panels) return renderGrid(el, scene);
  el.style.position = "relative";
  const W = () => el.clientWidth || 640, H = () => el.clientHeight || 480;
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(W(), H());
  el.appendChild(renderer.domElement);

  const three = new THREE.Scene();
  three.add(new THREE.AmbientLight(0xffffff, 1.1));
  const headlight = new THREE.DirectionalLight(0xffffff, 2.2);
  three.add(headlight);
  const camera = new THREE.PerspectiveCamera(40, W() / H(), 0.01, 1000);
  const controls = new OrbitControls(camera, renderer.domElement);

  const tooltip = document.createElement("div");
  tooltip.style.cssText = "position:absolute;display:none;pointer-events:none;font:11px monospace;background:#222d;color:#fff;padding:3px 6px;border-radius:3px;white-space:pre";
  const pngButton = button("PNG", "right:12px;bottom:12px", () => {
    const a = document.createElement("a");
    a.href = api.toPNG(); a.download = "splinepy.png"; a.click();
  });
  el.append(tooltip, pngButton);

  let content = [];       // everything owned by the current scene
  let pickables = [];
  let labels = [];        // {div, group, positions}
  let materials = [];
  let pending = [];       // textures still loading
  let gizmo = null;
  let diag = 1, center = new THREE.Vector3(), first = true, uiOn = true, layersOpen = false;
  const raycaster = new THREE.Raycaster();

  function clear() {
    for (const o of content) {
      if (o.isObject3D) {
        three.remove(o);
        o.traverse((c) => { c.geometry?.dispose(); c.material?.dispose?.(); });
      } else o.remove();
    }
    content = []; pickables = []; labels = []; materials = []; pending = []; gizmo = null;
  }

  function setCamera(c) {
    if (c.position) camera.position.set(...c.position);
    if (c.target) controls.target.set(...c.target);
    if (c.up) camera.up.set(...c.up);
    if (c.fov) { camera.fov = c.fov; camera.updateProjectionMatrix(); }
    controls.update();
    draw();
  }

  let flat = false;
  const fitDistance = () => 0.55 * diag / Math.sin((camera.fov * Math.PI) / 360) * (camera.aspect < 1 ? 1 / camera.aspect : 1);
  // place camera on the side opposite to `dir` (view direction), keeping target at center
  function viewAlong(dir, up, fit = true) {
    const d = new THREE.Vector3(...dir).normalize();
    const dist = fit ? fitDistance() : camera.position.distanceTo(controls.target);
    if (fit) controls.target.copy(center);
    camera.up.set(...up);
    camera.position.copy(controls.target).addScaledVector(d, -dist);
    controls.update(); draw();
  }
  const AXES = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  function snapToAxis() {
    const d = controls.target.clone().sub(camera.position).normalize();
    const a = [Math.abs(d.x), Math.abs(d.y), Math.abs(d.z)];
    const i = a.indexOf(Math.max(...a));
    const dir = AXES[i].map((c) => c * Math.sign(d.getComponent(i)));
    // up: axis (other than view axis) closest to current up
    const u = camera.up;
    let best = -1, bi = 0;
    for (let j = 0; j < 3; j++) if (j !== i && Math.abs(u.getComponent(j)) > best) { best = Math.abs(u.getComponent(j)); bi = j; }
    viewAlong(dir, AXES[bi].map((c) => c * Math.sign(u.getComponent(bi) || 1)), false);
  }
  function fitView() {
    const d = camera.position.clone().sub(controls.target).normalize();
    controls.target.copy(center);
    camera.position.copy(center).addScaledVector(d, fitDistance());
    controls.update(); draw();
  }
  // hotkeys: 1/3/7 front/right/top (Ctrl: back/left/bottom), 0 iso, f fit, a snap
  function onKey(e) {
    if (e.altKey || e.metaKey) return;
    const s = e.ctrlKey ? -1 : 1;
    const k = e.key.toLowerCase();
    if (k === "1") viewAlong([0, s, 0], [0, 0, 1]);
    else if (k === "3") viewAlong([-s, 0, 0], [0, 0, 1]);
    else if (k === "7") viewAlong([0, 0, -s], [0, 1, 0]);
    else if (k === "0") { viewAlong([-1, -0.8, -1.4], flat ? [0, 1, 0] : [0, 1, 0]); }
    else if (k === "f" || k === "home") fitView();
    else if (k === "a") snapToAxis();
    else return;
    e.preventDefault();
  }
  el.tabIndex = 0; el.style.outline = "none";
  el.addEventListener("keydown", onKey);
  renderer.domElement.addEventListener("pointerenter", () => el.focus({ preventScroll: true }));

  function setScene(sc) {
    clear();
    const opt = sc.options;
    const [bmin, bmax] = opt.bounds.map((b) => new THREE.Vector3(...b));
    center = bmin.clone().add(bmax).multiplyScalar(0.5);
    diag = Math.max(bmax.clone().sub(bmin).length(), 1e-9);
    three.background = new THREE.Color(opt.background || "#ffffff");
    camera.near = diag * 1e-3; camera.far = diag * 100; camera.updateProjectionMatrix();

    const toggles = document.createElement("div");
    toggles.className = "spv-ui";
    toggles.style.cssText = "position:absolute;left:12px;top:calc(12px + var(--spv-top, 0px));font:12px sans-serif;color:#333;background:#fffc;padding:4px 8px;border-radius:4px";
    const body = document.createElement("div");
    body.style.display = layersOpen ? "" : "none";
    const head = document.createElement("div");
    head.style.cssText = "cursor:pointer;user-select:none";
    const paintHead = () => { head.textContent = `${layersOpen ? "\u25be" : "\u25b8"} Layers (${sc.layers.length})`; };
    head.onclick = () => { layersOpen = !layersOpen; body.style.display = layersOpen ? "" : "none"; paintHead(); };
    paintHead();
    toggles.append(head, body);
    let cbar = null;

    for (const layer of sc.layers) {
      const pos = unpack(layer.positions), idx = unpack(layer.indices), vals = unpack(layer.scalars);
      const [lo, hi] = vals ? range(layer, vals) : [0, 1];
      let obj;
      if (layer.kind === "mesh") {
        obj = makeMesh(layer, pos, idx, vals, lo, hi, renderer, pending, draw);
        obj.userData = { layer, pos, idx, vals };
        pickables.push(obj);
        if (vals && !layer.texture && !cbar) cbar = colorbarEl(layer, lo, hi);
      } else if (layer.kind === "lines") obj = makeLines(layer, pos, idx, materials);
      else if (layer.kind === "points") {
        obj = makePoints(layer, pos, vals, lo, hi);
        obj.userData = { layer, pos, vals };
        pickables.push(obj);
        if (vals && !cbar) cbar = colorbarEl(layer, lo, hi);
      } else if (layer.kind === "arrows") {
        const a = makeArrows(layer, pos, unpack(layer.vectors), diag);
        obj = a.object;
        if (!cbar && a.hi > a.lo) cbar = colorbarEl(layer, a.lo, a.hi);
      } else if (layer.kind === "labels") {
        obj = new THREE.Group();
        for (let i = 0; i < layer.text.length; i++) {
          const d = document.createElement("div");
          d.textContent = layer.text[i];
          d.style.cssText = `position:absolute;left:0;top:0;font:11px sans-serif;color:${layer.color || "#000"};pointer-events:none;transform:translate(-50%,-50%)`;
          el.appendChild(d); content.push(d);
          labels.push({ div: d, group: obj, p: new THREE.Vector3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]) });
        }
      }
      if (!obj) continue;
      if (layer.visible === false) obj.visible = false;
      three.add(obj); content.push(obj);
      const label = document.createElement("label");
      label.style.cssText = "display:block;cursor:pointer";
      label.innerHTML = `<input type="checkbox" ${obj.visible ? "checked" : ""}> ${layer.name}`;
      label.firstChild.onchange = (e) => { obj.visible = e.target.checked; draw(); };
      body.appendChild(label);
    }
    if (cbar) { el.appendChild(cbar); content.push(cbar); }
    if (sc.layers.length > 1) { el.appendChild(toggles); content.push(toggles); }
    gizmo = opt.axes ? axesGizmo(renderer, camera, controls.target) : null;

    flat = bmax.z - bmin.z < 1e-6 * diag;  // planar (2D) data lies in z = const
    if (first || opt.camera) {
      first = false;
      if (flat) viewAlong([0, 0, -1], [0, 1, 0]);  // look from +z downwards
      else {
        camera.position.copy(center).add(new THREE.Vector3(1, 0.8, 1.4).normalize().multiplyScalar(fitDistance()));
        controls.target.copy(center);
        camera.up.set(0, 1, 0);
        controls.update();
      }
      setCamera(opt.camera || {});
    }
    setUI(uiOn);
    draw();
  }

  function setUI(on) {
    uiOn = on;
    for (const e of el.querySelectorAll(".spv-ui")) e.style.display = on ? "" : "none";
    draw();
  }

  const v = new THREE.Vector3();
  function draw() {
    headlight.position.copy(camera.position);
    for (const m of materials) m.resolution.set(W(), H());
    renderer.render(three, camera);
    if (gizmo) gizmo();
    for (const l of labels) {
      v.copy(l.p).project(camera);
      const show = l.group.visible && v.z < 1 && Math.abs(v.x) < 1.1 && Math.abs(v.y) < 1.1;
      l.div.style.display = show ? "" : "none";
      if (show) l.div.style.transform = `translate(${(v.x * 0.5 + 0.5) * W()}px,${(-v.y * 0.5 + 0.5) * H()}px) translate(-50%,-50%)`;
    }
  }

  // hover readout: position, scalar (barycentric for meshes) or point index
  renderer.domElement.addEventListener("pointermove", (e) => {
    const r = renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    raycaster.setFromCamera(ndc, camera);
    raycaster.params.Points.threshold = diag * 0.01;
    const hit = raycaster.intersectObjects(pickables.filter((o) => o.visible), false)[0];
    if (!hit) { tooltip.style.display = "none"; return; }
    const { layer, pos, idx, vals } = hit.object.userData;
    let text = `${layer.name}\n(${hit.point.toArray().map((x) => x.toPrecision(4)).join(", ")})`;
    if (hit.face && vals) {
      const [a, b, c] = [hit.face.a, hit.face.b, hit.face.c];
      const P = (i) => new THREE.Vector3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
      const w = new THREE.Vector3();
      new THREE.Triangle(P(a), P(b), P(c)).getBarycoord(hit.point, w);
      text += `\nvalue ${(w.x * vals[a] + w.y * vals[b] + w.z * vals[c]).toPrecision(4)}`;
    } else if (hit.index !== undefined) {
      text += `\nid ${hit.index}` + (vals ? `\nvalue ${vals[hit.index].toPrecision(4)}` : "");
    }
    tooltip.textContent = text;
    tooltip.style.display = "block";
    tooltip.style.left = `${e.clientX - r.left + 12}px`;
    tooltip.style.top = `${e.clientY - r.top + 12}px`;
  });
  renderer.domElement.addEventListener("pointerleave", () => { tooltip.style.display = "none"; });

  controls.addEventListener("change", draw);
  const observer = new ResizeObserver(() => {
    renderer.setSize(W(), H());
    camera.aspect = W() / H(); camera.updateProjectionMatrix(); draw();
  });
  observer.observe(el);

  const api = {
    viewAlong, snapToAxis, fitView,
    renderer, camera, controls, draw, setScene, setCamera, setUI,
    ready: () => Promise.all(pending),
    dispose: () => {
      observer.disconnect(); clear();
      renderer.dispose(); renderer.forceContextLoss(); renderer.domElement.remove();
    },
    toPNG: () => (draw(), renderer.domElement.toDataURL("image/png")),
  };
  setScene(scene);
  return api;
}

globalThis.splinepyViewer = { renderScene };

// anywidget entry point
export default {
  async render({ model, el }) {
    el.style.height = `${model.get("height")}px`;
    let handle = await renderScene(el, JSON.parse(model.get("scene_json")));
    model.on("change:scene_json", () => handle.setScene(JSON.parse(model.get("scene_json"))));
  },
};
