// ============================================
// SIMPLIFIED EDITOR - Critical fixes applied
// ============================================

console.log('Editor starting...');

// Check if editorData exists
if (typeof editorData === 'undefined') {
    console.error('ERROR: editorData is not defined! Check Python script injection.');
    document.getElementById('status').textContent = 'ERROR: No data loaded';
    throw new Error('editorData is undefined');
}

console.log('editorData:', editorData);
console.log('Frames count:', editorData.frames?.length || 0);

// Basic scene setup
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.xr.enabled = true;
renderer.xr.setReferenceSpaceType('local-floor');
document.getElementById('container').appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a1a); // Slightly lighter for debugging
const clock = new THREE.Clock();

// Add axes helper for reference
const axesHelper = new THREE.AxesHelper(5);
scene.add(axesHelper);
console.log('Added axes helper');

const camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.1, 2000);
camera.up.set(0, 0, 1); // Z-up for LiDAR

const orbit = new OrbitControls(camera, renderer.domElement);
orbit.enableDamping = true;
orbit.dampingFactor = 0.05;

// REMOVED TransformControls - too complex, using keyboard-only editing
let transformMode = 'translate'; // 'translate' or 'scale'

const vrButton = VRButton.createButton(renderer);
document.body.appendChild(vrButton);

// UI elements
const statusEl = document.getElementById('status');
const frameLabelEl = document.getElementById('frame-label');
const labelSelectEl = document.getElementById('label-select');
const preserveViewEl = document.getElementById('preserve-view');
const legendEl = document.getElementById('legend');

// State
const state = {
    framesById: new Map(),
    order: [],
    currentFrameId: null,
    dirtyFrames: new Set(),
    cameraPosByFrame: new Map(),
    lastCameraPose: null,
    objects: {
        points: null,
        boxes: new Map(),
    },
    selection: null,
    selectedMesh: null,
    keys: { w: false, a: false, s: false, d: false, shift: false, up: false, down: false },
};

function setStatus(msg) {
    console.log('Status:', msg);
    statusEl.textContent = msg;
}

function buildLegend() {
    if (!editorData.labelColors) {
        console.warn('No labelColors in editorData');
        return;
    }
    legendEl.innerHTML = '<div><strong>Legend</strong></div>';
    Object.entries(editorData.labelColors).forEach(([label, color]) => {
        const r = Math.floor(color[0] * 255);
        const g = Math.floor(color[1] * 255);
        const b = Math.floor(color[2] * 255);
        const row = document.createElement('div');
        row.className = 'legend-item';
        const swatch = document.createElement('div');
        swatch.className = 'legend-color';
        swatch.style.backgroundColor = `rgb(${r},${g},${b})`;
        const text = document.createElement('div');
        text.textContent = label;
        row.appendChild(swatch);
        row.appendChild(text);
        legendEl.appendChild(row);
    });
}

function initFrames() {
    if (!editorData.frames || !Array.isArray(editorData.frames)) {
        console.error('No frames array in editorData');
        setStatus('ERROR: No frames data');
        return;
    }

    console.log('Initializing frames:', editorData.frames.length);

    editorData.frames.forEach((frame) => {
        if (!frame.frameId) {
            console.warn('Frame missing frameId:', frame);
            return;
        }
        state.framesById.set(frame.frameId, {
            ...frame,
            points: new Float32Array(frame.points || []),
            boxes: frame.boxes || [],
        });
        state.order.push(frame.frameId);
    });

    console.log('Loaded frames:', state.order.length);
    console.log('Frame IDs:', state.order);

    // Determine initial frame
    let initialId = null;
    if (editorData.initialFrameId && state.framesById.has(editorData.initialFrameId)) {
        initialId = editorData.initialFrameId;
    } else if (state.order.length > 0) {
        initialId = state.order[0];
    }

    console.log('Initial frame ID:', initialId);
    return initialId;
}

function populateLabelDropdown() {
    labelSelectEl.innerHTML = '';
    const labels = editorData.labels || Object.keys(editorData.labelColors || {});
    if (labels.length === 0) {
        console.warn('No labels found');
        return;
    }
    labels.forEach((label) => {
        const opt = document.createElement('option');
        opt.value = label;
        opt.textContent = label;
        labelSelectEl.appendChild(opt);
    });
}

function colorForLabel(label) {
    const c = (editorData.labelColors || {})[label] || [1, 1, 1];
    return new THREE.Color(c[0], c[1], c[2]);
}

function clearObjects() {
    if (state.objects.points) {
        scene.remove(state.objects.points);
        state.objects.points.geometry.dispose();
        state.objects.points.material.dispose();
        state.objects.points = null;
    }
    state.objects.boxes.forEach((group) => {
        scene.remove(group);
    });
    state.objects.boxes.clear();
    state.selection = null;
    state.selectedMesh = null;
}

function buildPoints(frame) {
    if (!frame.points || frame.points.length === 0) {
        console.warn('No points in frame:', frame.frameId);
        return;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(frame.points, 3));
    const material = new THREE.PointsMaterial({
        color: 0xffffff, // Changed to white for better visibility
        size: 0.05, // Increased size
        sizeAttenuation: true,
        opacity: 0.9,
        transparent: true,
    });
    const points = new THREE.Points(geometry, material);
    state.objects.points = points;
    scene.add(points);
    console.log('Added points:', frame.points.length / 3);
}

function createBoxGroup(box) {
    const geom = new THREE.BoxGeometry(1, 1, 1);
    const wireGeom = new THREE.EdgesGeometry(geom);
    const wireMat = new THREE.LineBasicMaterial({ color: colorForLabel(box.label) });
    const meshMat = new THREE.MeshBasicMaterial({
        color: 0x00ffff,
        wireframe: false,
        transparent: true,
        opacity: 0.01,
        depthWrite: false,
        side: THREE.DoubleSide
    });

    const mesh = new THREE.Mesh(geom, meshMat);
    mesh.scale.set(box.dims[0], box.dims[1], box.dims[2]);
    mesh.position.set(box.center[0], box.center[1], box.center[2]);
    mesh.rotation.z = box.yaw;
    mesh.userData.boxId = box.id;
    mesh.userData.label = box.label;
    mesh.userData.frameId = state.currentFrameId;

    const wire = new THREE.LineSegments(wireGeom, wireMat);
    wire.scale.copy(mesh.scale);
    wire.position.copy(mesh.position);
    wire.rotation.copy(mesh.rotation);
    wire.userData.boxId = box.id;
    wire.userData.frameId = state.currentFrameId;

    const group = new THREE.Group();
    group.add(mesh);
    group.add(wire);
    group.userData.boxId = box.id;
    group.userData.frameId = state.currentFrameId;
    group.userData.mesh = mesh;

    return { group, mesh, wire };
}

function highlightSelection(boxId) {
    state.objects.boxes.forEach((group, id) => {
        const isSelected = id === boxId;
        const color = colorForLabel(group.userData.label || 'Unknown');
        const targetColor = isSelected ? new THREE.Color(1, 0.5, 0) : color; // Orange for selection
        group.children.forEach((child) => {
            if (child.type === 'LineSegments') {
                child.material.color = targetColor.clone();
            }
        });
    });
}

function buildBoxes(frame) {
    if (!frame.boxes || frame.boxes.length === 0) {
        console.log('No boxes in frame:', frame.frameId);
        return;
    }

    console.log('Building boxes:', frame.boxes.length);
    frame.boxes.forEach((box) => {
        const { group } = createBoxGroup(box);
        group.userData.label = box.label;
        state.objects.boxes.set(box.id, group);
        scene.add(group);
    });
    highlightSelection(null);
}

function applyCameraPoseForFrame(frameId, frame) {
    // Only use saved pose if we're switching frames (not initial load)
    const isInitialLoad = state.currentFrameId === null;

    if (!isInitialLoad && preserveViewEl.checked && state.lastCameraPose) {
        camera.position.copy(state.lastCameraPose.position);
        camera.quaternion.copy(state.lastCameraPose.quaternion);
        if (state.lastCameraPose.target) {
            orbit.target.copy(state.lastCameraPose.target);
        }
        orbit.update();
        console.log('Applied saved camera pose');
        return;
    }

    if (!isInitialLoad && preserveViewEl.checked && state.cameraPosByFrame.has(frameId)) {
        const pose = state.cameraPosByFrame.get(frameId);
        camera.position.copy(pose.position);
        camera.quaternion.copy(pose.quaternion);
        if (pose.target) orbit.target.copy(pose.target);
        orbit.update();
        console.log('Applied frame-specific camera pose');
        return;
    }

    // Default: match visualization script behavior
    const c = frame.center || [0, 0, 0];
    const dist = frame.cameraDistance || 20;
    console.log('Setting default camera:', { center: c, distance: dist });

    // Match visualization script: center + distance * 0.7
    camera.position.set(
        c[0] + dist * 0.7,
        c[1] + dist * 0.7,
        c[2] + dist * 0.7
    );

    // Set target to LiDAR origin (0,0,0) like visualization script
    orbit.target.set(0, 0, 0);
    orbit.minDistance = 0.01;
    orbit.maxDistance = dist * 10;
    orbit.update();

    console.log('Camera position:', camera.position.toArray());
    console.log('Camera target:', orbit.target.toArray());
    console.log('Scene children:', scene.children.length);
}

function loadFrame(frameId) {
    console.log('Loading frame:', frameId);
    const frame = state.framesById.get(frameId);
    if (!frame) {
        setStatus(`Frame ${frameId} not found`);
        console.error('Frame not found:', frameId);
        return;
    }

    const isInitialLoad = state.currentFrameId === null;

    // Save current camera pose only if we're switching frames (not initial load)
    if (!isInitialLoad && state.currentFrameId !== null && preserveViewEl.checked) {
        state.cameraPosByFrame.set(state.currentFrameId, {
            position: camera.position.clone(),
            quaternion: camera.quaternion.clone(),
            target: orbit.target.clone(),
        });
        state.lastCameraPose = {
            position: camera.position.clone(),
            quaternion: camera.quaternion.clone(),
            target: orbit.target.clone(),
        };
    } else if (isInitialLoad) {
        // Clear any stale camera pose on initial load
        state.lastCameraPose = null;
    }

    clearObjects();
    state.currentFrameId = frameId;

    buildPoints(frame);
    buildBoxes(frame);
    applyCameraPoseForFrame(frameId, frame);

    const firstFrame = state.order[0];
    const lastFrame = state.order[state.order.length - 1];
    frameLabelEl.textContent = `Frame ${frameId}  (Range: ${firstFrame} - ${lastFrame})`;
    setStatus(`Loaded frame ${frameId} - ${frame.boxes?.length || 0} boxes`);
}

function nextFrame(step = 1) {
    if (state.order.length === 0) {
        setStatus('No frames available');
        return;
    }
    const idx = state.order.indexOf(state.currentFrameId);
    if (idx === -1) {
        loadFrame(state.order[0]);
        return;
    }
    const nextIdx = (idx + step + state.order.length) % state.order.length;
    loadFrame(state.order[nextIdx]);
}

function setMode(mode) {
    transformMode = mode;
    setStatus(`Mode: ${mode}`);
}

function selectBox(boxId) {
    const group = state.objects.boxes.get(boxId);
    if (!group) return;
    state.selection = boxId;
    const mesh = group.userData.mesh || group.children.find((c) => c.type === 'Mesh');
    state.selectedMesh = mesh;
    highlightSelection(boxId);
    const label = group.userData.label;
    if (labelSelectEl.value !== label) {
        labelSelectEl.value = label;
    }
    setStatus(`Selected box ${boxId}. Arrow keys to edit`);
}

function updateBoxStateFromMesh(mesh) {
    const frame = state.framesById.get(state.currentFrameId);
    if (!frame) return;
    const boxId = mesh.userData.boxId;
    const box = frame.boxes.find((b) => b.id === boxId);
    if (!box) return;

    box.center = [mesh.position.x, mesh.position.y, mesh.position.z];
    box.dims = [mesh.scale.x, mesh.scale.y, mesh.scale.z];
    box.yaw = mesh.rotation.z;

    const group = state.objects.boxes.get(boxId);
    if (group) {
        box.label = group.userData.label || box.label;
        group.children.forEach((child) => {
            if (child.type === 'LineSegments') {
                child.position.copy(mesh.position);
                child.scale.copy(mesh.scale);
                child.rotation.copy(mesh.rotation);
            }
        });
    }

    state.dirtyFrames.add(state.currentFrameId);
}

function onPointerDown(event) {
    event.preventDefault();
    if (!state.objects.boxes.size) return;
    const rect = renderer.domElement.getBoundingClientRect();
    const pointer = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1
    );
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(pointer, camera);
    const meshes = [];
    state.objects.boxes.forEach((group) => {
        const mesh = group.userData.mesh || group.children.find((c) => c.type === 'Mesh');
        if (mesh) meshes.push(mesh);
    });
    const hits = raycaster.intersectObjects(meshes, false);
    if (hits.length > 0) {
        selectBox(hits[0].object.userData.boxId);
    } else {
        // Unselect on click background
        if (state.selection) {
            state.selection = null;
            state.selectedMesh = null;
            highlightSelection(null);
            setStatus('Ready');
        }
    }
}

// Event listeners
renderer.domElement.addEventListener('pointerdown', onPointerDown);

document.getElementById('prev-frame').addEventListener('click', () => nextFrame(-1));
document.getElementById('next-frame').addEventListener('click', () => nextFrame(1));
document.getElementById('jump-back').addEventListener('click', () => nextFrame(-Math.max(1, editorData.window || 5)));
document.getElementById('jump-forward').addEventListener('click', () => nextFrame(Math.max(1, editorData.window || 5)));

document.getElementById('mode-move').addEventListener('click', () => setMode('translate'));
document.getElementById('mode-scale').addEventListener('click', () => setMode('scale'));

document.getElementById('save-label').addEventListener('click', () => {
    if (!state.selection) return;
    const frame = state.framesById.get(state.currentFrameId);
    if (!frame) return;
    const box = frame.boxes.find((b) => b.id === state.selection);
    if (!box) return;
    const newLabel = labelSelectEl.value;
    box.label = newLabel;
    const group = state.objects.boxes.get(state.selection);
    if (group) {
        group.userData.label = newLabel;
        const color = colorForLabel(newLabel);
        group.children.forEach((child) => {
            if (child.type === 'LineSegments') {
                child.material.color = color.clone();
            }
        });
    }
    state.dirtyFrames.add(state.currentFrameId);
    highlightSelection(state.selection);
});

function cycleLabel() {
    if (!state.selection) return;
    const labels = Array.from(labelSelectEl.options).map((o) => o.value);
    if (!labels.length) return;
    const current = labelSelectEl.value;
    const idx = labels.indexOf(current);
    const next = labels[(idx + 1) % labels.length];
    labelSelectEl.value = next;
    document.getElementById('save-label').click();
}

function exportEdits() {
    if (state.dirtyFrames.size === 0) {
        setStatus('No edits to export');
        return;
    }
    state.dirtyFrames.forEach((frameId) => {
        const frame = state.framesById.get(frameId);
        if (!frame) return;
        const lines = frame.boxes.map((b) => {
            return [
                b.label,
                b.center[0].toFixed(4),
                b.center[1].toFixed(4),
                b.center[2].toFixed(4),
                b.dims[0].toFixed(4),
                b.dims[1].toFixed(4),
                b.dims[2].toFixed(4),
                b.yaw.toFixed(4),
            ].join(' ');
        });
        const blob = new Blob([lines.join('\n')], { type: 'text/plain' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `frame_${frameId}_edited.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    });
    setStatus('Exported edited frames');
}

document.getElementById('export-edits').addEventListener('click', exportEdits);

window.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
        if (state.selection) {
            state.selection = null;
            state.selectedMesh = null;
            highlightSelection(null);
            setStatus('Ready');
        }
        return;
    }

    if (event.key === '/') event.preventDefault();

    if (event.key === 'm' || event.key === 'M') {
        setMode('translate');
    } else if (event.key === 'r' || event.key === 'R') {
        setMode('scale');
    } else if (event.key === 'l' || event.key === 'L') {
        cycleLabel();
    } else if (event.key === '[') {
        nextFrame(event.shiftKey ? -(Math.max(1, editorData.window || 5)) : -1);
    } else if (event.key === ']') {
        nextFrame(event.shiftKey ? Math.max(1, editorData.window || 5) : 1);
    } else if (event.key === 'ArrowLeft' && !state.selection) {
        nextFrame(-1);
    } else if (event.key === 'ArrowRight' && !state.selection) {
        nextFrame(1);
    }

    // Box editing
    if (state.selection && state.selectedMesh) {
        const step = event.shiftKey ? 0.5 : 0.1;
        const scaleStep = event.shiftKey ? 0.1 : 0.05;

        if (transformMode === 'translate') {
            if (event.key === 'ArrowLeft') {
                state.selectedMesh.position.x -= step;
                updateBoxStateFromMesh(state.selectedMesh);
                event.preventDefault();
            } else if (event.key === 'ArrowRight') {
                state.selectedMesh.position.x += step;
                updateBoxStateFromMesh(state.selectedMesh);
                event.preventDefault();
            } else if (event.key === 'ArrowUp') {
                if (event.ctrlKey || event.metaKey) {
                    state.selectedMesh.position.z += step;
                } else {
                    state.selectedMesh.position.y += step;
                }
                updateBoxStateFromMesh(state.selectedMesh);
                event.preventDefault();
            } else if (event.key === 'ArrowDown') {
                if (event.ctrlKey || event.metaKey) {
                    state.selectedMesh.position.z -= step;
                } else {
                    state.selectedMesh.position.y -= step;
                }
                updateBoxStateFromMesh(state.selectedMesh);
                event.preventDefault();
            }
        } else if (transformMode === 'scale') {
            if (event.key === 'ArrowLeft') {
                state.selectedMesh.scale.x = Math.max(0.1, state.selectedMesh.scale.x - scaleStep);
                updateBoxStateFromMesh(state.selectedMesh);
                event.preventDefault();
            } else if (event.key === 'ArrowRight') {
                state.selectedMesh.scale.x += scaleStep;
                updateBoxStateFromMesh(state.selectedMesh);
                event.preventDefault();
            } else if (event.key === 'ArrowUp') {
                if (event.ctrlKey || event.metaKey) {
                    state.selectedMesh.scale.z += scaleStep;
                } else {
                    state.selectedMesh.scale.y += scaleStep;
                }
                updateBoxStateFromMesh(state.selectedMesh);
                event.preventDefault();
            } else if (event.key === 'ArrowDown') {
                if (event.ctrlKey || event.metaKey) {
                    state.selectedMesh.scale.z = Math.max(0.1, state.selectedMesh.scale.z - scaleStep);
                } else {
                    state.selectedMesh.scale.y = Math.max(0.1, state.selectedMesh.scale.y - scaleStep);
                }
                updateBoxStateFromMesh(state.selectedMesh);
                event.preventDefault();
            }
        }
    }

    // WASD camera (only when no selection)
    if (!state.selection) {
        if (event.key === 'w' || event.key === 'W') state.keys.w = true;
        if (event.key === 'a' || event.key === 'A') state.keys.a = true;
        if (event.key === 's' || event.key === 'S') state.keys.s = true;
        if (event.key === 'd' || event.key === 'D') state.keys.d = true;
        if (event.key === 'Shift') state.keys.shift = true;
        if (event.key === 'q' || event.key === 'Q') state.keys.down = true;
        if (event.key === 'e' || event.key === 'E') state.keys.up = true;
    }
});

window.addEventListener('keyup', (event) => {
    if (event.key === 'w' || event.key === 'W') state.keys.w = false;
    if (event.key === 'a' || event.key === 'A') state.keys.a = false;
    if (event.key === 's' || event.key === 'S') state.keys.s = false;
    if (event.key === 'd' || event.key === 'D') state.keys.d = false;
    if (event.key === 'Shift') state.keys.shift = false;
    if (event.key === 'q' || event.key === 'Q') state.keys.down = false;
    if (event.key === 'e' || event.key === 'E') state.keys.up = false;
});

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

function updateWASD(delta) {
    if (state.selection) return;
    const speed = (state.keys.shift ? 8 : 3) * delta;
    if (!speed) return;
    const dir = new THREE.Vector3();
    camera.getWorldDirection(dir);
    dir.z = 0;
    dir.normalize();
    const right = new THREE.Vector3().crossVectors(dir, new THREE.Vector3(0, 0, 1)).normalize();
    const move = new THREE.Vector3();
    if (state.keys.w) move.add(dir);
    if (state.keys.s) move.sub(dir);
    if (state.keys.d) move.add(right);
    if (state.keys.a) move.sub(right);
    if (state.keys.up) move.z += 1;
    if (state.keys.down) move.z -= 1;
    if (move.lengthSq() > 0) {
        move.normalize().multiplyScalar(speed);
        camera.position.add(move);
        orbit.target.add(move);
        if (preserveViewEl.checked && state.currentFrameId !== null) {
            state.lastCameraPose = {
                position: camera.position.clone(),
                quaternion: camera.quaternion.clone(),
                target: orbit.target.clone(),
            };
        }
    }
}

let frameCount = 0;
function animate() {
    const delta = renderer.xr.isPresenting ? 0.016 : clock.getDelta();
    updateWASD(delta);
    orbit.update();
    renderer.render(scene, camera);

    // Debug: log first few frames
    frameCount++;
    if (frameCount <= 3) {
        console.log(`Render frame ${frameCount}:`, {
            sceneChildren: scene.children.length,
            cameraPos: camera.position,
            cameraTarget: orbit.target,
            points: state.objects.points ? 'yes' : 'no',
            boxes: state.objects.boxes.size
        });
    }
}
renderer.setAnimationLoop(animate);
console.log('Animation loop started');

// ============================================
// INITIALIZATION
// ============================================
console.log('Initializing editor...');

try {
    buildLegend();
    populateLabelDropdown();
    const initialId = initFrames();

    if (initialId != null) {
        console.log('Loading initial frame:', initialId);
        loadFrame(initialId);
    } else {
        setStatus('No frames to load');
        console.error('No frames available');
    }
} catch (error) {
    console.error('Initialization error:', error);
    setStatus(`ERROR: ${error.message}`);
}

console.log('Editor initialized');
