// Scene setup
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a1a);

const camera = new THREE.PerspectiveCamera(
    75,
    window.innerWidth / window.innerHeight,
    0.1,
    10000
);
camera.up.set(0, 0, 1); // Set Z as up axis for LiDAR data

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.getElementById('container').appendChild(renderer.domElement);

// WebXR (VR)
// - Desktop usage is unchanged unless you explicitly enter a VR session.
renderer.xr.enabled = true;
renderer.xr.setReferenceSpaceType('local-floor');

// On-screen XR diagnostics (shows up in the existing info overlay)
const infoDiv = document.getElementById('info');
const xrStatusDiv = document.createElement('div');
xrStatusDiv.style.marginTop = '8px';
xrStatusDiv.style.fontSize = '12px';
xrStatusDiv.style.opacity = '0.9';
xrStatusDiv.innerHTML = '<strong>WebXR:</strong> checking…';
if (infoDiv) infoDiv.appendChild(xrStatusDiv);

async function updateXRStatus() {
    if (!('xr' in navigator)) {
        xrStatusDiv.innerHTML = '<strong>WebXR:</strong> navigator.xr not available (Quest Browser WebXR may be disabled)';
        return { hasXR: false, immersiveVR: false };
    }
    try {
        const supported = await navigator.xr.isSessionSupported('immersive-vr');
        xrStatusDiv.innerHTML = `<strong>WebXR:</strong> immersive-vr supported = ${supported ? 'YES' : 'NO'}`;
        return { hasXR: true, immersiveVR: supported };
    } catch (e) {
        xrStatusDiv.innerHTML = `<strong>WebXR:</strong> isSessionSupported error: ${e && e.name ? e.name : 'Error'} ${e && e.message ? e.message : ''}`;
        return { hasXR: true, immersiveVR: false };
    }
}

// Standard VRButton (if available)
if (typeof VRButton !== 'undefined' && VRButton && VRButton.createButton) {
    const vrBtn = VRButton.createButton(renderer);
    document.body.appendChild(vrBtn);
} else {
    console.warn("VRButton not available; WebXR button will not be shown.");
}

// Fallback "Enter VR (debug)" button that directly requests an XR session and
// prints the exact error if the browser blocks it.
const debugBtn = document.createElement('button');
debugBtn.textContent = 'Enter VR (debug)';
debugBtn.style.position = 'absolute';
debugBtn.style.left = '10px';
debugBtn.style.bottom = '10px';
debugBtn.style.zIndex = '101';
debugBtn.style.padding = '8px 10px';
debugBtn.style.borderRadius = '6px';
debugBtn.style.border = '1px solid rgba(255,255,255,0.4)';
debugBtn.style.background = 'rgba(0,0,0,0.6)';
debugBtn.style.color = 'white';
debugBtn.style.cursor = 'pointer';
debugBtn.title = 'If this fails, the error text explains why immersive VR is blocked.';
document.body.appendChild(debugBtn);

debugBtn.addEventListener('click', async () => {
    const st = await updateXRStatus();
    if (!st.hasXR) return;
    try {
        const sessionInit = {
            optionalFeatures: [
                'local-floor',
                'bounded-floor',
                'hand-tracking',
                'layers',
            ],
        };
        const session = await navigator.xr.requestSession('immersive-vr', sessionInit);
        await renderer.xr.setSession(session);
        xrStatusDiv.innerHTML = '<strong>WebXR:</strong> XR session started';
        session.addEventListener('end', () => {
            xrStatusDiv.innerHTML = '<strong>WebXR:</strong> XR session ended';
        });
    } catch (e) {
        xrStatusDiv.innerHTML = `<strong>WebXR:</strong> requestSession failed: ${e && e.name ? e.name : 'Error'} ${e && e.message ? e.message : ''}`;
        console.error("requestSession failed", e);
    }
});

// Kick off initial status check
updateXRStatus();

// Lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambientLight);
const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
directionalLight.position.set(1, 1, 1);
scene.add(directionalLight);

// Point cloud
// Expecting global variable 'pointsData' (flat array of coordinates)
const pointsGeometry = new THREE.BufferGeometry();
const pointsArray = new Float32Array(pointsData);
pointsGeometry.setAttribute('position', new THREE.BufferAttribute(pointsArray, 3));

// Create a circular texture for points
function createCircleTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 32;
    canvas.height = 32;
    const ctx = canvas.getContext('2d');
    
    ctx.beginPath();
    ctx.arc(16, 16, 15, 0, 2 * Math.PI);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
    
    return new THREE.CanvasTexture(canvas);
}

const pointsMaterial = new THREE.PointsMaterial({
    color: 0x999999,
    size: 0.02, // Even smaller points
    map: createCircleTexture(), // Make points round
    alphaTest: 0.5,
    sizeAttenuation: true,
    opacity: 0.8,
    transparent: true
});

const pointsCloud = new THREE.Points(pointsGeometry, pointsMaterial);
scene.add(pointsCloud);

// Bounding boxes
// Expecting global variable 'boxesData' (object with label -> {lines, color})
const numBoxes = Object.keys(boxesData).length;
console.log("Rendering boxes:", numBoxes, "labels found");

if (numBoxes === 0) {
    if (infoDiv) {
        const warning = document.createElement('div');
        warning.style.color = '#ffaa00';
        warning.style.marginTop = '10px';
        warning.innerHTML = '<strong>Warning: No Ground Truth Boxes Found</strong>';
        infoDiv.appendChild(warning);
    }
} else {
    // Populate Legend
    const legendDiv = document.getElementById('legend');
    if (legendDiv) {
        legendDiv.innerHTML = '<div><strong>Legend</strong></div>';
        for (const [label, boxData] of Object.entries(boxesData)) {
            if (boxData.lines.length === 0) continue;
            
            const color = boxData.color;
            // Convert [0-1] RGB float array to CSS color string
            const r = Math.floor(color[0] * 255);
            const g = Math.floor(color[1] * 255);
            const b = Math.floor(color[2] * 255);
            const cssColor = `rgb(${r}, ${g}, ${b})`;
            
            const item = document.createElement('div');
            item.className = 'legend-item';
            
            const colorBox = document.createElement('div');
            colorBox.className = 'legend-color';
            colorBox.style.backgroundColor = cssColor;
            
            const labelText = document.createElement('div');
            labelText.innerText = label + ` (${boxData.lines.length})`;
            
            item.appendChild(colorBox);
            item.appendChild(labelText);
            legendDiv.appendChild(item);
        }
    }
}

for (const [label, boxData] of Object.entries(boxesData)) {
    const lines = boxData.lines;
    const color = boxData.color;
    console.log(`Label ${label}: ${lines.length} lines`);
    
    if (lines.length === 0) continue;

    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(lines.flat());
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    
    const material = new THREE.LineBasicMaterial({
        color: new THREE.Color(color[0], color[1], color[2]),
        linewidth: 2 // Note: linewidth > 1 often ignored by browsers due to OpenGL core profile
    });
    
    const lineSegments = new THREE.LineSegments(geometry, material);
    scene.add(lineSegments);
}

// Camera positioning
// Expecting global variables 'center' ([x,y,z]) and 'cameraDistance' (number)
camera.position.set(
    center[0] + cameraDistance * 0.7,
    center[1] + cameraDistance * 0.7,
    center[2] + cameraDistance * 0.7
);

// Controls Setup
const clock = new THREE.Clock();
let isFlyMode = false;

// Orbit Controls (Default)
const orbitControls = new THREE.OrbitControls(camera, renderer.domElement);
// Set target to (0,0,0) - the LiDAR sensor origin - instead of geometric center of points
orbitControls.target.set(0, 0, 0); 
orbitControls.enableDamping = true;
orbitControls.dampingFactor = 0.05;
orbitControls.minDistance = 0.01; // Allow very close zoom
orbitControls.maxDistance = cameraDistance * 10; // Allow far zoom out
orbitControls.enablePan = true;
orbitControls.panSpeed = 0.8;
orbitControls.rotateSpeed = 0.8;
orbitControls.zoomSpeed = 1.2;

// Fly Controls
const flyControls = new THREE.FlyControls(camera, renderer.domElement);
flyControls.movementSpeed = cameraDistance * 0.5; // Adjust speed relative to scene size
flyControls.domElement = renderer.domElement;
flyControls.rollSpeed = Math.PI / 6;
flyControls.autoForward = false;
flyControls.dragToLook = true;
flyControls.enabled = false; // Start disabled

// XR rig + controllers
// We move a "rig" Group for teleport / locomotion (camera pose comes from headset).
const xrRig = new THREE.Group();
xrRig.add(camera);
scene.add(xrRig);

// Simple ground plane (invisible) used for teleport raycasts.
// Note: Z is up in this viewer, so a plane in XY at z=0 is a "ground" plane.
const xrGround = new THREE.Mesh(
    new THREE.PlaneGeometry(2000, 2000),
    new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.0, side: THREE.DoubleSide })
);
xrGround.position.set(0, 0, 0);
scene.add(xrGround);

const raycaster = new THREE.Raycaster();
const _tmpMatrix = new THREE.Matrix4();

function addXRController(index) {
    const controller = renderer.xr.getController(index);
    controller.userData.index = index;
    controller.userData.isSelecting = false;

    // Ray line (controller forward is -Z in local space)
    const lineGeom = new THREE.BufferGeometry();
    lineGeom.setAttribute(
        'position',
        new THREE.Float32BufferAttribute([0, 0, 0, 0, 0, -1], 3)
    );
    const lineMat = new THREE.LineBasicMaterial({ color: 0xffffff });
    const line = new THREE.Line(lineGeom, lineMat);
    line.name = 'xr-ray';
    line.scale.z = 20; // ray length
    controller.add(line);

    controller.addEventListener('selectstart', () => { controller.userData.isSelecting = true; });
    controller.addEventListener('selectend', () => {
        controller.userData.isSelecting = false;

        // Teleport: cast ray to ground plane and move rig there.
        const hit = intersectGround(controller);
        if (hit) {
            // Move rig to target (keep z at current rig height offset)
            xrRig.position.x = hit.point.x;
            xrRig.position.y = hit.point.y;
            // Keep existing vertical offset to avoid snapping eye height
        }
    });

    xrRig.add(controller);

    // Controller grip with model (visual controller)
    if (typeof XRControllerModelFactory !== 'undefined') {
        const controllerGrip = renderer.xr.getControllerGrip(index);
        const factory = new XRControllerModelFactory();
        controllerGrip.add(factory.createControllerModel(controllerGrip));
        xrRig.add(controllerGrip);
    }

    return controller;
}

function intersectGround(controller) {
    _tmpMatrix.identity().extractRotation(controller.matrixWorld);
    raycaster.ray.origin.setFromMatrixPosition(controller.matrixWorld);
    raycaster.ray.direction.set(0, 0, -1).applyMatrix4(_tmpMatrix);
    const hits = raycaster.intersectObject(xrGround, false);
    if (hits && hits.length > 0) return hits[0];
    return null;
}

const xrController1 = addXRController(0);
const xrController2 = addXRController(1);

function setDesktopControlsEnabled(enabled) {
    orbitControls.enabled = enabled;
    // Fly controls are mutually exclusive with orbit (keep previous mode if needed)
    if (!enabled) {
        flyControls.enabled = false;
        isFlyMode = false;
    }
}

renderer.xr.addEventListener('sessionstart', () => {
    setDesktopControlsEnabled(false);
    console.log("XR session started");
});
renderer.xr.addEventListener('sessionend', () => {
    setDesktopControlsEnabled(true);
    console.log("XR session ended");
});

function toggleControls() {
    isFlyMode = !isFlyMode;
    
    if (isFlyMode) {
        orbitControls.enabled = false;
        flyControls.enabled = true;
        // Sync FlyControls position/rotation if needed (they share the camera)
        console.log("Switched to Fly Mode");
    } else {
        flyControls.enabled = false;
        orbitControls.enabled = true;
        // Reset orbit target to slightly in front of camera to avoid disorientation?
        // Or keep original target. Keeping original target is safer for now.
        console.log("Switched to Orbit Mode");
    }
}

// Mouse/Key listeners for interaction
// Robust Ctrl+Click handling for Panning
renderer.domElement.addEventListener('mousedown', function(event) {
    if (!orbitControls.enabled) return;
    
    // Check Ctrl key state explicitly on mouse down
    if (event.ctrlKey) {
        orbitControls.mouseButtons.LEFT = THREE.MOUSE.PAN;
    } else {
        orbitControls.mouseButtons.LEFT = THREE.MOUSE.ROTATE;
    }
}, true); // Use capture phase to ensure this runs before OrbitControls

// Also listen to Keydown/Keyup for UX (e.g. if user presses key before clicking)
window.addEventListener('keydown', function(event) {
    if (event.key === 'c' || event.key === 'C') {
        toggleControls();
    }
    if (event.key === 'r' || event.key === 'R') {
        // Reset view to look at origin (LiDAR center)
        if (orbitControls.enabled) {
            orbitControls.target.set(0, 0, 0);
            camera.position.set(
                center[0] + cameraDistance * 0.7,
                center[1] + cameraDistance * 0.7,
                center[2] + cameraDistance * 0.7
            );
            orbitControls.update();
            console.log("Reset view to origin");
        }
    }
    if (event.key === 'Control') {
        if (orbitControls.enabled) {
            orbitControls.mouseButtons.LEFT = THREE.MOUSE.PAN;
        }
    }
});

window.addEventListener('keyup', function(event) {
    if (event.key === 'Control') {
        if (orbitControls.enabled) {
            orbitControls.mouseButtons.LEFT = THREE.MOUSE.ROTATE;
        }
    }
});

orbitControls.update();

// Animation loop (works for both desktop and WebXR)
function animate() {
    const delta = clock.getDelta();

    // Desktop controls only when not in XR
    if (!renderer.xr.isPresenting) {
        if (isFlyMode) {
            flyControls.update(delta);
        } else {
            orbitControls.update();
        }
    }

    // In XR we keep the controller rays visible; teleport is handled on selectend.
    renderer.render(scene, camera);
}

// Handle window resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

renderer.setAnimationLoop(animate);
