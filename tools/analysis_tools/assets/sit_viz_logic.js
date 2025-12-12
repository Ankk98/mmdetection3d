// Scene setup
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a1a);

const camera = new THREE.PerspectiveCamera(
    75,
    window.innerWidth / window.innerHeight,
    0.1,
    10000
);

// LiDAR data is Z-up, but WebXR expects Y-up.
// We'll put all content in a container that we can rotate for VR.
const sceneContainer = new THREE.Group();
scene.add(sceneContainer);

// For desktop viewing, set camera up to Z
camera.up.set(0, 0, 1);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.getElementById('container').appendChild(renderer.domElement);

// WebXR (VR)
// - Desktop usage is unchanged unless you explicitly enter a VR session.
renderer.xr.enabled = true;
renderer.xr.setReferenceSpaceType('local-floor');
if (renderer.xr.setFoveation) renderer.xr.setFoveation(1.0);

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

// Standard VRButton (automatically detects WebXR support and shows button or error)
console.log('[VRButton] Attempting to create VRButton...');
console.log('[VRButton] VRButton available:', typeof VRButton !== 'undefined');
console.log('[VRButton] renderer.xr.enabled:', renderer.xr.enabled);

try {
    const vrButton = VRButton.createButton(renderer);
    console.log('[VRButton] Button created:', vrButton);
    console.log('[VRButton] Button text:', vrButton.textContent);
    console.log('[VRButton] Button style.display:', vrButton.style.display);
    document.body.appendChild(vrButton);
    console.log('[VRButton] Button appended to body successfully');
} catch (e) {
    console.error("[VRButton] Failed to create button:", e);
    xrStatusDiv.innerHTML += '<br><span style="color: #ff6666;">VRButton creation failed - see console</span>';
}

// Enter VR button that directly requests an XR session and prints the exact
// error if the browser blocks it (useful on Quest).
const debugBtn = document.createElement('button');
debugBtn.textContent = 'Enter VR';
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
    if (!st.immersiveVR) {
        xrStatusDiv.innerHTML = '<strong>WebXR:</strong> immersive-vr not supported/allowed (check https + WebXR settings)';
        return;
    }
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
updateXRStatus().then(status => {
    console.log('[WebXR] Initial status check:', status);
    
    // Add a visible WebXR status indicator
    const statusIndicator = document.createElement('div');
    statusIndicator.style.position = 'fixed';
    statusIndicator.style.top = '50%';
    statusIndicator.style.left = '50%';
    statusIndicator.style.transform = 'translate(-50%, -50%)';
    statusIndicator.style.background = 'rgba(0, 0, 0, 0.9)';
    statusIndicator.style.color = 'white';
    statusIndicator.style.padding = '20px';
    statusIndicator.style.borderRadius = '10px';
    statusIndicator.style.zIndex = '1000';
    statusIndicator.style.maxWidth = '80%';
    statusIndicator.style.textAlign = 'center';
    statusIndicator.style.fontFamily = 'sans-serif';
    
    if (!status.hasXR) {
        statusIndicator.innerHTML = `
            <h3>⚠️ WebXR Not Available</h3>
            <p>navigator.xr is not available in this browser.</p>
            <p><strong>On Quest:</strong> Use Quest Browser and enable WebXR in chrome://flags</p>
            <button onclick="this.parentElement.remove()" style="margin-top: 10px; padding: 8px 16px; cursor: pointer;">Close</button>
        `;
        document.body.appendChild(statusIndicator);
    } else if (!status.immersiveVR) {
        statusIndicator.innerHTML = `
            <h3>⚠️ WebXR Available but Immersive VR Not Supported</h3>
            <p>navigator.xr exists but immersive-vr is not supported.</p>
            <p><strong>Common fixes:</strong></p>
            <ul style="text-align: left;">
                <li>Ensure you're using HTTPS (not HTTP)</li>
                <li>Enable WebXR in chrome://flags on Quest Browser</li>
                <li>Check if site is marked as VR-enabled in browser permissions</li>
            </ul>
            <button onclick="this.parentElement.remove()" style="margin-top: 10px; padding: 8px 16px; cursor: pointer;">Close</button>
        `;
        document.body.appendChild(statusIndicator);
    } else {
        // Success - briefly show and auto-hide
        statusIndicator.innerHTML = `
            <h3>✅ WebXR Ready</h3>
            <p>immersive-vr is supported!</p>
            <p>Look for the VR button to enter VR mode.</p>
        `;
        document.body.appendChild(statusIndicator);
        setTimeout(() => statusIndicator.remove(), 3000);
    }
});

// Lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
sceneContainer.add(ambientLight);
const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
directionalLight.position.set(1, 1, 1);
sceneContainer.add(directionalLight);

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
sceneContainer.add(pointsCloud);

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
    sceneContainer.add(lineSegments);
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
const orbitControls = new OrbitControls(camera, renderer.domElement);
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
const flyControls = new FlyControls(camera, renderer.domElement);
flyControls.movementSpeed = cameraDistance * 0.5; // Adjust speed relative to scene size
flyControls.domElement = renderer.domElement;
flyControls.rollSpeed = Math.PI / 6;
flyControls.autoForward = false;
flyControls.dragToLook = true;
flyControls.enabled = false; // Start disabled

// XR rig + controllers/hands
// We move a "rig" Group for teleport / locomotion (camera pose comes from headset).
const xrRig = new THREE.Group();
xrRig.add(camera);
scene.add(xrRig);

// Ground plane for teleport (1.6m below LiDAR origin = typical floor level)
// LiDAR sensor at (0,0,0) is at eye height, so ground is at y=-1.6
const xrGround = new THREE.Mesh(
    new THREE.PlaneGeometry(2000, 2000),
    new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.0, side: THREE.DoubleSide })
);
xrGround.rotation.x = -Math.PI / 2; // Rotate to horizontal (XZ plane)
xrGround.position.set(0, -1.6, 0); // Ground 1.6m below sensor
scene.add(xrGround);

// Teleport target marker (shows where you'll land)
const teleportMarker = new THREE.Mesh(
    new THREE.RingGeometry(0.2, 0.3, 32),
    new THREE.MeshBasicMaterial({ color: 0x00ff00, side: THREE.DoubleSide, transparent: true, opacity: 0.7 })
);
teleportMarker.rotation.x = -Math.PI / 2;
teleportMarker.visible = false;
scene.add(teleportMarker);

// Optional: grid helper at ground level for better spatial reference in VR
const gridHelper = new THREE.GridHelper(100, 50, 0x444444, 0x222222);
gridHelper.position.y = -1.6; // Ground is 1.6m below origin (typical eye height)
gridHelper.visible = false; // Will be shown only in VR
scene.add(gridHelper);

// LiDAR sensor origin indicator (small axes at 0,0,0 - where sensor was)
const axesHelper = new THREE.AxesHelper(0.5);
axesHelper.visible = false;
scene.add(axesHelper);

// Add a glowing sphere at origin to mark LiDAR sensor position
const originMarker = new THREE.Mesh(
    new THREE.SphereGeometry(0.08, 16, 16),
    new THREE.MeshBasicMaterial({ 
        color: 0x00ffff,
        transparent: true,
        opacity: 0.8
    })
);
originMarker.visible = false;
scene.add(originMarker);

// Add a label for the origin
function createOriginLabel() {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 128;
    const ctx = canvas.getContext('2d');
    
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(0, 0, 512, 128);
    
    ctx.fillStyle = '#00ffff';
    ctx.font = 'bold 40px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('LiDAR Sensor Origin', 256, 40);
    ctx.font = '32px Arial';
    ctx.fillStyle = '#ffffff';
    ctx.fillText('(0, 0, 0)', 256, 85);
    
    const texture = new THREE.CanvasTexture(canvas);
    const geometry = new THREE.PlaneGeometry(0.5, 0.125);
    const material = new THREE.MeshBasicMaterial({ 
        map: texture, 
        transparent: true,
        side: THREE.DoubleSide
    });
    
    const label = new THREE.Mesh(geometry, material);
    label.position.set(0, 0.15, 0);
    label.visible = false;
    return label;
}

const originLabel = createOriginLabel();
scene.add(originLabel);

// VR Controls UI Panel (created with canvas texture)
function createVRControlsPanel() {
    const canvas = document.createElement('canvas');
    canvas.width = 1024;
    canvas.height = 512;
    const ctx = canvas.getContext('2d');
    
    // Background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
    ctx.fillRect(0, 0, 1024, 512);
    
    // Border
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 4;
    ctx.strokeRect(10, 10, 1004, 492);
    
    // Title
    ctx.fillStyle = '#00ff00';
    ctx.font = 'bold 56px Arial';
    ctx.textAlign = 'left';
    ctx.fillText('VR Controls', 40, 80);
    
    // Controls list
    ctx.fillStyle = '#ffffff';
    ctx.font = '36px Arial';
    let y = 150;
    const controls = [
        '🎯 Trigger: Point at ground → Teleport',
        '🕹️  Left Stick: Walk/Strafe',
        '⬆️  Left Grip + Stick Up/Down: Height Adjust',
        '🎮 Right Stick: Look Around (Yaw/Pitch)',
        '🤏 Right Grip: Toggle Help Panels',
    ];
    
    controls.forEach(text => {
        ctx.fillText(text, 60, y);
        y += 70;
    });
    
    const texture = new THREE.CanvasTexture(canvas);
    const geometry = new THREE.PlaneGeometry(2, 1);
    const material = new THREE.MeshBasicMaterial({ 
        map: texture, 
        transparent: true,
        side: THREE.DoubleSide
    });
    
    const panel = new THREE.Mesh(geometry, material);
    panel.visible = false;
    return panel;
}

const controlsPanel = createVRControlsPanel();
// Position panel 2m in front, slightly above eye level, and to the left
controlsPanel.position.set(-0.75, 0.4, -2);
xrRig.add(controlsPanel); // Attach to rig so it follows the user

// Create VR Legend Panel showing box colors/classes
function createVRLegendPanel() {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext('2d');
    
    // Background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
    ctx.fillRect(0, 0, 512, 512);
    
    // Border
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 3;
    ctx.strokeRect(8, 8, 496, 496);
    
    // Title
    ctx.fillStyle = '#00ff00';
    ctx.font = 'bold 40px Arial';
    ctx.textAlign = 'left';
    ctx.fillText('Legend', 30, 50);
    
    // Draw legend items
    let y = 100;
    const entries = Object.entries(boxesData);
    
    if (entries.length === 0) {
        ctx.fillStyle = '#ffffff';
        ctx.font = '28px Arial';
        ctx.fillText('No boxes to display', 30, y);
    } else {
        for (const [label, boxData] of entries) {
            if (boxData.lines.length === 0) continue;
            
            const color = boxData.color;
            const r = Math.floor(color[0] * 255);
            const g = Math.floor(color[1] * 255);
            const b = Math.floor(color[2] * 255);
            
            // Color box
            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            ctx.fillRect(30, y - 20, 35, 35);
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1;
            ctx.strokeRect(30, y - 20, 35, 35);
            
            // Label text
            ctx.fillStyle = '#ffffff';
            ctx.font = '28px Arial';
            ctx.fillText(`${label} (${boxData.lines.length})`, 80, y + 5);
            
            y += 55;
        }
    }
    
    const texture = new THREE.CanvasTexture(canvas);
    const geometry = new THREE.PlaneGeometry(1, 1);
    const material = new THREE.MeshBasicMaterial({ 
        map: texture, 
        transparent: true,
        side: THREE.DoubleSide
    });
    
    const panel = new THREE.Mesh(geometry, material);
    panel.visible = false;
    return panel;
}

const vrLegendPanel = createVRLegendPanel();
// Position legend to the right of controls (with spacing to avoid overlap)
vrLegendPanel.position.set(0.75, 0.4, -2);
xrRig.add(vrLegendPanel);

// Create small button label helpers for controllers
function createButtonLabel(text, color = '#00ff00') {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    
    ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
    ctx.fillRect(0, 0, 256, 64);
    
    ctx.fillStyle = color;
    ctx.font = 'bold 24px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 128, 32);
    
    const texture = new THREE.CanvasTexture(canvas);
    const geometry = new THREE.PlaneGeometry(0.12, 0.03);
    const material = new THREE.MeshBasicMaterial({ 
        map: texture, 
        transparent: true,
        side: THREE.DoubleSide
    });
    
    return new THREE.Mesh(geometry, material);
}

const raycaster = new THREE.Raycaster();
const _tmpMatrix = new THREE.Matrix4();

const controllerModelFactory = new XRControllerModelFactory();
const handModelFactory = new XRHandModelFactory();

function addXRController(index) {
    const controller = renderer.xr.getController(index);
    controller.userData.index = index;
    controller.userData.isSelecting = false;
    controller.userData.gamepad = null;
    controller.userData.handedness = null;
    controller.userData.inputSource = null;

    controller.addEventListener('connected', (event) => {
        // event.data is the XRInputSource
        controller.userData.inputSource = event.data ?? null;
        controller.userData.gamepad = event.data?.gamepad ?? null;
        controller.userData.handedness = event.data?.handedness ?? null;
    });
    controller.addEventListener('disconnected', () => {
        controller.userData.inputSource = null;
        controller.userData.gamepad = null;
        controller.userData.handedness = null;
    });

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

    controller.addEventListener('selectstart', () => { 
        controller.userData.isSelecting = true;
    });
    
    controller.addEventListener('selectend', () => {
        controller.userData.isSelecting = false;
        teleportMarker.visible = false;

        // Teleport: cast ray to ground plane and move rig there.
        const hit = intersectGround(controller);
        if (hit) {
            // Move rig to target in XZ plane (horizontal), keep Y (height) constant
            xrRig.position.x = hit.point.x;
            xrRig.position.z = hit.point.z;
            // Y (vertical) stays at current user height
        }
    });
    
    // Squeeze (grip) button on RIGHT controller toggles help panels
    // (Left grip is used for height adjustment)
    controller.addEventListener('squeezestart', () => {
        // Only toggle panels if this is the right controller
        const isRightController = controller.userData.handedness === 'right' ||
                                   (controller.userData.handedness !== 'left' && controller.userData.index === 1);
        
        if (isRightController && controlsPanel && vrLegendPanel) {
            const newState = !controlsPanel.visible;
            controlsPanel.visible = newState;
            vrLegendPanel.visible = newState;
        }
    });

    xrRig.add(controller);

    // Controller grip with model (visual controller)
    const controllerGrip = renderer.xr.getControllerGrip(index);
    controllerGrip.add(controllerModelFactory.createControllerModel(controllerGrip));
    
    // Add button hint labels to controller (positioned above controller)
    const triggerLabel = createButtonLabel('Trigger: Teleport');
    triggerLabel.position.set(0, 0.08, 0);
    triggerLabel.rotation.x = -Math.PI / 4; // Tilt toward user
    triggerLabel.visible = false; // Initially hidden
    controllerGrip.add(triggerLabel);
    controller.userData.triggerLabel = triggerLabel;
    
    const gripLabel = createButtonLabel('Grip: Toggle Help');
    gripLabel.position.set(0, 0.04, 0);
    gripLabel.rotation.x = -Math.PI / 4;
    gripLabel.visible = false;
    controllerGrip.add(gripLabel);
    controller.userData.gripLabel = gripLabel;
    
    xrRig.add(controllerGrip);

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

// Optional hand tracking (Quest supports it when enabled)
try {
    const hand1 = renderer.xr.getHand(0);
    const hand2 = renderer.xr.getHand(1);
    hand1.add(handModelFactory.createHandModel(hand1, 'mesh'));
    hand2.add(handModelFactory.createHandModel(hand2, 'mesh'));
    xrRig.add(hand1);
    xrRig.add(hand2);
} catch (e) {
    console.warn("Hand tracking unavailable:", e);
}

function setDesktopControlsEnabled(enabled) {
    orbitControls.enabled = enabled;
    // Fly controls are mutually exclusive with orbit (keep previous mode if needed)
    if (!enabled) {
        flyControls.enabled = false;
        isFlyMode = false;
    }
}

let controlsHelpTimeout = null;

renderer.xr.addEventListener('sessionstart', () => {
    setDesktopControlsEnabled(false);
    
    // Rotate scene content from Z-up (LiDAR) to Y-up (VR standard)
    // Rotate -90° around X axis: Z-up becomes Y-up
    sceneContainer.rotation.x = -Math.PI / 2;
    
    // Position user so LiDAR sensor origin (0,0,0) is at eye level:
    // - local-floor reference space: user's floor is at y=0, eyes at y≈1.6
    // - To make origin at eye level, place floor 1.6m below origin
    // - So xrRig.y = -1.6, then user eyes are at y = -1.6 + 1.6 = 0 (origin)
    xrRig.position.set(0, -1.6, 3); // Start 3m in front of origin, eyes at origin height
    
    // Show VR helpers
    gridHelper.visible = true;
    axesHelper.visible = true;
    originMarker.visible = true;
    originLabel.visible = true;
    controlsPanel.visible = true;
    vrLegendPanel.visible = true;
    
    // Show controller button labels for 8 seconds, then hide
    if (xrController1?.userData?.triggerLabel) xrController1.userData.triggerLabel.visible = true;
    if (xrController1?.userData?.gripLabel) xrController1.userData.gripLabel.visible = true;
    if (xrController2?.userData?.triggerLabel) xrController2.userData.triggerLabel.visible = true;
    if (xrController2?.userData?.gripLabel) xrController2.userData.gripLabel.visible = true;
    
    controlsHelpTimeout = setTimeout(() => {
        if (xrController1?.userData?.triggerLabel) xrController1.userData.triggerLabel.visible = false;
        if (xrController1?.userData?.gripLabel) xrController1.userData.gripLabel.visible = false;
        if (xrController2?.userData?.triggerLabel) xrController2.userData.triggerLabel.visible = false;
        if (xrController2?.userData?.gripLabel) xrController2.userData.gripLabel.visible = false;
    }, 8000);
    
    console.log("XR session started - LiDAR origin at eye level");
});

renderer.xr.addEventListener('sessionend', () => {
    setDesktopControlsEnabled(true);
    
    // Clear any pending timeouts
    if (controlsHelpTimeout) {
        clearTimeout(controlsHelpTimeout);
        controlsHelpTimeout = null;
    }
    
    // Restore original Z-up orientation for desktop
    sceneContainer.rotation.x = 0;
    
    // Hide VR helpers
    gridHelper.visible = false;
    axesHelper.visible = false;
    originMarker.visible = false;
    originLabel.visible = false;
    controlsPanel.visible = false;
    vrLegendPanel.visible = false;
    
    // Hide controller labels
    if (xrController1?.userData?.triggerLabel) xrController1.userData.triggerLabel.visible = false;
    if (xrController1?.userData?.gripLabel) xrController1.userData.gripLabel.visible = false;
    if (xrController2?.userData?.triggerLabel) xrController2.userData.triggerLabel.visible = false;
    if (xrController2?.userData?.gripLabel) xrController2.userData.gripLabel.visible = false;
    
    console.log("XR session ended - scene restored to Z-up");
});

// Simple XR thumbstick locomotion + camera rotation (works in Y-up VR space)
function updateXrLocomotion(delta) {
    if (!renderer.xr.isPresenting) return;

    // Left controller for locomotion (walking/strafing)
    const leftController =
        (xrController1?.userData?.handedness === 'left') ? xrController1 :
        (xrController2?.userData?.handedness === 'left') ? xrController2 :
        xrController1;

    const leftGp = leftController?.userData?.gamepad;
    if (leftGp && leftGp.axes && leftGp.axes.length >= 2) {
        // Quest controllers: axes[2,3] are thumbstick (left) or [0,1] fallback
        const x = leftGp.axes[2] ?? leftGp.axes[0] ?? 0;
        const y = leftGp.axes[3] ?? leftGp.axes[1] ?? 0;
        const deadzone = 0.15;
        const ax = Math.abs(x) > deadzone ? x : 0;
        const ay = Math.abs(y) > deadzone ? y : 0;
        
        // Check if grip is pressed for height adjustment mode
        const gripPressed = leftGp.buttons && leftGp.buttons.length > 1 && leftGp.buttons[1].pressed;
        
        if (gripPressed && ay !== 0) {
            // Height adjustment mode: thumbstick up/down adjusts height
            const heightSpeed = 1.0; // m/s
            xrRig.position.y += ay * heightSpeed * delta;
            console.log(`Height: ${xrRig.position.y.toFixed(2)}m`);
        } else if (ax !== 0 || ay !== 0) {
            // Normal locomotion mode
            const speed = 2.0; // 2 m/s locomotion speed

            // Get camera forward direction projected onto horizontal plane (XZ in Y-up VR)
            const dir = new THREE.Vector3();
            camera.getWorldDirection(dir);
            dir.y = 0; // Keep movement horizontal in Y-up VR space
            if (dir.lengthSq() > 1e-6) {
                dir.normalize();
                
                // Right vector perpendicular to forward in XZ plane
                const right = new THREE.Vector3(-dir.z, 0, dir.x);

                // Thumbstick: y-axis negative = forward, x-axis = strafe
                xrRig.position.addScaledVector(dir, (-ay) * speed * delta);
                xrRig.position.addScaledVector(right, ax * speed * delta);
            }
        }
    }

    // Right controller for camera rotation (smooth turning / look around)
    const rightController =
        (xrController1?.userData?.handedness === 'right') ? xrController1 :
        (xrController2?.userData?.handedness === 'right') ? xrController2 :
        xrController2;

    const rightGp = rightController?.userData?.gamepad;
    if (rightGp && rightGp.axes && rightGp.axes.length >= 2) {
        // Quest controllers: axes[2,3] are thumbstick
        const x = rightGp.axes[2] ?? rightGp.axes[0] ?? 0;
        const y = rightGp.axes[3] ?? rightGp.axes[1] ?? 0;
        const deadzone = 0.15;
        const ax = Math.abs(x) > deadzone ? x : 0;
        const ay = Math.abs(y) > deadzone ? y : 0;
        
        if (ax !== 0 || ay !== 0) {
            // Rotation speed (radians per second)
            const rotSpeed = 2.0;
            
            // Yaw rotation (left/right) - rotate rig around Y axis
            if (ax !== 0) {
                const yawDelta = ax * rotSpeed * delta;
                xrRig.rotateY(-yawDelta); // Negative for intuitive right = turn right
            }
            
            // Pitch rotation (up/down) - tilt view without changing rig
            // Note: Natural head tracking pitch is preserved; this adds manual adjustment
            if (ay !== 0) {
                const pitchDelta = ay * rotSpeed * delta;
                
                // Get current pitch from camera's world direction
                const worldDir = new THREE.Vector3();
                camera.getWorldDirection(worldDir);
                const currentPitch = Math.asin(-worldDir.y);
                const newPitch = currentPitch + pitchDelta;
                
                // Clamp pitch to prevent disorientation (±80 degrees)
                const maxPitch = Math.PI * 0.44; // ~80 degrees
                if (Math.abs(newPitch) < maxPitch) {
                    // Rotate camera around its local X axis (pitch)
                    const axis = new THREE.Vector3(1, 0, 0);
                    camera.rotateOnAxis(axis, pitchDelta);
                }
            }
        }
    }
}

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
    } else {
        updateXrLocomotion(delta);
        
        // Update teleport marker position based on controller pointing
        let foundTarget = false;
        for (const controller of [xrController1, xrController2]) {
            if (controller && controller.userData.isSelecting) {
                const hit = intersectGround(controller);
                if (hit) {
                    teleportMarker.position.copy(hit.point);
                    teleportMarker.visible = true;
                    foundTarget = true;
                    break;
                }
            }
        }
        if (!foundTarget) {
            teleportMarker.visible = false;
        }
        
        // Make origin label always face the camera (billboard)
        if (originLabel && originLabel.visible) {
            originLabel.lookAt(camera.position);
        }
        
        // Pulse the origin marker for visibility
        if (originMarker && originMarker.visible) {
            const scale = 1.0 + 0.3 * Math.sin(Date.now() * 0.003);
            originMarker.scale.setScalar(scale);
        }
    }

    renderer.render(scene, camera);
}

// Handle window resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

renderer.setAnimationLoop(animate);
