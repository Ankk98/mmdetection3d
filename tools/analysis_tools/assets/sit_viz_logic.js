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
    const infoDiv = document.getElementById('info');
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

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    
    const delta = clock.getDelta();
    
    if (isFlyMode) {
        flyControls.update(delta);
    } else {
        orbitControls.update();
    }
    
    renderer.render(scene, camera);
}

// Handle window resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

animate();
