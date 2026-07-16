let scene, camera, renderer, controls;
let poses3D = null;
let validMask = null;
let currentFrame = 0;
let isPlaying = false;
let animationFrameId = null;

// Joints and lines
let jointMeshes = [];
let boneLines = [];

const COCO_KEYPOINTS = 17;
const COCO_SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
    [5, 11], [6, 12], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
    [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]
];

// Initialize Three.js scene
function initViewer() {
    const container = document.getElementById('viewer-container');
    if (!container) return;
    
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0e);
    
    // Grid
    const grid = new THREE.GridHelper(5, 50, 0x444444, 0x222222);
    scene.add(grid);
    
    // Axes helper (X red, Y green, Z blue)
    const axesHelper = new THREE.AxesHelper(1);
    scene.add(axesHelper);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(5, 10, 5);
    scene.add(dirLight);

    camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 100);
    camera.position.set(2, 2, 3);
    camera.lookAt(0, 0, 0);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);
    
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0);
    controls.update();

    // Create skeleton meshes
    const sphereGeo = new THREE.SphereGeometry(0.03, 16, 16);
    const materialValid = new THREE.MeshStandardMaterial({ color: 0x3b82f6 }); // Blue

    for (let i = 0; i < COCO_KEYPOINTS; i++) {
        const mesh = new THREE.Mesh(sphereGeo, materialValid);
        mesh.visible = false;
        scene.add(mesh);
        jointMeshes.push(mesh);
    }

    const lineMat = new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 2 });
    for (let i = 0; i < COCO_SKELETON.length; i++) {
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
        const line = new THREE.Line(geo, lineMat);
        line.visible = false;
        scene.add(line);
        boneLines.push(line);
    }

    const resizeObserver = new ResizeObserver(() => {
        if (!container || container.clientWidth === 0) return;
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
        render();
    });
    resizeObserver.observe(container);
    
    // Controls
    document.getElementById('play-btn')?.addEventListener('click', play);
    document.getElementById('pause-btn')?.addEventListener('click', pause);
    document.getElementById('timeline')?.addEventListener('input', (e) => {
        pause();
        currentFrame = parseInt(e.target.value);
        updateSkeleton();
        render();
    });

    // Reset View Button
    document.getElementById('reset-view-btn')?.addEventListener('click', () => {
        camera.position.set(2, 2, 3);
        controls.target.set(0, 0, 0);
        camera.lookAt(0, 0, 0);
        controls.update();
        render();
    });
}

function loadPlaybackData(dataPoses, dataValidMask) {
    pause();
    poses3D = dataPoses;
    validMask = dataValidMask;
    currentFrame = 0;
    
    const timeline = document.getElementById('timeline');
    if (timeline) {
        timeline.max = Math.max(0, poses3D.length - 1);
        timeline.value = 0;
    }
    
    updateSkeleton();
    
    // Center camera on the median point of the first valid frame
    if (poses3D && poses3D.length > 0) {
        let firstFrame = poses3D[0];
        let sum = new THREE.Vector3();
        let count = 0;
        for (let i=0; i<COCO_KEYPOINTS; i++) {
            if (firstFrame[i] && firstFrame[i][0] !== null) {
                sum.x += firstFrame[i][0];
                sum.y += firstFrame[i][1]; // Assuming Y is up
                sum.z += firstFrame[i][2];
                count++;
            }
        }
        if (count > 0) {
            sum.divideScalar(count);
            controls.target.copy(sum);
            camera.position.set(sum.x + 2, sum.y + 1, sum.z + 3);
            controls.update();
        }
    }
    
    render();
}

function updateSkeleton() {
    if (!poses3D || poses3D.length === 0) return;
    
    const frame = Math.min(currentFrame, poses3D.length - 1);
    const joints = poses3D[frame];
    
    // Update joint positions
    for (let i = 0; i < COCO_KEYPOINTS; i++) {
        if (joints[i] && joints[i][0] !== null) {
            jointMeshes[i].position.set(joints[i][0], joints[i][1], joints[i][2]);
            jointMeshes[i].visible = true;
        } else {
            jointMeshes[i].visible = false;
        }
    }
    
    // Update bone lines
    for (let i = 0; i < COCO_SKELETON.length; i++) {
        const [u, v] = COCO_SKELETON[i];
        if (jointMeshes[u].visible && jointMeshes[v].visible) {
            const pos = boneLines[i].geometry.attributes.position.array;
            pos[0] = jointMeshes[u].position.x;
            pos[1] = jointMeshes[u].position.y;
            pos[2] = jointMeshes[u].position.z;
            pos[3] = jointMeshes[v].position.x;
            pos[4] = jointMeshes[v].position.y;
            pos[5] = jointMeshes[v].position.z;
            boneLines[i].geometry.attributes.position.needsUpdate = true;
            boneLines[i].visible = true;
        } else {
            boneLines[i].visible = false;
        }
    }
}

function play() {
    if (isPlaying || !poses3D || poses3D.length === 0) return;
    isPlaying = true;
    
    let totalFrames = poses3D.length;
    
    if (currentFrame >= totalFrames - 1) {
        currentFrame = 0;
    }
    
    let lastTime = performance.now();
    
    function loop(time) {
        if (!isPlaying) return;
        
        // 30fps playback
        if (time - lastTime > 33.3) {
            currentFrame++;
            if (currentFrame >= totalFrames) {
                currentFrame = 0;
                if(document.getElementById('timeline')) document.getElementById('timeline').value = currentFrame;
                updateSkeleton();
                render();
                pause();
                return;
            }
            if(document.getElementById('timeline')) document.getElementById('timeline').value = currentFrame;
            updateSkeleton();
            render();
            lastTime = time;
        }
        animationFrameId = requestAnimationFrame(loop);
    }
    animationFrameId = requestAnimationFrame(loop);
}

function pause() {
    isPlaying = false;
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
}

function render() {
    if (renderer && scene && camera) {
        if (controls) controls.update();
        renderer.render(scene, camera);
    }
}

function renderLoop() {
    requestAnimationFrame(renderLoop);
    if (!isPlaying) {
        render();
    }
}

window.loadPlaybackData = loadPlaybackData;

document.addEventListener('DOMContentLoaded', () => {
    initViewer();
    renderLoop();
});
