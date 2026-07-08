let scene, camera, renderer, robot, controls;
let playbackData = null;
let currentFrame = 0;
let isPlaying = false;
let animationFrameId = null;

// Initialize Three.js scene
function initViewer() {
    const container = document.getElementById('viewer-container');
    if (!container) return;
    
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0e);
    
    // Add grid and lights
    const grid = new THREE.GridHelper(5, 50, 0x444444, 0x222222);
    scene.add(grid);
    
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(5, 10, 5);
    scene.add(dirLight);

    camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 100);
    camera.position.set(2.5, 1.5, 2.5);
    camera.lookAt(0, 0.5, 0);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);
    
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0.5, 0);
    controls.update();
    
    // Load URDF
    const manager = new THREE.LoadingManager();
    const loader = new URDFLoader(manager);
    
    // Important: Tell URDF loader how to resolve package:// URLs
    loader.packages = {
        'g1_description': 'assets/g1_description/'
    };
    
    loader.load(
        'assets/g1_description/g1_29dof.urdf', 
        result => {
            robot = result;
            // ROS uses Z-up, Three.js uses Y-up. Rotate the robot to match.
            robot.rotation.x = -Math.PI / 2;
            scene.add(robot);
            render();
        },
        undefined,
        error => {
            console.error("URDFLoader Error:", error);
        }
    );
    
    const resizeObserver = new ResizeObserver(() => {
        if (!container || container.clientWidth === 0) return;
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
        render();
    });
    resizeObserver.observe(container);
    
    // Controls
    document.getElementById('play-btn').addEventListener('click', play);
    document.getElementById('pause-btn').addEventListener('click', pause);
    document.getElementById('timeline').addEventListener('input', (e) => {
        pause();
        currentFrame = parseInt(e.target.value);
        updateRobotPose();
        render();
    });
}

function loadPlaybackData(data) {
    pause(); // Stop any ongoing playback
    playbackData = data;
    currentFrame = 0;
    const timeline = document.getElementById('timeline');
    timeline.max = Math.max(0, data.q.length - 1);
    timeline.value = 0;
    updateRobotPose();
    render();
}

const G1_JOINT_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
];

function updateRobotPose() {
    if (!robot || !playbackData || playbackData.q.length === 0) return;
    const qRow = playbackData.q[currentFrame];
    
    // Explicitly map the 29 canonical array values to their exact joint names
    for (let i = 0; i < G1_JOINT_ORDER.length && i < qRow.length; i++) {
        const jointName = G1_JOINT_ORDER[i];
        const joint = robot.joints[jointName];
        if (joint) {
            joint.setJointValue(qRow[i]);
        }
    }
}

function play() {
    if (isPlaying || !playbackData || !playbackData.q || playbackData.q.length === 0) return;
    isPlaying = true;
    
    // Auto-restart if we are at the end
    if (currentFrame >= playbackData.q.length - 1) {
        currentFrame = 0;
    }
    
    let lastTime = performance.now();
    
    function loop(time) {
        if (!isPlaying) return;
        
        // 30fps playback (33.3ms per frame)
        if (time - lastTime > 33.3) {
            currentFrame++;
            if (currentFrame >= playbackData.q.length) {
                currentFrame = 0;
                document.getElementById('timeline').value = currentFrame;
                updateRobotPose();
                render();
                pause();
                return;
            }
            document.getElementById('timeline').value = currentFrame;
            updateRobotPose();
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

// Continuous render loop for orbit controls when paused
function renderLoop() {
    requestAnimationFrame(renderLoop);
    if (!isPlaying) {
        render();
    }
}

// Expose load function for script.js
window.loadPlaybackData = loadPlaybackData;

document.addEventListener('DOMContentLoaded', () => {
    initViewer();
    renderLoop();
});
