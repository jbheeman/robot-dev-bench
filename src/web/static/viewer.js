let scene, camera, renderer, robot, robotB, controls;
let playbackData = null;
let playbackDataB = null;
let currentFrame = 0;
let isPlaying = false;
let animationFrameId = null;
let comSphere = null;

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

    // Create CoM Sphere
    const sphereGeometry = new THREE.SphereGeometry(0.04, 16, 16);
    const sphereMaterial = new THREE.MeshBasicMaterial({ color: 0xef4444, wireframe: true, transparent: true, opacity: 0.7 });
    comSphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
    scene.add(comSphere);

    camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 100);
    camera.position.set(2.5, 1.5, 2.5);
    camera.lookAt(0, 0.1, 0); // Aim slightly lower towards the torso

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);
    
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0.1, 0); // Focus controls on the torso
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
            
            // Load baseline robot (Robot B)
            loader.load('assets/g1_description/g1_29dof.urdf', resultB => {
                robotB = resultB;
                robotB.rotation.x = -Math.PI / 2;
                scene.add(robotB);
                robotB.visible = false; // Hide initially
                render();
            });
            
            render();
        },
        undefined,
        error => {
            console.error("URDFLoader Error:", error);
        }
    );
    
    // URDFLoader loads STL meshes asynchronously. We must wait for the entire manager to finish 
    // before applying the material recursively to ensure the meshes actually exist.
    manager.onLoad = () => {
        if (robotB) {
            const ghostMat = new THREE.MeshStandardMaterial({
                color: 0x3b82f6, // Blue
                transparent: true,
                opacity: 0.5,
                depthWrite: false
            });
            
            robotB.traverse(child => {
                if (child.isMesh) {
                    child.material = ghostMat;
                }
            });
            render();
        }
    };
    
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

function loadPlaybackData(data, dataB = null) {
    pause(); // Stop any ongoing playback
    playbackData = data;
    playbackDataB = dataB;
    currentFrame = 0;
    
    if (robotB) {
        robotB.visible = !!dataB;
    }
    
    const timeline = document.getElementById('timeline');
    let totalFrames = Math.max(0, data.q.length - 1);
    if (dataB && dataB.q) {
        totalFrames = Math.max(totalFrames, dataB.q.length - 1);
    }
    timeline.max = totalFrames;
    timeline.value = 0;
    
    // Render Anomaly Markers
    const markersContainer = document.getElementById('anomaly-markers');
    if (markersContainer) {
        markersContainer.innerHTML = ''; // Clear previous
        if (data.anomalies && totalFrames > 0) {
            for (const [name, frameIndex] of Object.entries(data.anomalies)) {
                const percent = (frameIndex / totalFrames) * 100;
                const marker = document.createElement('div');
                marker.style.position = 'absolute';
                marker.style.left = `${percent}%`;
                marker.style.top = '25%';
                marker.style.bottom = '25%';
                marker.style.width = '2px';
                marker.style.backgroundColor = '#ef4444'; // Red
                marker.style.boxShadow = '0 0 4px #ef4444';
                marker.title = `${name} (Frame ${frameIndex})`;
                
                // Optional label
                const label = document.createElement('span');
                label.textContent = name;
                label.style.position = 'absolute';
                label.style.top = '-20px';
                label.style.left = '50%';
                label.style.transform = 'translateX(-50%)';
                label.style.fontSize = '10px';
                label.style.color = '#ef4444';
                label.style.whiteSpace = 'nowrap';
                marker.appendChild(label);
                
                markersContainer.appendChild(marker);
            }
        }
    }
    
    if (robotB) {
        robotB.position.set(0, 0, 0);
    }
    
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
    if (!playbackData) return;
    
    if (robot && playbackData.q.length > 0) {
        const frameA = Math.min(currentFrame, playbackData.q.length - 1);
        const qRow = playbackData.q[frameA];
        
        for (let i = 0; i < G1_JOINT_ORDER.length && i < qRow.length; i++) {
            const jointName = G1_JOINT_ORDER[i];
            const joint = robot.joints[jointName];
            if (joint) {
                joint.setJointValue(qRow[i]);
            }
        }
    }
    
    if (robotB && playbackDataB && playbackDataB.q && playbackDataB.q.length > 0) {
        const frameB = Math.min(currentFrame, playbackDataB.q.length - 1);
        const qRowB = playbackDataB.q[frameB];
        
        for (let i = 0; i < G1_JOINT_ORDER.length && i < qRowB.length; i++) {
            const jointName = G1_JOINT_ORDER[i];
            const joint = robotB.joints[jointName];
            if (joint) {
                joint.setJointValue(qRowB[i]);
            }
        }
    }
    
    // Update CoM Sphere position
    if (comSphere && robot && robot.links) {
        let targetLink = robot.links['pelvis'] || robot.links['torso'] || robot.links['base_link'];
        if (targetLink) {
            targetLink.getWorldPosition(comSphere.position);
        } else {
            // Fallback: average position of all links
            let total = 0;
            let com = new THREE.Vector3();
            let v = new THREE.Vector3();
            for (let name in robot.links) {
                robot.links[name].getWorldPosition(v);
                com.add(v);
                total++;
            }
            if (total > 0) {
                com.divideScalar(total);
                comSphere.position.copy(com);
            }
        }
    }
    

    
    // Sync telemetry charts if available
    if (window.updateChartCrosshair) {
        window.updateChartCrosshair(currentFrame);
    }
}

function play() {
    if (isPlaying || !playbackData || !playbackData.q || playbackData.q.length === 0) return;
    isPlaying = true;
    
    let totalFrames = playbackData.q.length;
    if (playbackDataB && playbackDataB.q) {
        totalFrames = Math.max(totalFrames, playbackDataB.q.length);
    }
    
    // Auto-restart if we are at the end
    if (currentFrame >= totalFrames - 1) {
        currentFrame = 0;
    }
    
    let lastTime = performance.now();
    
    function loop(time) {
        if (!isPlaying) return;
        
        // 30fps playback (33.3ms per frame)
        if (time - lastTime > 33.3) {
            currentFrame++;
            if (currentFrame >= totalFrames) {
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
