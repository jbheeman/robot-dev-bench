Hi! Because you asked me to explain the plan before making any changes, here is my analysis and proposed solution:

### Why are the legs behind the center of gravity?
This happens because we are mapping a **human** pose to a **robot** frame. 
In a natural human standing pose, the spine is curved and the upper body naturally leans slightly forward from the hips to maintain balance. The H36M neural network accurately captures this human forward lean. 

However, the Unitree G1 robot has a completely **rigid, straight torso**. When we map the human's forward-leaning spine to the robot's rigid torso, the entire robot torso gets tilted forward. Because the torso is tilted forward, the pelvis (and therefore the legs attached to it) appear to be pushed *backward* relative to the robot's overall center of mass.

### How I plan to fix it:
To fix this, we need to correct the posture by adjusting the robot's global pitch (forward/backward tilt) or by specifically shifting the legs forward. 

There are two ways we can do this:
1. **Global Rotation (Recommended):** I can add a small corrective pitch rotation (tilting the whole skeleton backward by a few degrees) whenever the robot is standing. This will bring the torso upright and naturally swing the legs forward so they are directly underneath the center of gravity.
2. **Hip Translation:** I can manually add a forward offset (in the Z-axis) to the hip joints in `apply_g1_morphology`. Since the knees and ankles are attached to the hips, this will shift the entire leg structure forward.

Do you have a preference between these two approaches, or would you like me to proceed with the Global Rotation method?
