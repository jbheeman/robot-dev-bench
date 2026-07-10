import time
import logging
import math

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.core.channel import ChannelPublisher
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorState_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import IMUState_
except ImportError:
    logging.error("unitree_sdk2py not found. Please install the official SDK to run the mock publisher.")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Initialize CycloneDDS on the local loopback interface
    ChannelFactoryInitialize(0, "lo")
    
    # Create a publisher for rt/lowstate
    pub = ChannelPublisher("rt/lowstate", LowState_)
    pub.Init()

    logger.info("Starting mock publisher on interface 'lo' (Topic: rt/lowstate)...")
    
    tick = 0
    try:
        while True:
            msg = unitree_go_msg_dds__LowState_()
            msg.tick = tick
            
            # Mock 12 joints (typical for a humanoid/quadruped)
            msg.motorState = []
            for i in range(12):
                motor = MotorState_()
                # Generate a sine wave to simulate joint movement
                motor.q = math.sin(tick * 0.1 + i)
                motor.dq = math.cos(tick * 0.1 + i)
                motor.tauEst = 0.5 * math.sin(tick * 0.05)
                msg.motorState.append(motor)
                
            # Mock IMU
            msg.imuState = IMUState_()
            msg.imuState.rpy = [0.0, 0.0, 0.0]
            msg.imuState.accelerometer = [0.0, 0.0, 9.81]

            # Publish the message
            pub.Write(msg)
            
            tick += 1
            time.sleep(0.01) # Publish at ~100Hz
            
            if tick % 100 == 0:
                logger.info(f"Published 100 messages (tick: {tick})")

    except KeyboardInterrupt:
        logger.info("Mock publisher stopped by user.")

if __name__ == "__main__":
    main()
