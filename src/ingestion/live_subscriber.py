import logging
import time
from typing import List

# These imports assume unitree_sdk2_python is installed
try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__HighState_
except ImportError:
    logging.warning("unitree_sdk2py not found. Please install from Unitree SDK2 GitHub.")

from .data_models import LowStateData, HighStateData

logger = logging.getLogger(__name__)

class UnitreeLiveSubscriber:
    """
    Subscribes to live CycloneDDS topics using unitree_sdk2_python.
    """
    def __init__(self, interface: str = "eth0"):
        self.interface = interface
        self.low_states: List[LowStateData] = []
        self.high_states: List[HighStateData] = []
        self._is_recording = False
        self._lowstate_sub = None
        self._highstate_sub = None

    def initialize(self):
        """Initializes the CycloneDDS channel factory."""
        logger.info(f"Initializing ChannelFactory on interface {self.interface}...")
        try:
            ChannelFactoryInitialize(0, self.interface)
        except Exception as e:
            logger.error(f"Failed to initialize ChannelFactory: {e}")
            raise

    def start_recording(self):
        """Starts the DDS subscribers."""
        logger.info("Starting live telemetry recording...")
        self._is_recording = True
        
        try:
            self._lowstate_sub = ChannelSubscriber("rt/lowstate", unitree_go_msg_dds__LowState_)
            self._lowstate_sub.Init(self._lowstate_handler, 10)
            
            # Using placeholder types for highstate based on standard unitree schemas
            # Note: The exact IDL name for HighState might vary slightly (e.g. SportModeState)
            self._highstate_sub = ChannelSubscriber("rt/highstate", unitree_go_msg_dds__HighState_)
            self._highstate_sub.Init(self._highstate_handler, 10)
        except Exception as e:
            logger.error(f"Failed to initialize subscribers: {e}")
            self._is_recording = False

    def stop_recording(self):
        """Stops the recording."""
        logger.info("Stopping telemetry recording.")
        self._is_recording = False

    def _lowstate_handler(self, msg: 'unitree_go_msg_dds__LowState_'):
        """Callback for rt/lowstate messages."""
        if not self._is_recording:
            return
            
        try:
            # Extract lists from the nested SDK structures
            # Assuming msg.motorState is an array of motor objects
            q = [m.q for m in msg.motorState] if hasattr(msg, 'motorState') else []
            dq = [m.dq for m in msg.motorState] if hasattr(msg, 'motorState') else []
            tau = [m.tauEst for m in msg.motorState] if hasattr(msg, 'motorState') else []
            
            data = LowStateData(
                tick=msg.tick if hasattr(msg, 'tick') else int(time.time() * 1000),
                q=q,
                dq=dq,
                tau=tau,
                rpy=list(msg.imuState.rpy) if hasattr(msg, 'imuState') else [],
                accel=list(msg.imuState.accelerometer) if hasattr(msg, 'imuState') else []
            )
            self.low_states.append(data)
        except Exception as e:
            logger.debug(f"Error parsing lowstate: {e}")

    def _highstate_handler(self, msg: 'unitree_go_msg_dds__HighState_'):
        """Callback for rt/highstate messages."""
        if not self._is_recording:
            return
            
        try:
            data = HighStateData(
                tick=msg.tick if hasattr(msg, 'tick') else int(time.time() * 1000),
                base_velocity=list(msg.velocity) if hasattr(msg, 'velocity') else [],
                odometry=list(msg.position) if hasattr(msg, 'position') else [],
                foot_contact=list(msg.footForce) if hasattr(msg, 'footForce') else []
            )
            self.high_states.append(data)
        except Exception as e:
            logger.debug(f"Error parsing highstate: {e}")
