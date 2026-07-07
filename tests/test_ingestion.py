import os
import tempfile
import pandas as pd
from ingestion.live_subscriber import UnitreeLiveSubscriber
from ingestion.exporter import DataExporter
from ingestion.data_models import LowStateData

# Mimics the IDL message object published by the Unitree SDK over DDS/CycloneDX,
# so tests can run without a live robot or network connection.
class MockLowStateMsg:
    def __init__(self):
        self.tick = 12345  # monotonic hardware timestamp counter

        # Represents one joint's state: position (q), velocity (dq), estimated torque (tauEst).
        class MotorState:
            def __init__(self, q, dq, tau):
                self.q = q
                self.dq = dq
                self.tauEst = tau

        # Two motors with distinct values so we can verify the handler reads all of them.
        self.motorState = [MotorState(0.1, 0.01, 1.0), MotorState(0.2, 0.02, 2.0)]

        # Represents the onboard IMU: roll/pitch/yaw and linear acceleration.
        class ImuState:
            def __init__(self):
                self.rpy = [0.1, 0.2, 0.3]
                self.accelerometer = [0.0, 0.0, 9.8]  # near-gravity Z at rest

        self.imuState = ImuState()


def test_live_subscriber_handler():
    """Test the subscriber's ability to extract data from IDL objects."""
    subscriber = UnitreeLiveSubscriber(interface="mock")
    # Enable recording so the handler actually appends to low_states instead of dropping the message.
    subscriber._is_recording = True

    msg = MockLowStateMsg()
    # Directly invoke the DDS callback that would normally fire on each robot broadcast.
    subscriber._lowstate_handler(msg)

    # Exactly one frame should have been captured.
    assert len(subscriber.low_states) == 1
    data = subscriber.low_states[0]

    # Verify the handler correctly flattened nested IDL fields into a plain dict.
    assert data["tick"] == 12345
    assert data["q"] == [0.1, 0.2]
    assert data["dq"] == [0.01, 0.02]
    assert data["tau"] == [1.0, 2.0]
    assert data["rpy"] == [0.1, 0.2, 0.3]


def test_exporter_to_dataframe():
    """Test converting parsed dictionaries to Pandas DataFrame."""
    # Two frames with different ticks to confirm row ordering is preserved.
    data = [
        LowStateData(tick=1, q=[0.1], dq=[0.01], tau=[1.0], rpy=[0.0], accel=[9.8]),
        LowStateData(tick=2, q=[0.2], dq=[0.02], tau=[2.0], rpy=[0.1], accel=[9.8])
    ]

    df = DataExporter.to_dataframe(data)

    assert not df.empty
    assert len(df) == 2
    # "tick" is the primary time key — must survive the conversion.
    assert "tick" in df.columns
    assert df.iloc[0]["tick"] == 1


def test_exporter_save_parquet():
    """Test saving a dataframe to parquet."""
    data = [
        LowStateData(tick=1, q=[0.1], dq=[0.01], tau=[1.0], rpy=[0.0], accel=[9.8])
    ]

    # Use a temp directory so the test leaves no files on disk after it runs.
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.parquet")
        DataExporter.export_parquet(data, filepath)

        # Confirm the file was actually written before trying to read it back.
        assert os.path.exists(filepath)

        # Round-trip check: data read from disk should match what was written.
        df = pd.read_parquet(filepath)
        assert len(df) == 1
        assert df.iloc[0]["tick"] == 1
