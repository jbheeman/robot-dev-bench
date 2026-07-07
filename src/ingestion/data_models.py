from typing import TypedDict, List

class LowStateData(TypedDict):
    tick: int
    q: List[float]
    dq: List[float]
    tau: List[float]
    rpy: List[float]
    accel: List[float]

class HighStateData(TypedDict):
    tick: int
    base_velocity: List[float]
    odometry: List[float]
    foot_contact: List[int]
