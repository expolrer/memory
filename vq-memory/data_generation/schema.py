from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class TrajectoryFrame:
    timestamp: float
    joint_state: List[float]
    action: List[float]
    rgb_first_path: Optional[str] = None
    rgb_third_path: Optional[str] = None
    point_cloud_path: Optional[str] = None


@dataclass
class TrajectoryRecord:
    rule_id: str
    language_instruction: str
    frames: List[TrajectoryFrame]
    success: bool
    process_score: float
    metadata: Dict[str, object]
