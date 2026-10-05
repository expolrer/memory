from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SafeState:
    """Symbolic safe state used before the SAPIEN environment is wired in."""

    knob_open: bool = False
    handle_open: bool = False
    door_open: bool = False
    door_locked: bool = True
    input_buffer: List[str] = field(default_factory=list)
    metadata: Dict[str, int] = field(default_factory=dict)

    def snapshot(self) -> Dict[str, object]:
        return {
            "knob_open": self.knob_open,
            "handle_open": self.handle_open,
            "door_open": self.door_open,
            "door_locked": self.door_locked,
            "input_buffer": list(self.input_buffer),
            "metadata": dict(self.metadata),
        }


class Rule:
    rule_id = "base"
    description = ""

    def reset(self, state: SafeState) -> None:
        state.door_locked = True
        state.input_buffer.clear()
        state.metadata.clear()

    def on_knob_change(self, state: SafeState, opened: bool) -> None:
        state.knob_open = opened
        self.update_lock(state)

    def on_handle_change(self, state: SafeState, opened: bool) -> None:
        state.handle_open = opened
        self.update_lock(state)

    def on_door_attempt(self, state: SafeState) -> bool:
        self.update_lock(state)
        if not state.door_locked:
            state.door_open = True
            return True
        return False

    def update_lock(self, state: SafeState) -> None:
        raise NotImplementedError

    def process_score(self, state: SafeState) -> float:
        raise NotImplementedError
