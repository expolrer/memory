"""RuleSafe rules transcribed from VQ-Memory Appendix Table 5.

The historical reconstruction is intentionally kept intact. This module only
overrides the four rules whose event-to-token mapping differs from the paper.
"""

from .base import SafeState
from .generated_rules import (
    Rule001,
    Rule002,
    Rule003,
    Rule004,
    Rule005,
    Rule006,
    Rule007,
    Rule008,
    Rule009,
    Rule010,
    Rule011,
    Rule012,
    Rule013,
    Rule014,
    Rule015,
    Rule016,
    Rule017,
    Rule018,
    Rule019,
    Rule020,
    _append_buffer,
)


class PaperRule006(Rule006):
    description = (
        "The safe door remains locked unless password 010 is entered. "
        "A knob state change inputs 0 while the handle is open, otherwise 1."
    )

    def on_knob_change(self, state: SafeState, opened: bool) -> None:
        state.knob_open = opened
        _append_buffer(state, "0" if state.handle_open else "1", len(self.password))
        self.update_lock(state)


class PaperRule009(Rule009):
    description = (
        "The safe door unlocks after password 1001. "
        "A knob state change inputs 1 and a handle state change inputs 0; only the last four inputs count."
    )

    def on_knob_change(self, state: SafeState, opened: bool) -> None:
        state.knob_open = opened
        _append_buffer(state, "1", len(self.password))
        self.update_lock(state)

    def on_handle_change(self, state: SafeState, opened: bool) -> None:
        state.handle_open = opened
        _append_buffer(state, "0", len(self.password))
        self.update_lock(state)


class PaperRule010(Rule010):
    description = (
        "The safe door unlocks when password 0110 is entered. "
        "A handle state change inputs 0 while the knob is open, otherwise 1."
    )

    def on_handle_change(self, state: SafeState, opened: bool) -> None:
        _append_buffer(state, "0" if state.knob_open else "1", len(self.password))
        state.handle_open = opened
        self.update_lock(state)


class PaperRule011(Rule011):
    description = (
        "The safe door unlocks after password 1010. Closing the knob inputs 1, "
        "opening the knob inputs 0, and handle changes are ignored."
    )

    def on_knob_change(self, state: SafeState, opened: bool) -> None:
        if state.knob_open != opened:
            _append_buffer(state, "0" if opened else "1", len(self.password))
        state.knob_open = opened
        self.update_lock(state)


PAPER_RULE_CLASSES = [
    Rule001,
    Rule002,
    Rule003,
    Rule004,
    Rule005,
    PaperRule006,
    Rule007,
    Rule008,
    PaperRule009,
    PaperRule010,
    PaperRule011,
    Rule012,
    Rule013,
    Rule014,
    Rule015,
    Rule016,
    Rule017,
    Rule018,
    Rule019,
    Rule020,
]

PAPER_RULES = {cls.rule_id: cls for cls in PAPER_RULE_CLASSES}
PAPER_RULE_DESCRIPTIONS = {rule_id: cls.description for rule_id, cls in PAPER_RULES.items()}


__all__ = [
    "PaperRule006",
    "PaperRule009",
    "PaperRule010",
    "PaperRule011",
    "PAPER_RULE_CLASSES",
    "PAPER_RULES",
    "PAPER_RULE_DESCRIPTIONS",
]
