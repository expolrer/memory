from .base import Rule, SafeState


def _append_buffer(state: SafeState, token: str, max_len: int) -> None:
    state.input_buffer.append(str(token))
    state.input_buffer[:] = state.input_buffer[-max_len:]


def _set_progress(state: SafeState, value: float) -> None:
    state.metadata["progress"] = int(max(0.0, min(1.0, value)) * 1000)


def _get_progress(state: SafeState) -> float:
    return float(state.metadata.get("progress", 0)) / 1000.0


class ProgressRule(Rule):
    def process_score(self, state: SafeState) -> float:
        if state.door_open:
            return 1.0
        if not state.door_locked:
            return max(_get_progress(state), 0.85)
        return _get_progress(state)


class Rule001(ProgressRule):
    rule_id = "rule001"
    description = "The safe door remains locked unless both the knob and handle are open."

    def update_lock(self, state: SafeState) -> None:
        _set_progress(state, (int(state.knob_open) + int(state.handle_open)) / 2.0)
        state.door_locked = not (state.knob_open and state.handle_open)


class Rule002(ProgressRule):
    rule_id = "rule002"
    description = "The safe door remains locked unless the knob is open."

    def update_lock(self, state: SafeState) -> None:
        _set_progress(state, float(state.knob_open))
        state.door_locked = not state.knob_open


class Rule003(ProgressRule):
    rule_id = "rule003"
    description = "The safe door remains locked unless the handle is open."

    def update_lock(self, state: SafeState) -> None:
        _set_progress(state, float(state.handle_open))
        state.door_locked = not state.handle_open


class Rule004(Rule001):
    rule_id = "rule004"
    description = "The safe door remains locked unless both knob and handle are open; knob cannot be opened unless handle is open."

    def on_knob_change(self, state: SafeState, opened: bool) -> None:
        if opened and not state.handle_open:
            state.metadata["blocked_knob_attempts"] = state.metadata.get("blocked_knob_attempts", 0) + 1
        else:
            state.knob_open = opened
        self.update_lock(state)


class Rule005(ProgressRule):
    rule_id = "rule005"
    description = "The safe door remains locked unless password 001 is entered. Knob change inputs 0; handle change inputs 1."
    password = ["0", "0", "1"]

    def on_knob_change(self, state: SafeState, opened: bool) -> None:
        state.knob_open = opened
        _append_buffer(state, "0", len(self.password))
        self.update_lock(state)

    def on_handle_change(self, state: SafeState, opened: bool) -> None:
        state.handle_open = opened
        _append_buffer(state, "1", len(self.password))
        self.update_lock(state)

    def update_lock(self, state: SafeState) -> None:
        _set_progress(state, len(state.input_buffer) / len(self.password))
        state.door_locked = state.input_buffer != self.password


class Rule006(ProgressRule):
    rule_id = "rule006"
    description = "The safe door remains locked unless password 010 is entered. Knob change inputs 1 when handle is open, otherwise 0."
    password = ["0", "1", "0"]

    def on_knob_change(self, state: SafeState, opened: bool) -> None:
        state.knob_open = opened
        _append_buffer(state, "1" if state.handle_open else "0", len(self.password))
        self.update_lock(state)

    def on_handle_change(self, state: SafeState, opened: bool) -> None:
        state.handle_open = opened
        self.update_lock(state)

    def update_lock(self, state: SafeState) -> None:
        _set_progress(state, len(state.input_buffer) / len(self.password))
        state.door_locked = state.input_buffer != self.password


class Rule007(ProgressRule):
    rule_id = "rule007"
    description = "The safe door unlocks when knob and handle are in opposite states and both have been toggled at least once."

    def on_knob_change(self, state: SafeState, opened: bool) -> None:
        state.knob_open = opened
        state.metadata["knob_toggles"] = state.metadata.get("knob_toggles", 0) + 1
        self.update_lock(state)

    def on_handle_change(self, state: SafeState, opened: bool) -> None:
        state.handle_open = opened
        state.metadata["handle_toggles"] = state.metadata.get("handle_toggles", 0) + 1
        self.update_lock(state)

    def update_lock(self, state: SafeState) -> None:
        toggled = int(state.metadata.get("knob_toggles", 0) > 0) + int(state.metadata.get("handle_toggles", 0) > 0)
        opposite = state.knob_open != state.handle_open
        _set_progress(state, (toggled + int(opposite)) / 3.0)
        state.door_locked = not (opposite and toggled == 2)


class Rule008(ProgressRule):
    rule_id = "rule008"
    description = "The safe door unlocks only if knob is open and handle has been toggled an even number of times greater than one."

    def on_knob_change(self, state: SafeState, opened: bool) -> None:
        state.knob_open = opened
        self.update_lock(state)

    def on_handle_change(self, state: SafeState, opened: bool) -> None:
        state.handle_open = opened
        state.metadata["handle_toggles"] = state.metadata.get("handle_toggles", 0) + 1
        self.update_lock(state)

    def update_lock(self, state: SafeState) -> None:
        n = state.metadata.get("handle_toggles", 0)
        ok = state.knob_open and n > 1 and n % 2 == 0
        _set_progress(state, (int(state.knob_open) + min(n, 2) / 2.0) / 2.0)
        state.door_locked = not ok


class Rule009(ProgressRule):
    rule_id = "rule009"
    description = "The safe door unlocks after password 1001. Knob change inputs 0; handle change inputs 1."
    password = ["1", "0", "0", "1"]

    def on_knob_change(self, state: SafeState, opened: bool) -> None:
        state.knob_open = opened
        _append_buffer(state, "0", len(self.password))
        self.update_lock(state)

    def on_handle_change(self, state: SafeState, opened: bool) -> None:
        state.handle_open = opened
        _append_buffer(state, "1", len(self.password))
        self.update_lock(state)

    def update_lock(self, state: SafeState) -> None:
        _set_progress(state, len(state.input_buffer) / len(self.password))
        state.door_locked = state.input_buffer != self.password


class Rule010(ProgressRule):
    rule_id = "rule010"
    description = "The safe door unlocks when password 0110 is entered. Handle change while knob open inputs 1, while knob closed inputs 0."
    password = ["0", "1", "1", "0"]

    def on_handle_change(self, state: SafeState, opened: bool) -> None:
        _append_buffer(state, "1" if state.knob_open else "0", len(self.password))
        state.handle_open = opened
        self.update_lock(state)

    def on_knob_change(self, state: SafeState, opened: bool) -> None:
        state.knob_open = opened
        self.update_lock(state)

    def update_lock(self, state: SafeState) -> None:
        _set_progress(state, len(state.input_buffer) / len(self.password))
        state.door_locked = state.input_buffer != self.password


class Rule011(ProgressRule):
    rule_id = "rule011"
    description = "The safe door unlocks after password 1010. Knob closed-to-open inputs 1; open-to-closed inputs 0. Handle changes ignored."
    password = ["1", "0", "1", "0"]

    def on_knob_change(self, state: SafeState, opened: bool) -> None:
        if state.knob_open != opened:
            _append_buffer(state, "1" if opened else "0", len(self.password))
        state.knob_open = opened
        self.update_lock(state)

    def update_lock(self, state: SafeState) -> None:
        _set_progress(state, len(state.input_buffer) / len(self.password))
        state.door_locked = state.input_buffer != self.password


class Rule012(ProgressRule):
    rule_id = "rule012"
    description = "The safe door unlocks after knob and handle have been in all four open/closed combinations."

    def reset(self, state: SafeState) -> None:
        super().reset(state)
        state.metadata["combos"] = {"00": 1}

    def _mark(self, state: SafeState) -> None:
        combos = state.metadata.setdefault("combos", {})
        combos[("1" if state.knob_open else "0") + ("1" if state.handle_open else "0")] = 1

    def on_knob_change(self, state: SafeState, opened: bool) -> None:
        state.knob_open = opened
        self._mark(state)
        self.update_lock(state)

    def on_handle_change(self, state: SafeState, opened: bool) -> None:
        state.handle_open = opened
        self._mark(state)
        self.update_lock(state)

    def update_lock(self, state: SafeState) -> None:
        combos = state.metadata.get("combos", {})
        _set_progress(state, len(combos) / 4.0)
        state.door_locked = len(combos) < 4


class Rule013(ProgressRule):
    rule_id = "rule013"
    description = "The safe door unlocks when handle is open and knob has changed state a non-zero number of times divisible by 3."

    def on_knob_change(self, state: SafeState, opened: bool) -> None:
        if state.knob_open != opened:
            state.metadata["knob_toggles"] = state.metadata.get("knob_toggles", 0) + 1
        state.knob_open = opened
        self.update_lock(state)

    def update_lock(self, state: SafeState) -> None:
        n = state.metadata.get("knob_toggles", 0)
        _set_progress(state, (int(state.handle_open) + min(n, 3) / 3.0) / 2.0)
        state.door_locked = not (state.handle_open and n > 0 and n % 3 == 0)


class Rule014(ProgressRule):
    rule_id = "rule014"
    description = "The safe door remains locked unless knob is closed and handle is open; handle cannot be opened unless knob is open."

    def on_handle_change(self, state: SafeState, opened: bool) -> None:
        if opened and not state.knob_open:
            state.metadata["blocked_handle_attempts"] = state.metadata.get("blocked_handle_attempts", 0) + 1
        else:
            state.handle_open = opened
        self.update_lock(state)

    def update_lock(self, state: SafeState) -> None:
        _set_progress(state, (int(not state.knob_open) + int(state.handle_open)) / 2.0)
        state.door_locked = not ((not state.knob_open) and state.handle_open)


class Rule015(ProgressRule):
    rule_id = "rule015"
    description = "The safe door unlocks when knob open count equals handle close count, minimum 2 each; counts reset if either reaches 3."

    def on_knob_change(self, state: SafeState, opened: bool) -> None:
        if opened and not state.knob_open:
            state.metadata["knob_open_count"] = state.metadata.get("knob_open_count", 0) + 1
        state.knob_open = opened
        self.update_lock(state)

    def on_handle_change(self, state: SafeState, opened: bool) -> None:
        if (not opened) and state.handle_open:
            state.metadata["handle_close_count"] = state.metadata.get("handle_close_count", 0) + 1
        state.handle_open = opened
        self.update_lock(state)

    def update_lock(self, state: SafeState) -> None:
        ko = state.metadata.get("knob_open_count", 0)
        hc = state.metadata.get("handle_close_count", 0)
        if ko >= 3 or hc >= 3:
            ko = 0
            hc = 0
            state.metadata["knob_open_count"] = 0
            state.metadata["handle_close_count"] = 0
        _set_progress(state, (min(ko, 2) + min(hc, 2)) / 4.0)
        state.door_locked = not (ko == hc and ko >= 2 and hc >= 2)


class Rule016(ProgressRule):
    rule_id = "rule016"
    description = "The safe door unlocks after password 1110. Closing knob inputs 1; opening handle inputs 0."
    password = ["1", "1", "1", "0"]

    def on_knob_change(self, state: SafeState, opened: bool) -> None:
        if state.knob_open and not opened:
            _append_buffer(state, "1", len(self.password))
        state.knob_open = opened
        self.update_lock(state)

    def on_handle_change(self, state: SafeState, opened: bool) -> None:
        if opened and not state.handle_open:
            _append_buffer(state, "0", len(self.password))
        state.handle_open = opened
        self.update_lock(state)

    def update_lock(self, state: SafeState) -> None:
        _set_progress(state, len(state.input_buffer) / len(self.password))
        state.door_locked = state.input_buffer != self.password


class Rule017(ProgressRule):
    rule_id = "rule017"
    description = "The safe door unlocks when the product of knob and handle toggle counts equals 4."

    def on_knob_change(self, state: SafeState, opened: bool) -> None:
        if state.knob_open != opened:
            state.metadata["knob_toggles"] = state.metadata.get("knob_toggles", 0) + 1
        state.knob_open = opened
        self.update_lock(state)

    def on_handle_change(self, state: SafeState, opened: bool) -> None:
        if state.handle_open != opened:
            state.metadata["handle_toggles"] = state.metadata.get("handle_toggles", 0) + 1
        state.handle_open = opened
        self.update_lock(state)

    def update_lock(self, state: SafeState) -> None:
        k = state.metadata.get("knob_toggles", 0)
        h = state.metadata.get("handle_toggles", 0)
        _set_progress(state, min(k * h, 4) / 4.0)
        state.door_locked = (k * h) != 4


class Rule018(ProgressRule):
    rule_id = "rule018"
    description = "The safe door unlocks when password 123 is entered. Knob change increments a counter; handle change appends counter and resets."
    password = ["1", "2", "3"]

    def reset(self, state: SafeState) -> None:
        super().reset(state)
        state.metadata["counter"] = 1

    def on_knob_change(self, state: SafeState, opened: bool) -> None:
        if state.knob_open != opened:
            state.metadata["counter"] = state.metadata.get("counter", 1) + 1
        state.knob_open = opened
        self.update_lock(state)

    def on_handle_change(self, state: SafeState, opened: bool) -> None:
        if state.handle_open != opened:
            _append_buffer(state, str(state.metadata.get("counter", 1)), len(self.password))
            state.metadata["counter"] = 1
        state.handle_open = opened
        self.update_lock(state)

    def update_lock(self, state: SafeState) -> None:
        _set_progress(state, len(state.input_buffer) / len(self.password))
        state.door_locked = state.input_buffer != self.password


class Rule019(ProgressRule):
    rule_id = "rule019"
    description = "The safe door unlocks after password 01 via knob interactions while handle is open; handle closes to unlock."
    password = ["0", "1"]

    def reset(self, state: SafeState) -> None:
        super().reset(state)
        state.metadata["password_ok"] = 0
        state.metadata["recording"] = 0

    def on_knob_change(self, state: SafeState, opened: bool) -> None:
        if state.handle_open:
            _append_buffer(state, "1" if opened else "0", len(self.password))
            state.metadata["password_ok"] = int(state.input_buffer == self.password)
        state.knob_open = opened
        self.update_lock(state)

    def on_handle_change(self, state: SafeState, opened: bool) -> None:
        state.handle_open = opened
        state.metadata["recording"] = int(opened)
        self.update_lock(state)

    def update_lock(self, state: SafeState) -> None:
        _set_progress(state, (len(state.input_buffer) / len(self.password) + state.metadata.get("password_ok", 0)) / 2.0)
        state.door_locked = not (bool(state.metadata.get("password_ok", 0)) and not state.handle_open)


class Rule020(Rule019):
    rule_id = "rule020"
    description = "The safe door unlocks after password 11 via knob interactions while handle is open; handle closes to unlock."
    password = ["1", "1"]


ALL_RULE_CLASSES = [
    Rule001, Rule002, Rule003, Rule004, Rule005, Rule006, Rule007, Rule008, Rule009, Rule010,
    Rule011, Rule012, Rule013, Rule014, Rule015, Rule016, Rule017, Rule018, Rule019, Rule020,
]
