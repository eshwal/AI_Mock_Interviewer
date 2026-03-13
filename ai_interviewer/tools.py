import time
from google.adk.tools import ToolContext

_state = {
    "start_time": None,
    "role": None,
    "jd": None,
    "technical_notes": [],
    "behavioral_notes": [],
    "screen_observations": [],
    "base_duration": 30 * 60,
    "grace_period": 5 * 60,
    "complete": False,
    "coding_started_time": None,
    "coding_phase_started": False,
    "code_complete_confirmed": False,
    "candidate_spoken": False,
    "first_concept_asked": False,
    "screen_has_code": False,
    "last_screen_check_time": None,
    "screen_check_count": 0,
    "timer_checked": False,  # True after check_timer() called — gates end_interview
    "last_candidate_speech_time": None,  # updated on each stream_end
    "closing_spoken": False,  # set True after turn_complete following check_timer — gates end_interview
}


def reset_state(base_duration: int = 30 * 60):
    _state.update({
        "start_time": None,
        "role": None,
        "jd": None,
        "technical_notes": [],
        "behavioral_notes": [],
        "screen_observations": [],
        "complete": False,
        "coding_started_time": None,
        "coding_phase_started": False,
        "code_complete_confirmed": False,
        "candidate_spoken": False,
        "first_concept_asked": False,
        "screen_has_code": False,
        "last_screen_check_time": None,
        "screen_check_count": 0,
        "base_duration": base_duration,
        "timer_checked": False,
        "last_candidate_speech_time": None,
        "closing_spoken": False,
    })


def mark_first_concept_asked():
    _state["first_concept_asked"] = True


def save_interview_context(role: str, jd: str, tool_context: ToolContext) -> str:
    if not _state["candidate_spoken"]:
        return "BLOCKED: candidate has not spoken yet."
    _state["start_time"] = time.time()
    _state["role"] = role
    _state["jd"] = jd
    return "SAVED."


def start_coding_phase(tool_context: ToolContext) -> str:
    if len(_state["behavioral_notes"]) < 2:
        return f"BLOCKED: only {len(_state['behavioral_notes'])} concept answers logged. Need at least 2."
    if _state["coding_phase_started"]:
        return "BLOCKED: already started."
    _state["coding_phase_started"] = True
    _state["coding_started_time"] = time.time()
    _state["screen_check_count"] = 0
    return "CODING_PHASE_STARTED."


def report_screen_state(has_code: bool, is_complete: bool, description: str, tool_context: ToolContext) -> str:
    if not _state["coding_phase_started"]:
        return "BLOCKED: call start_coding_phase() first."

    now = time.time()
    last = _state["last_screen_check_time"]
    _state["screen_check_count"] += 1
    if _state["screen_check_count"] > 2 and last and (now - last) < 30:
        return "BLOCKED: checked too recently. Wait."
    _state["last_screen_check_time"] = now
    _state["screen_has_code"] = has_code
    ts = round(now - (_state["start_time"] or now), 1)

    is_lookup = any(w in description.lower() for w in [
        "google", "search", "browser", "stackoverflow", "chatgpt", "copilot",
        "wikipedia", "docs", "tab", "url", "website", "youtube"
    ])
    if is_lookup:
        _state["screen_observations"].append({"time_elapsed_s": ts, "type": "external_lookup", "description": description})
        return "LOOKUP_DETECTED."

    if not has_code:
        _state["screen_observations"].append({"time_elapsed_s": ts, "type": "no_code", "description": description})
        return "NO_CODE."

    _state["screen_observations"].append({"time_elapsed_s": ts, "type": "code_visible", "description": description})

    if not is_complete:
        return "IN_PROGRESS."

    _state["code_complete_confirmed"] = True
    return "CODE_COMPLETE."


def log_behavioral_note(note: str, tool_context: ToolContext) -> str:
    if not _state["start_time"]:
        return "BLOCKED: interview not started yet."
    if not _state["first_concept_asked"]:
        return "BLOCKED: no concept question asked yet."
    ts = round(time.time() - _state["start_time"], 1)
    _state["behavioral_notes"].append({"time_elapsed_s": ts, "note": note})
    return f"NOTED. ({len(_state['behavioral_notes'])} total)"


def log_technical_note(note: str, tool_context: ToolContext) -> str:
    """Log anything observed during the coding phase — approach, complexity, mistakes, strengths."""
    if not _state["coding_phase_started"]:
        return "BLOCKED: coding phase not started."
    ts = round(time.time() - (_state["start_time"] or time.time()), 1)
    _state["technical_notes"].append({"time_elapsed_s": ts, "note": note})
    return "NOTED."


def check_timer(tool_context: ToolContext) -> str:
    if not _state["start_time"]:
        return "TIMER_NOT_STARTED."
    if not _state["coding_phase_started"]:
        return "BLOCKED: complete concept phase and call start_coding_phase() before checking timer."
    elapsed   = time.time() - _state["start_time"]
    base      = _state["base_duration"]
    hard      = base + _state["grace_period"]
    remaining = base - elapsed

    if elapsed >= hard:
        _state["timer_checked"] = True
        return "FORCE_END."
    if elapsed >= base:
        _state["timer_checked"] = True
        left = int((hard - elapsed) // 60)
        return f"WRAP_NOW: {left}m left."
    warn_threshold = max(60, base * 0.20)
    if remaining <= warn_threshold:
        _state["timer_checked"] = True
        m, s = int(remaining // 60), int(remaining % 60)
        return f"WARNING: {m}m {s}s left."
    _state["timer_checked"] = True
    m, s = int(remaining // 60), int(remaining % 60)
    return f"OK: {m}m {s}s remaining — continue the interview, do not end yet."


def end_interview(tool_context: ToolContext) -> str:
    if len(_state["behavioral_notes"]) < 2:
        return (f"BLOCKED: only {len(_state['behavioral_notes'])} concept answers logged. "
                "You must complete the concept phase (ask at least 2 questions and log their answers) before ending.")
    if not _state["coding_phase_started"]:
        return "BLOCKED: coding phase never started. Give the candidate a coding problem first."
    if not _state["technical_notes"]:
        return "BLOCKED: no technical notes logged yet. Assess the candidate's code first."
    if not _state["timer_checked"]:
        return "BLOCKED: call check_timer() first, then speak your closing sentence, then call end_interview()."
    if not _state.get("closing_spoken"):
        return "BLOCKED: you must speak your closing sentence to the candidate FIRST, then call end_interview() in the next turn."
    _state["complete"] = True
    return "INTERVIEW_COMPLETE."


def reset_screen_check():
    _state["screen_check_count"] = 0
    _state["last_screen_check_time"] = None


def get_all_notes() -> dict:
    return {
        "role": _state.get("role", "Unknown"),
        "jd": _state.get("jd", ""),
        "technical_notes": _state["technical_notes"],
        "behavioral_notes": _state["behavioral_notes"],
        "screen_observations": _state["screen_observations"],
        "elapsed_seconds": round(time.time() - _state["start_time"]) if _state["start_time"] else 0,
    }