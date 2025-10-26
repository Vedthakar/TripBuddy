from __future__ import annotations
import os, json, pathlib, base64
from typing import Optional, Union, Dict, Any, List

# --- NEW MINIMAL ITINERARY FORMAT ---
# The target structure is List[List[Dict[str, Any]]] where the outer list
# is the days, and the inner list is the events.

# Helper to generate a single event object in the required format
def _make_event(title: str, time: str, duration: int, isCompleted: bool = False) -> Dict[str, Any]:
    return {
        "title": title,
        "time": time,
        "duration": duration,
        "isCompleted": isCompleted
    }

def update_itinerary(current_itinerary: List[List[Dict[str, Any]]], transcript: str) -> List[List[Dict[str, Any]]]:
    """DUMMY: Simulates Gemini update. Must be replaced with actual call."""
    print(f"**DUMMY GEMINI CALL: UPDATE** Command: '{transcript}'")

    updated = current_itinerary[:] # Copy the outer list

    # Simple simulation of removing Louvre and adding cooking class (Test 2)
    # This targets Day 2 (index 1) in the current itinerary.
    if len(updated) > 1 and "louvre" in transcript.lower() and "cooking class" in transcript.lower():
        # Replace all events on Day 2
        updated[1] = [
            _make_event("Morning Cooking Class", "9:00 AM", 180),
            _make_event("Lunch After Class", "12:30 PM", 60),
            _make_event("Free Afternoon", "2:00 PM", 240)
        ]
        return updated

    return current_itinerary

def create_initial_itinerary_flexible(transcript: str) -> List[List[Dict[str, Any]]]:
    """DUMMY: Simulates Gemini creation. Must be replaced with actual call."""
    print(f"**DUMMY GEMINI CALL: CREATE** Command: '{transcript}'")

    # Simple simulation of creating a 5-day Rome itinerary (Test 1)
    return [
        # Day 1
        [_make_event("Arrive in Rome", "2:00 PM", 60), _make_event("Dinner near hotel", "7:00 PM", 90)],
        # Day 2
        [_make_event("Colosseum & Forum Tour", "9:00 AM", 240), _make_event("Pasta Making Class", "5:00 PM", 180)],
        # Day 3
        [_make_event("Vatican City Visit", "8:30 AM", 300), _make_event("Gelato tasting", "4:00 PM", 60)],
        # Day 4
        [_make_event("Trevi Fountain & Pantheon", "10:00 AM", 180), _make_event("Evening Walk", "8:00 PM", 90)],
        # Day 5
        [_make_event("Check out & Airport Transfer", "10:00 AM", 120)]
    ]

# --------------------------------------------
# ElevenLabs STT — paste your call below
# --------------------------------------------
def _stt_with_elevenlabs(audio_bytes: bytes, *, api_key: Optional[str] = None) -> str:
    """
    Return transcript string from raw audio bytes using ElevenLabs.
    >>> Replace the body with YOUR working ElevenLabs STT snippet <<<
    """
    raise NotImplementedError("Paste your ElevenLabs STT call in _stt_with_elevenlabs()")


# --------------------------------------------
# One-shot method (The function under test)
# --------------------------------------------
def speech_to_itinerary_json(
        *,
        audio: Union[bytes, str, pathlib.Path, None] = None,
        # Updated type hint for current_itinerary
        current_itinerary: Optional[List[List[Dict[str, Any]]]] = None,
        text_command: Optional[str] = None,
        elevenlabs_api_key: Optional[str] = None,
) -> str:
    """
    One method: speech/audio -> STT -> Gemini -> finished itinerary JSON string.
    [... docstring omitted for brevity ...]
    """
    if text_command is None and audio is None:
        raise ValueError("Provide either 'audio' (bytes or file path) or 'text_command'.")

    # 1) Get transcript (use provided text or run STT)
    if text_command is not None:
        transcript = str(text_command)
    # Omitted audio processing for brevity, assume it works or is skipped
    else:
        # Normalize audio to bytes...
        if isinstance(audio, (str, pathlib.Path)):
            p = pathlib.Path(audio)
            if not p.exists():
                raise ValueError(f"Audio file not found: {p}")
            audio_bytes = p.read_bytes()
        elif isinstance(audio, (bytes, bytearray)):
            audio_bytes = bytes(audio)
        else:
            raise ValueError("Invalid 'audio' type. Use bytes or a file path.")

        try:
            transcript = _stt_with_elevenlabs(audio_bytes, api_key=elevenlabs_api_key)
        except NotImplementedError as e:
            raise RuntimeError(f"STT not wired: {e}") from e
        except Exception as e:
            raise RuntimeError(f"ElevenLabs STT failed: {e.__class__.__name__}: {e}") from e

    # 2) Send to Gemini: update existing itinerary OR create a new one
    try:
        if current_itinerary:
            updated = update_itinerary(current_itinerary, transcript)
        else:
            updated = create_initial_itinerary_flexible(transcript)
    except Exception as e:
        raise RuntimeError(f"Gemini call failed: {e.__class__.__name__}: {e}") from e

    # The updated object must be a list (the array of arrays)
    if not isinstance(updated, list):
        raise RuntimeError("Gemini did not return the expected List[List[...]] structure.")
    if not updated or not isinstance(updated[0], list):
        raise RuntimeError("Gemini returned an empty or invalid list structure.")

    # 3) Return a finished JSON string
    # We use separators for compact output, as requested.
    return json.dumps(updated, ensure_ascii=False, separators=(",", ":"))


# --------------------------------------------
# TESTS (Run directly via python voice_flow.py)
# --------------------------------------------

def run_test_cases():
    """Defines and runs the two key text-based test cases."""

    if not os.getenv("GEMINI_API_KEY"):
        print("WARNING: GEMINI_API_KEY environment variable is not set. Using DUMMY functions.")

    # --- Setup for Test Case 2: Update Itinerary (New Minimal Format) ---
    CURRENT_PLAN = [
        # Day 1
        [_make_event("Eiffel Tower", "3:00 PM", 120)],
        # Day 2
        [_make_event("Louvre Museum", "10:00 AM", 180)], # <--- This is the target for removal
        # Day 3
        [_make_event("Arc de Triomphe", "9:00 AM", 60)]
    ]

    # TEST 1: Creation Pathway (current_itinerary=None)
    print("==================================================")
    print("   TEST 1: CREATE NEW ITINERARY (Rome, 5 Days)")
    print("==================================================")
    try:
        new_itinerary_json = speech_to_itinerary_json(
            text_command="Plan a five-day itinerary for two people in Rome, Italy.",
            current_itinerary=None
        )
        print("SUCCESS: New Itinerary Created.")
        # Pretty print the JSON output for human review
        print(json.dumps(json.loads(new_itinerary_json), indent=2))
        print(f"Result is a list with {len(json.loads(new_itinerary_json))} days.")
    except Exception as e:
        print(f"FAILURE: Test 1 failed with error: {e}")


    # TEST 2: Update Pathway (current_itinerary provided)
    print("\n==================================================")
    print("   TEST 2: UPDATE EXISTING ITINERARY (Paris Plan)")
    print("==================================================")
    try:
        print(f"Starting Plan has {len(CURRENT_PLAN)} days. Day 2 Event: {CURRENT_PLAN[1][0]['title']}")

        updated_itinerary_json = speech_to_itinerary_json(
            text_command="I don't like museums. Can you remove the Louvre and add a cooking class instead?",
            current_itinerary=CURRENT_PLAN
        )
        print("SUCCESS: Itinerary Updated.")
        updated_plan = json.loads(updated_itinerary_json)
        print(json.dumps(updated_plan, indent=2))

        # Validation Check: Ensure the update logic worked
        day_2_event_title = updated_plan[1][0]["title"].lower()
        if "louvre" not in day_2_event_title and "cook" in day_2_event_title:
            print("Validation Check Passed: Day 2 event successfully replaced.")
        else:
            print("Validation Check Failed: Day 2 event title did not change as expected.")

    except Exception as e:
        print(f"FAILURE: Test 2 failed with error: {e}")

if __name__ == "__main__":
    run_test_cases()
