# main.py - AI Mock Interviewer backend
#
# Uses client.aio.live.connect() directly instead of ADK run_live because
# ADK's wrapper was swallowing VAD events and sessions went silent after turn 1.
#
# A few things that burned me and are worth noting:
# - send_client_content mid-response = 1011 deadline error, gate on not ai_speaking
# - Keepalive needs actual audio bytes, not just pings. Silent PCM chunk works.
# - Default Gemini VAD didn't fire reliably on this model so VAD is done server-side
#   via RMS on raw PCM. Barge-in is also manual for the same reason.
# - Keepalive must not fire before candidate has spoken once (stream_ended_once gate)
#   otherwise Gemini treats the silence as a completed turn and responds to itself.
import os, re, json, base64, asyncio, uuid, logging, struct, math, time as _time
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("CRITICAL: GOOGLE_API_KEY not found!")
else:
    print(f"✅ API Key: {api_key[:4]}****")

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from google import genai
from google.genai import types
from google.genai.types import Content, Part

from ai_interviewer.agent import  SYSTEM_INSTRUCTION, TOOL_DECLARATIONS
from ai_interviewer.tools import (
    reset_state, get_all_notes, _state as tool_state,
    save_interview_context, report_screen_state,
    check_timer, log_behavioral_note, log_technical_note,
    start_coding_phase, end_interview, reset_screen_check,
    mark_first_concept_asked,
)

logging.basicConfig(level=logging.INFO)
logging.getLogger("google.adk").setLevel(logging.WARNING)

current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"

# demo mode config
DEMO_MODE = os.environ.get("DEMO_MODE", "0") == "1"
if DEMO_MODE:
    print("🎬 DEMO MODE — short interview (~5 min)")
    INTERVIEW_MINUTES = 5
    HARD_STOP         = 8 * 60
else:
    INTERVIEW_MINUTES = 30
    HARD_STOP         = 35 * 60

# VAD thresholds
# AudioWorklet sends 4096-sample chunks at 16kHz = 256ms per chunk
SILENCE_RMS    = 400   # RMS below this = silence (raised from 200 — 261/320 were false triggers)
SILENCE_CHUNKS = 10    # ~2.5s of silence before stream_end (was 6/1.5s — too aggressive, cut off mid-answer)

# Barge-in: RMS must be above BARGE_MIN_RMS for BARGE_THRESHOLD consecutive chunks
BARGE_THRESHOLD = 3
BARGE_MIN_RMS   = 600  # higher than SILENCE_RMS to avoid echo triggering barge-in

# Keepalive: send silence every N seconds to prevent Gemini 1011 idle timeout
KEEPALIVE_INTERVAL = 20
# 20ms of silence at 16kHz 16-bit = 320 bytes
KEEPALIVE_CHUNK = bytes(320)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

genai_client = genai.Client(api_key=api_key)


def pcm_rms(data: bytes) -> float:
    if len(data) < 2:
        return 0.0
    count = len(data) // 2
    samples = struct.unpack_from(f"<{count}h", data)
    return math.sqrt(sum(s * s for s in samples) / count)


def dispatch_tool(name: str, args: dict) -> str:
    print(f"🔧 Tool: {name}({args})")
    try:
        result = _dispatch_tool_inner(name, args)
        if "BLOCKED" in result or "Already" in result:
            print(f"   ⚠️  Tool result: {result[:80]}")
        return result
    except Exception as e:
        return f"Tool error: {e}"


def _dispatch_tool_inner(name: str, args: dict) -> str:
    try:
        if name == "save_interview_context":
            return save_interview_context(role=args["role"], jd=args["jd"], tool_context=None)
        if name == "report_screen_state":
            has_code = args["has_code"]
            if isinstance(has_code, str):
                has_code = has_code.lower() == "true"
            is_complete = args.get("is_complete", False)
            if isinstance(is_complete, str):
                is_complete = is_complete.lower() == "true"
            return report_screen_state(has_code=bool(has_code), is_complete=bool(is_complete), description=args["description"], tool_context=None)
        if name == "check_timer":
            return check_timer(tool_context=None)
        if name == "log_behavioral_note":
            return log_behavioral_note(note=args["note"], tool_context=None)
        if name == "log_technical_note":
            return log_technical_note(note=args["note"], tool_context=None)
        if name == "start_coding_phase":
            return start_coding_phase(tool_context=None)
        if name == "end_interview":
            return end_interview(tool_context=None)
        return f"Unknown tool: {name}"
    except Exception as e:
        return f"Tool error: {e}"


def clean_transcript(s: str) -> str:
    s = re.sub(r'<ctrl\d+>', '', s)
    s = "".join(c for c in s if ord(c) >= 32 or c == "\n")
    # Strip function-call syntax: tool_name(...)
    s = re.sub(r'\b(log_behavioral_note|log_technical_note|save_interview_context|start_coding_phase|report_screen_state|check_timer|end_interview)\s*\([^)]*\)\.?', '', s)
    # Strip spoken/natural-language variants the model sometimes vocalises
    s = re.sub(r'\b(check(?:ing)? timer|log behavioral note|log technical note|end interview|start coding phase)\.?\s*', '', s, flags=re.IGNORECASE)
    # Strip leftover JSON-like fragments
    s = re.sub(r'\{["\'](?:note|role|jd|has_code|is_complete|description)["\']\s*:', '', s)
    # Collapse multiple spaces
    s = re.sub(r' {2,}', ' ', s)
    return s.strip()


def dedup_sentences(text: str) -> str:
    """Remove immediately-repeated sentences (model sometimes says closing twice)."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    result = []
    seen = set()
    for p in parts:
        key = p.strip().lower().rstrip('.!?')
        if key and key not in seen:
            seen.add(key)
            result.append(p)
    return ' '.join(result)


async def build_report(session_id: str) -> str:
    notes = get_all_notes()
    print(f"📋 build_report snapshot: role={notes.get('role')}, "
          f"behavioral={len(notes['behavioral_notes'])}, "
          f"technical={len(notes['technical_notes'])}, "
          f"elapsed={notes['elapsed_seconds']}s")
    m, s = notes["elapsed_seconds"] // 60, notes["elapsed_seconds"] % 60
    obs  = notes.get("screen_observations", [])

    from ai_interviewer.agent import scorer_agent
    system_prompt = scorer_agent.instruction

    user_prompt = (
        f"Role: {notes['role']} | Stack: {notes['jd']} | Duration: {m}m {s}s\n\n"
        f"Behavioral notes:\n{json.dumps(notes['behavioral_notes'], indent=2) or 'None recorded'}\n\n"
        f"Technical notes:\n{json.dumps(notes['technical_notes'], indent=2) or 'None recorded'}\n\n"
        f"Screen observations:\n{json.dumps(obs, indent=2) or 'None recorded'}\n\n"
        "Write the complete feedback report covering ALL sections. "
        "Do not truncate — write every section in full."
    )

    # Call Gemini directly (not via ADK runner) so max_output_tokens is respected
    response = await genai_client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=[{"role": "user", "parts": [{"text": user_prompt}]}],
        config={
            "system_instruction": system_prompt,
            "temperature": 0.2,
            "max_output_tokens": 4096,
        }
    )
    text = ""
    for part in (response.candidates[0].content.parts if response.candidates else []):
        if hasattr(part, "text") and part.text:
            text += part.text
    return text.strip() or "Report generation failed."


@app.websocket("/live-ws/{session_id}")
async def interview_ws(websocket: WebSocket, session_id: str):
    await websocket.accept()
    sid = f"{session_id}_{uuid.uuid4().hex[:8]}"
    print(f"🚀 Session: {sid}")
    reset_state(base_duration=INTERVIEW_MINUTES * 60)

    ended              = asyncio.Event()
    report_sent        = asyncio.Event()
    stream_ended_once  = asyncio.Event()  # gates keepalive — set after first real stream_end
    ai_first_turn_done = asyncio.Event()  # set when AI greeting completes — gates stream_end

    # ai_speaking: True while Gemini audio chunks arriving.
    # Blocks normal stream_end (prevents 1011). Flipped False on barge-in before stream_end.
    ai_speaking = False
    ai_stopped_at = [_time.monotonic()]  # init to now so cooldown is active from session start
    AI_COOLDOWN   = 2.0          # seconds after AI stops before stream_end is allowed
    candidate_loud_chunks  = [0]   # list — mutated across nested async functions
    candidate_speech_seen_l = [False]  # list — mutated across nested async functions

    # tool_pending: True while recv_loop is processing a tool call + sending tool_response.
    # send_loop skips audio chunks while True to prevent 1011 deadline race.
    # Using a list so mutation is visible across nested async functions without nonlocal.
    tool_pending = [False]

    # Queue types: "audio" | "video" | "stream_end" | "text"
    to_gemini: asyncio.Queue = asyncio.Queue(maxsize=300)

    last_user_activity = _time.monotonic()

    system_instruction = SYSTEM_INSTRUCTION
    if DEMO_MODE:
        system_instruction += (
            "\n\n━━━ DEMO MODE — OVERRIDES ALL EARLIER INSTRUCTIONS ━━━\n"
            f"This is a {INTERVIEW_MINUTES}-minute demo. The '4 to 6 concept exchanges' rule does NOT apply here.\n"
            "  - Ask exactly 2 concept questions only, then move straight to coding.\n"
            "  - After 2 concept questions, say 'Let's move to a coding problem.' immediately.\n"
            "  - Keep all your turns to 15 words maximum.\n"
            "  - The whole interview — concepts + coding + wrap up — must fit in 5 minutes.\n"
            "  - Do NOT linger on any topic. Keep the pace brisk.\n"
        )

    live_config = {
        "response_modalities": ["AUDIO"],
        "system_instruction": system_instruction,
        "tools": TOOL_DECLARATIONS,
        "speech_config": {
            "voice_config": {"prebuilt_voice_config": {"voice_name": "Puck"}}
        },
        "output_audio_transcription": {},
        "thinking_config": {"thinking_budget": 0},
    }

    # called when end_interview tool fires or hard stop hits
    async def finish_interview():
        print("🏁 Generating report...")
        try:
            await websocket.send_json({"status": "generating_report"})
            report = await build_report(sid)
            await websocket.send_json({"report": report})
            print("✅ Report sent.")
        except Exception as e:
            print(f"❌ Report error: {e}")
            try:
                await websocket.send_json({"report": "## Report Unavailable"})
            except Exception:
                pass
        finally:
            report_sent.set()
            ended.set()
            await to_gemini.put(None)

    # opens Gemini Live session, reconnects on 1011
    async def run_live():
        nonlocal ai_speaking
        ai_buf = []
        last_ai_utterance = [""]  # list so it's mutable from recv_loop closure
        RECONNECT_LIMIT = 3
        reconnect_count = 0

        while not ended.is_set():
            try:
                async with genai_client.aio.live.connect(model=MODEL, config=live_config) as session:

                    if reconnect_count > 0:
                        print(f"🔄 Reconnected ({reconnect_count}/{RECONNECT_LIMIT})")
                        from ai_interviewer.tools import _state as ts
                        if ts.get("role"):
                            ctx = (
                                f"[SYSTEM] Session reconnected. You are continuing the interview "
                                f"with a {ts['role']}. Pick up exactly where you left off. "
                                f"Do not re-introduce yourself."
                            )
                            await session.send_client_content(
                                turns=[Content(role="user", parts=[Part(text=ctx)])],
                                turn_complete=True
                            )
                    else:
                        print("✅ Gemini Live session open")
                        # Explicit greeting instruction — prevents model from hallucinating
                        # tool calls before any conversation has happened.
                        # Must name the exact words so model doesn't improvise a long intro.
                        await session.send_client_content(
                            turns=[Content(role="user", parts=[Part(
                                text="[SYSTEM] Begin the interview. "
                                     "Say exactly: 'Hello! What role and tech stack are you preparing for today?' "
                                     "Then stop and wait for the candidate to answer. Do not call any tools yet."
                            )])],
                            turn_complete=True
                        )

                    # send audio/video/stream_end/text to Gemini
                    async def send_loop():
                        while not ended.is_set():
                            try:
                                item = await asyncio.wait_for(to_gemini.get(), timeout=0.5)
                            except asyncio.TimeoutError:
                                continue
                            if item is None:
                                break
                            kind, data = item
                            try:
                                if kind == "audio":
                                    if tool_pending[0]:
                                        continue  # drop audio while tool response in flight
                                    await session.send_realtime_input(
                                        audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000")
                                    )
                                elif kind == "video":
                                    # Only send screen frames once coding phase has started
                                    # Sending frames during concept phase causes model to
                                    # hallucinate a coding session if it sees code on screen
                                    from ai_interviewer.tools import _state as _ts
                                    if _ts.get("coding_phase_started"):
                                        await session.send_realtime_input(
                                            video=types.Blob(data=data, mime_type="image/jpeg")
                                        )
                                elif kind == "stream_end":
                                    if tool_pending[0]:
                                        print("🔇 stream_end dropped (tool pending)")
                                        continue  # drop stream_end while tool response in flight
                                    print("🔇 Sending audio_stream_end")
                                    await session.send_realtime_input(audio_stream_end=True)
                                elif kind == "text":
                                    # send_client_content for text nudges — safe when not ai_speaking.
                                    # send_realtime_input(text=) causes Gemini to echo/repeat responses.
                                    if not ai_speaking:
                                        await session.send_client_content(
                                            turns=[Content(role="user", parts=[Part(text=data.decode())])],
                                            turn_complete=True
                                        )
                                    else:
                                        print("⚠️ Text nudge dropped (AI speaking)")
                            except Exception as e:
                                err = str(e)
                                if "1011" in err or "Deadline" in err:
                                    print("⚠️ send 1011 — reconnecting")
                                else:
                                    print(f"⚠️ send error: {e}")
                                break

                    # receive audio chunks, transcripts, tool calls from Gemini
                    async def recv_loop():
                        while not ended.is_set():
                            try:
                                async for msg in session.receive():
                                    if ended.is_set():
                                        return

                                    if msg.data:
                                        ai_speaking = True
                                        try:
                                            await websocket.send_json({
                                                "audio": base64.b64encode(msg.data).decode()
                                            })
                                        except Exception:
                                            return

                                    sc = msg.server_content
                                    if sc:
                                        if sc.output_transcription:
                                            t = clean_transcript(sc.output_transcription.text or "")
                                            if t and (not ai_buf or ai_buf[-1] != t):
                                                ai_buf.append(t)

                                        if sc.interrupted:
                                            ai_speaking = False
                                            ai_stopped_at[0] = _time.monotonic()
                                            ai_buf.clear()
                                            print("⚡ Barge-in confirmed")
                                            try:
                                                await websocket.send_json({"interrupted": True})
                                            except Exception:
                                                return

                                        if sc.turn_complete:
                                            ai_speaking = False
                                            ai_stopped_at[0] = _time.monotonic()
                                            # closing_spoken is set AFTER we see the transcript below
                                            # (we need the text to confirm it's actually a closing sentence)
                                            from ai_interviewer.tools import _state as _tool_state_tc
                                            if not ai_first_turn_done.is_set():
                                                # Greeting just finished — reset any pre-greeting noise
                                                # that may have falsely set candidate_speech_seen
                                                candidate_speech_seen_l[0] = False
                                                candidate_loud_chunks[0]   = 0
                                            else:
                                                # Any turn after greeting = model asked a question
                                                mark_first_concept_asked()
                                            ai_first_turn_done.set()
                                            ts = __import__('time').strftime('%H:%M:%S')
                                            print(f"🔄 turn_complete [{ts}]")
                                            if ai_buf:
                                                full = " ".join(ai_buf).strip()
                                                ai_buf.clear()
                                                if full:
                                                    # Auto-rescue spoken tool calls — model sometimes
                                                    # verbalises log_behavioral_note() instead of calling it
                                                    spoken_note = re.search(
                                                        r'log_behavioral_note\s*\(\s*["\'](.*?)["\']\s*\)',
                                                        full, re.DOTALL
                                                    )
                                                    if spoken_note:
                                                        from ai_interviewer.tools import _state as _ts
                                                        import time as _tt
                                                        note_text = spoken_note.group(1)
                                                        if _ts.get("first_concept_asked") and _ts.get("start_time"):
                                                            elapsed = round(_tt.time() - _ts["start_time"])
                                                            _ts["behavioral_notes"].append({"time_elapsed_s": elapsed, "note": note_text})
                                                            print(f"🔧 [auto-rescued spoken] log_behavioral_note: {note_text[:60]}")
                                                    # Clean and dedup for display
                                                    display = dedup_sentences(clean_transcript(full))
                                                    if display:
                                                        # Set closing_spoken only when model says actual closing words
                                                        # (not just any turn after check_timer — that was too loose)
                                                        CLOSING_WORDS = ("thank", "time today", "conclude", "wrap up", "that's all", "good luck")
                                                        if (_tool_state_tc.get("timer_checked")
                                                                and not _tool_state_tc.get("complete")
                                                                and any(w in display.lower() for w in CLOSING_WORDS)):
                                                            _tool_state_tc["closing_spoken"] = True
                                                            print("✅ closing_spoken set")
                                                        # Detect duplicate turns — model sometimes repeats
                                                        # the same sentence as a new audio turn
                                                        # Once interview complete, suppress any trailing AI turns
                                                        if _tool_state_tc.get("complete"):
                                                            print(f"🔇 [post-complete suppressed] {display[:60]}")
                                                            try:
                                                                await websocket.send_json({"interrupted": True})
                                                            except Exception:
                                                                pass
                                                            continue
                                                        prev = last_ai_utterance[0].lower().strip().rstrip(".!?")
                                                        curr = display.lower().strip().rstrip(".!?")
                                                        is_dup = (prev and curr and
                                                                  (curr == prev or curr in prev or prev in curr))
                                                        if is_dup:
                                                            print(f"🔇 [dup suppressed] {display[:60]}")
                                                            # Tell browser to discard the audio it just received
                                                            try:
                                                                await websocket.send_json({"interrupted": True})
                                                            except Exception:
                                                                pass
                                                        else:
                                                            last_ai_utterance[0] = display
                                                            print(f"🤖 [{ts}] {display}")
                                                            try:
                                                                await websocket.send_json({
                                                                    "transcript": "ai",
                                                                    "text": display,
                                                                    "ts": ts
                                                                })
                                                            except Exception:
                                                                return

                                    if msg.tool_call:
                                        tool_pending[0] = True   # block send_loop audio during tool round-trip
                                        # Yield to event loop so send_loop sees tool_pending=True
                                        # and stops sending audio before we drain + send_tool_response
                                        await asyncio.sleep(0.02)
                                        responses = []
                                        batch_names = [fc.name for fc in msg.tool_call.function_calls]
                                        # If check_timer and end_interview are in the same batch,
                                        # end_interview must be skipped — model hasn't spoken closing yet
                                        same_batch_end_blocked = (
                                            "check_timer" in batch_names and "end_interview" in batch_names
                                        )
                                        for fc in msg.tool_call.function_calls:
                                            if fc.name == "end_interview" and same_batch_end_blocked:
                                                result = "BLOCKED: call check_timer() first, speak your closing sentence, then call end_interview() in the next turn."
                                            else:
                                                result = dispatch_tool(fc.name, dict(fc.args or {}))
                                            responses.append(
                                                types.FunctionResponse(
                                                    name=fc.name, id=fc.id,
                                                    response={"result": result}
                                                )
                                            )
                                            if fc.name == "end_interview" and result == "INTERVIEW_COMPLETE." and not ended.is_set():
                                                asyncio.create_task(finish_interview())
                                        # Drain all audio + stream_end from local queue.
                                        # Audio already in-flight to Gemini can't be recalled,
                                        # but the 0.05s sleep above stops new audio being sent.
                                        drained = 0
                                        while True:
                                            try:
                                                item = to_gemini.get_nowait()
                                                if item and item[0] in ("audio", "stream_end"):
                                                    drained += 1
                                                elif item:
                                                    await to_gemini.put(item)  # keep video/text
                                            except asyncio.QueueEmpty:
                                                break
                                        if drained:
                                            print(f"🧹 Drained {drained} stale items before tool response")
                                        print(f"📤 Sending tool_response ({len(responses)} responses)")
                                        await session.send_tool_response(function_responses=responses)
                                        print(f"📤 tool_response sent OK")
                                        await asyncio.sleep(0.15)  # let Gemini process before unblocking
                                        tool_pending[0] = False  # unblock audio

                            except Exception as e:
                                err = str(e)
                                if "1011" in err or "Deadline" in err:
                                    print("⚠️ recv 1011 — reconnecting")
                                else:
                                    print(f"⚠️ recv error: {e}")
                                return

                    # keepalive - silent audio to prevent Gemini idle disconnect
                    # only starts after candidate has spoken once - before that
                    # it causes Gemini to treat the silence as a completed user turn
                    async def keepalive_loop():
                        last_sent = _time.monotonic()
                        while not ended.is_set():
                            await asyncio.sleep(2)
                            now = _time.monotonic()
                            # Extra guards: no tool in flight, no recent stream_end activity
                            # ai_stopped_at tracks when AI last finished speaking — proxy for "session quiet"
                            quiet_long_enough = (now - ai_stopped_at[0]) >= 5.0
                            if (now - last_sent >= KEEPALIVE_INTERVAL
                                    and not ai_speaking
                                    and not tool_pending[0]
                                    and quiet_long_enough
                                    and to_gemini.qsize() == 0
                                    and stream_ended_once.is_set()):
                                try:
                                    await session.send_realtime_input(
                                        audio=types.Blob(
                                            data=KEEPALIVE_CHUNK,
                                            mime_type="audio/pcm;rate=16000"
                                        )
                                    )
                                    last_sent = now
                                except Exception:
                                    break

                    send_task      = asyncio.create_task(send_loop())
                    recv_task      = asyncio.create_task(recv_loop())
                    keepalive_task = asyncio.create_task(keepalive_loop())

                    done, pending = await asyncio.wait(
                        [send_task, recv_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    for t in pending:
                        t.cancel()
                    keepalive_task.cancel()

            except Exception as e:
                err = str(e)
                if ("1011" in err or "Deadline" in err) and not ended.is_set():
                    reconnect_count += 1
                    if reconnect_count > RECONNECT_LIMIT:
                        print("❌ Reconnect limit reached")
                        if not ended.is_set():
                            asyncio.create_task(finish_interview())
                        break
                    ai_speaking = False
                    print(f"🔁 1011 — reconnecting in 2s ({reconnect_count}/{RECONNECT_LIMIT})")
                    await asyncio.sleep(2)
                    continue
                else:
                    import traceback; traceback.print_exc()
                    print(f"❌ Live session error: {e}")
                    break
            break  # clean exit

    # reads from browser websocket, handles VAD and barge-in
    async def ws_receive_loop():
        nonlocal ai_speaking, last_user_activity
        silent_chunks = 0
        stream_ended  = False
        barge_chunks  = 0
        # candidate_speech_seen and candidate_loud_chunks are session-scope lists

        try:
            while not ended.is_set():
                try:
                    msg = await asyncio.wait_for(websocket.receive(), timeout=30.0)
                except asyncio.TimeoutError:
                    try:
                        await websocket.send_json({"ping": True})
                    except Exception:
                        break
                    continue

                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("text"):
                    continue

                data = msg.get("bytes") or b""
                if not data:
                    continue

                # Screen frame (JPEG ff d8)
                if data[:2] == b"\xff\xd8":
                    await to_gemini.put(("video", data))
                    continue

                rms = pcm_rms(data)

                # Always forward mic audio (barge-in needs uninterrupted stream)
                if to_gemini.full():
                    try: to_gemini.get_nowait()
                    except asyncio.QueueEmpty: pass
                await to_gemini.put(("audio", data))

                if rms > SILENCE_RMS:
                    last_user_activity = _time.monotonic()
                    silent_chunks = 0

                    if ai_speaking:
                        # Barge-in: sustained loud chunks while AI is speaking
                        if rms > BARGE_MIN_RMS:
                            barge_chunks += 1
                            if barge_chunks >= BARGE_THRESHOLD:
                                barge_chunks  = 0
                                ai_speaking   = False
                                ai_stopped_at[0] = _time.monotonic()
                                stream_ended  = False
                                candidate_speech_seen_l[0] = True
                                tool_state["candidate_spoken"] = True
                                tool_state["last_candidate_speech_time"] = _time.time()
                                print(f"⚡ Barge-in (rms={rms:.0f})")
                                await to_gemini.put(("stream_end", b""))
                        else:
                            barge_chunks = 0
                    else:
                        barge_chunks = 0
                        # Require 3 consecutive loud chunks before marking candidate as speaking.
                        # This filters out single clicks, room noise, reverb tail after AI speaks.
                        candidate_loud_chunks[0] += 1
                        if candidate_loud_chunks[0] >= 3:
                            candidate_speech_seen_l[0] = True
                            # Note: tool_state["candidate_spoken"] is NOT set here.
                            # It only becomes True when a real stream_end fires (below),
                            # ensuring save_interview_context cannot be called on noise alone.
                        if candidate_speech_seen_l[0] and stream_ended:
                            stream_ended = False
                            reset_screen_check()
                            print(f"🎤 Speech resumed (rms={rms:.0f})")
                else:
                    barge_chunks = 0
                    candidate_loud_chunks[0] = 0
                    silent_chunks += 1
                    # Fire stream_end only when:
                    # 1. AI greeting done
                    # 2. Candidate has actually spoken (sustained, not just noise)
                    # 3. Past cooldown window after AI stopped speaking
                    now_mono = _time.monotonic()
                    past_cooldown = (now_mono - ai_stopped_at[0]) >= AI_COOLDOWN
                    if (ai_first_turn_done.is_set()
                            and candidate_speech_seen_l[0]
                            and past_cooldown
                            and silent_chunks >= SILENCE_CHUNKS
                            and not stream_ended
                            and not ai_speaking):
                        stream_ended  = True
                        silent_chunks = 0
                        stream_ended_once.set()
                        tool_state["candidate_spoken"] = True
                        tool_state["last_candidate_speech_time"] = _time.time()
                        candidate_speech_seen_l[0] = True
                        await to_gemini.put(("stream_end", b""))

        except WebSocketDisconnect:
            print(f"🔌 Disconnected: {sid}")
        except Exception as e:
            print(f"❌ WS error: {e}")
        finally:
            ended.set()
            await to_gemini.put(None)

    # absolute time ceiling regardless of interview state
    async def hard_stop():
        await asyncio.sleep(HARD_STOP)
        if not ended.is_set():
            print("⏰ Hard stop")
            ended.set()
            await to_gemini.put(None)
            asyncio.create_task(finish_interview())

    # server-side timer, injects wrap-up text independent of AI tool calls
    async def timer_watchdog():
        from ai_interviewer.tools import _state as tool_state
        warned = False
        forced = False

        while not ended.is_set():
            await asyncio.sleep(10)
            if ended.is_set():
                break
            start = tool_state.get("start_time")
            if not start:
                continue
            elapsed   = _time.time() - start
            base_secs = INTERVIEW_MINUTES * 60
            remaining = base_secs - elapsed

            # 2-min warning: only inject if NOT in coding phase (model is talking)
            # During coding phase the model should stay silent — don't interrupt it
            if remaining <= 2 * 60 and not warned and not forced:
                warned = True
                if not tool_state.get("coding_phase_started"):
                    msg = "[SYSTEM] About 2 minutes remain. Begin wrapping up."
                    print("⏱️ 2-min warning")
                    if not ai_speaking:
                        await to_gemini.put(("text", msg.encode()))

            # Force-wrap: trigger report directly in Python — don't rely on model calling end_interview
            if elapsed >= (INTERVIEW_MINUTES + 2) * 60 and not forced:
                forced = True
                print(f"⏱️ Force-wrap at {INTERVIEW_MINUTES + 2}min — ending directly")
                ended.set()
                await to_gemini.put(None)
                asyncio.create_task(finish_interview())

    # nudge candidate if they go quiet — only during coding phase
    async def stuck_nudge():
        NUDGE_1     = 45
        NUDGE_2     = 90
        nudge_count = 0

        while not ended.is_set():
            await asyncio.sleep(5)
            if ended.is_set() or ai_speaking:
                continue

            # Only nudge during coding phase — concept phase silence is normal thinking time
            from ai_interviewer.tools import _state as tool_state
            if not tool_state.get("coding_phase_started"):
                nudge_count = 0
                continue

            # Don't nudge if we already have technical notes — candidate is in discussion phase
            if tool_state.get("technical_notes"):
                nudge_count = 0
                continue

            silence = _time.monotonic() - last_user_activity

            if silence >= NUDGE_2 and nudge_count < 2:
                nudge_count = 2
                msg = "[SYSTEM] Candidate has been silent for 90 seconds. Say only: 'Would you like a hint?' then wait."
                print("🤔 90s nudge")
                if not ai_speaking:
                    await to_gemini.put(("text", msg.encode()))

            elif silence >= NUDGE_1 and nudge_count < 1:
                nudge_count = 1
                msg = "[SYSTEM] Candidate has been silent for 45 seconds. Say only: 'Take your time.' then wait."
                print("🤔 45s nudge")
                if not ai_speaking:
                    await to_gemini.put(("text", msg.encode()))

            elif silence < NUDGE_1:
                nudge_count = 0

    # kick off all concurrent tasks
    live_task     = asyncio.create_task(run_live())
    ws_task       = asyncio.create_task(ws_receive_loop())
    stop_task     = asyncio.create_task(hard_stop())
    nudge_task    = asyncio.create_task(stuck_nudge())
    watchdog_task = asyncio.create_task(timer_watchdog())

    await asyncio.gather(live_task, ws_task, return_exceptions=True)
    stop_task.cancel()
    nudge_task.cancel()
    watchdog_task.cancel()
    # Wait for report to be sent before letting FastAPI close the WebSocket
    # Without this, the WS closes while build_report() Gemini API call is still running
    try:
        await asyncio.wait_for(report_sent.wait(), timeout=60)
    except asyncio.TimeoutError:
        print("⚠️ Report timed out")
    print(f"🧹 Cleaned up: {sid}")


@app.get("/")
async def root():
    return FileResponse(os.path.join(current_dir, "ai_interviewer/static/index.html"))

@app.get("/processor.js")
async def processor():
    return FileResponse(os.path.join(current_dir, "ai_interviewer/static/processor.js"))

if __name__ == "__main__":
    uvicorn.run(
        app, host="0.0.0.0", port=8000,
        ws_ping_interval=20,
        ws_ping_timeout=60,
    )