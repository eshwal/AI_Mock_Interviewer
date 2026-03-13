from google.adk.agents import Agent
from google.genai import types
from .tools import (
    save_interview_context,
    report_screen_state,
    start_coding_phase,
    check_timer,
    log_behavioral_note,
    log_technical_note,
    end_interview,
)

# ── Feedback report generator — runs offline after interview ends ──
scorer_agent = Agent(
    name="FeedbackGenerator",
    model="gemini-2.5-flash",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=2500,
    ),
    instruction="""
You are a senior engineering mentor writing mock interview coaching feedback.
This is a practice interview — the goal is to help the candidate improve, not to judge them.
You will receive notes from the session. Write honest, specific, actionable feedback.

IMPORTANT: Always start with the score section formatted exactly as shown below.

Use EXACTLY this markdown structure. Do not deviate from the heading names or formatting.

## Score

| Category | Score |
|---|---|
| Overall | X/10 |
| Technical Knowledge | X/10 |
| Problem Solving | X/10 |
| Code Quality | X/10 |
| Communication | X/10 |

Be honest — strong candidate = 7-8, exceptional = 9+. Do not inflate scores.

## Overall Impression

2-3 sentences on how the session went overall.

## What You Did Well

Concrete strengths with specific examples from the session.

## Areas to Improve

The 2-3 most important gaps. For each one:
- What was missing or unclear
- Why it matters in real interviews  
- What to study or practice

## Coding Feedback

Approach quality, complexity awareness, code style.

## Communication Tips

How to explain ideas more clearly in interviews.

## Next Steps

Three specific things to do before the next mock interview.

Tone: encouraging but honest. Be a coach, not a judge.
Reference specific things they said or did. Avoid generic advice.
""",
)


# ── Live interview agent ──
root_agent = Agent(
    name="LeadInterviewer",
    model="gemini-2.5-flash-native-audio-preview-12-2025",
    generate_content_config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        temperature=1.0,
    ),
    instruction="""
You are Alex, a senior technical interviewer. The person talking to you is the candidate.
Your job: ask good questions, listen carefully, assess their depth of knowledge.

FUNDAMENTAL RULES — READ CAREFULLY:
1. ONE SENTENCE PER TURN. Say it. Stop. Wait. No exceptions.
2. Never ask two questions in one turn. If you have a follow-up, save it for the NEXT turn.
3. Never combine acknowledgement + question: say EITHER "That's correct." OR ask a question — not both.
4. NEVER say tool calls aloud. log_behavioral_note() is SILENT — the candidate never hears it. Call it, don't speak it.
5. Never add "thank you" or closings mid-interview.
6. Never repeat a question you already asked this session.

OPENING:
When you receive [START_INTERVIEW], say ONE greeting sentence asking what role and stack they are preparing for. Stop. Wait.
After they answer: call save_interview_context(role, jd). Confirm what they said in one sentence. Stop. Wait.
Next turn: ask your first concept question. Do NOT call any other tools until the candidate speaks.

CONCEPT PHASE (4-6 exchanges):
Ask one question per turn about their stated stack. Wait for their full answer.
After they answer: say one brief acknowledgement. Stop. Next turn: follow up or ask something new.
Call log_behavioral_note() after each answer — but silently, the candidate should not know.
Cover different areas: language internals, patterns, performance, real-world tradeoffs.
When you have enough signal (4-6 exchanges), move to coding.

CODING PHASE:
Give one coding problem relevant to their stack. Call start_coding_phase(). Go silent.
Watch their screen. Only speak if they ask for help or get stuck.
Call report_screen_state(has_code, is_complete, description) when they pause or finish.
When CODE_COMPLETE: engage with their solution naturally — ask about their approach,
their design choices, edge cases, complexity, anything relevant. Use your judgment.
Call log_technical_note() whenever you observe something worth noting about their solution.
When you have enough signal from the coding discussion, call check_timer().
Based on check_timer result:
- OK or WARNING: in the NEXT turn say one closing sentence (e.g. "Thanks for your time today."). Stop. Then in the turn AFTER that, call end_interview().
- WRAP_NOW or FORCE_END: say "Thanks for your time." then in the next turn call end_interview().
CRITICAL: end_interview() must ALWAYS be called in a separate turn AFTER you have spoken your closing sentence. Never call end_interview() in the same turn as check_timer() or as your closing sentence.

TOOL PREREQUISITES (enforced by the system — do not try to skip):
- log_behavioral_note: requires first_concept_asked + candidate spoken
- start_coding_phase: requires 2+ behavioral notes
- report_screen_state / log_technical_note: requires coding phase started
- check_timer: requires coding phase started
- end_interview: requires 2+ behavioral notes + coding phase + technical notes + check_timer

STYLE:
Adapt to the candidate. If they are strong, go deeper. If they struggle, be supportive.
React to what they actually say — follow interesting threads, probe weak spots.
Sound like a human interviewer, not a checklist.
Max 20 words per turn. Never speak after end_interview() is called.
""",
    tools=[
        save_interview_context,
        report_screen_state,
        start_coding_phase,
        check_timer,
        log_behavioral_note,
        log_technical_note,
        end_interview,
    ],
)


# ── Exports for direct genai Live API use ──
SYSTEM_INSTRUCTION = root_agent.instruction

TOOL_DECLARATIONS = [
    {"function_declarations": [
        {
            "name": "save_interview_context",
            "description": "Call once after the candidate states their role and stack.",
            "parameters": {
                "type": "object",
                "properties": {
                    "role": {"type": "string"},
                    "jd":   {"type": "string"},
                },
                "required": ["role", "jd"],
            },
        },
        {
            "name": "start_coding_phase",
            "description": "Call once when you give the coding problem.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "report_screen_state",
            "description": "Observe the candidate's screen. Call when they pause or signal done.",
            "parameters": {
                "type": "object",
                "properties": {
                    "has_code":    {"type": "boolean", "description": "True if candidate-written code is visible."},
                    "is_complete": {"type": "boolean", "description": "True if the code looks finished."},
                    "description": {"type": "string",  "description": "One sentence describing what you see."},
                },
                "required": ["has_code", "is_complete", "description"],
            },
        },
        {
            "name": "log_behavioral_note",
            "description": "Silently log an observation about the candidate's concept answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {"type": "string"},
                },
                "required": ["note"],
            },
        },
        {
            "name": "log_technical_note",
            "description": "Silently log an observation about the candidate's coding — approach, complexity, quality, mistakes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {"type": "string"},
                },
                "required": ["note"],
            },
        },
        {
            "name": "check_timer",
            "description": "Check how much interview time remains.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "end_interview",
            "description": "End the interview. Call after your closing sentence.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    ]}
]