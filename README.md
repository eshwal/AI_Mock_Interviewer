# AI Mock Interviewer

A real-time AI-powered technical mock interview system built with Google Gemini Live. The AI conducts a full technical interview — asking concept questions, giving coding problems, watching your screen as you code, and delivering a coaching report the moment the interview ends.

## What It Does

- **Voice-to-voice interview** — speak naturally, Alex (the AI interviewer) listens and responds in real time
- **Screen analysis** — shares your screen so Alex can see your code as you write it, track progress, and ask targeted follow-up questions
- **Structured interview flow** — concept questions → coding problem → complexity discussion → coaching report
- **Instant feedback** — scored report with specific strengths, areas to improve, and next steps delivered immediately after

## Tech Stack

- **Google Gemini Live** (`gemini-2.5-flash-native-audio`) — real-time bidirectional audio + vision
- **Google Gemini 2.5 Flash** — report generation
- **FastAPI + WebSockets** — Python backend
- **AudioWorklet API** — 16kHz mic capture, 24kHz playback
- **Google Cloud Run** — deployment

## Architecture

See `architecture.mermaid` for the full system diagram.

Key components:
- Custom VAD (Voice Activity Detection) pipeline with barge-in support
- Tool-based state machine enforcing correct interview structure
- 5 concurrent async tasks per session managing audio, video, tools, nudges, and timers
- Automatic transcript cleaning and duplicate audio suppression

## Spin-Up Instructions

### Prerequisites
- Python 3.11+
- Google Gemini API key ([get one here](https://aistudio.google.com))

### Local Development

```bash
# Clone the repo
git clone https://github.com/YOURUSERNAME/ai-mock-interviewer.git
cd ai-mock-interviewer

# Install dependencies
pip install -r requirements.txt
```

Set your API key:
```bash
# Mac/Linux
export GOOGLE_API_KEY=your_api_key_here

# Windows CMD
set GOOGLE_API_KEY=your_api_key_here

# Windows PowerShell
$env:GOOGLE_API_KEY="your_api_key_here"
```

Run:
```bash
python main.py
```

Open `http://localhost:8000` in Chrome. Allow microphone and screen sharing when prompted.

### Demo Mode (Short 5-min Interview)

```bash
# Mac/Linux
DEMO_MODE=1 python main.py

# Windows CMD
set DEMO_MODE=1 && python main.py

# Windows PowerShell
$env:DEMO_MODE="1"; python main.py
```

### Docker

```bash
docker build -t ai-mock-interviewer .
docker run -p 8000:8080 -e GOOGLE_API_KEY=your_key ai-mock-interviewer
```

### Deploy to Google Cloud Run

```bash
# Mac/Linux
export GOOGLE_API_KEY=your_api_key_here
gcloud run deploy ai-mock-interviewer \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_API_KEY=$GOOGLE_API_KEY \
  --memory 1Gi \
  --timeout 3600

# Windows CMD
set GOOGLE_API_KEY=your_api_key_here
gcloud run deploy ai-mock-interviewer --source . --region us-central1 --allow-unauthenticated --set-env-vars GOOGLE_API_KEY=%GOOGLE_API_KEY% --memory 1Gi --timeout 3600

# Windows PowerShell
$env:GOOGLE_API_KEY="your_api_key_here"
gcloud run deploy ai-mock-interviewer --source . --region us-central1 --allow-unauthenticated --set-env-vars GOOGLE_API_KEY=$env:GOOGLE_API_KEY --memory 1Gi --timeout 3600
```

## Project Structure

```
├── main.py                          # FastAPI server, WebSocket endpoint, VAD logic
├── ai_interviewer/
│   ├── agent.py                     # Gemini Live config, system prompt, tool declarations
│   ├── tools.py                     # Tool implementations and state machine
│   └── static/
│       ├── index.html               # Frontend UI
│       └── processor.js             # AudioWorklet PCM processor
├── Dockerfile
├── requirements.txt
└── architecture.mermaid
```

## How It Works

1. Browser captures mic audio (16kHz PCM) and screen frames (JPEG, every 5s)
2. Audio streams to FastAPI via WebSocket
3. FastAPI forwards to Gemini Live via bidirectional WebSocket
4. Custom VAD detects when candidate finishes speaking and fires `stream_end`
5. Gemini responds with audio + calls tools to log notes and manage interview state
6. Tool state machine enforces correct sequence: context → behavioral notes → coding phase → timer check → end
7. On interview end, Gemini 2.5 Flash generates a scored coaching report
8. Report rendered as markdown in the browser

## Requirements

```
fastapi
uvicorn
google-adk
```
