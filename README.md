# Autonomous AI Sales Voice Agent
A prototype real-time voice sales agent: the caller speaks, the agent transcribes,
thinks with Gemini (+ tools), and talks back with ElevenLabs over one WebSocket.

```
client audio ──▶ /ws/call/{session_id} ──▶ Deepgram streaming STT
                                                   │ final transcript
                                                   ▼
                              LangChain + Gemini agent (tool calling)
                     knowledge_search · save_lead · check_calendar_availability
                     book_appointment · log_turn_metadata
                                                   │ reply text
                                                   ▼
                                        ElevenLabs streaming TTS
                                                   │ audio bytes
                                                   ▼
                                             back to the client
```

On call end the transcript is summarised by Gemini into
`{lead_intent, objections, outcome, next_action}`, attached to the lead row in
SQLite, and returned to the client as the final JSON message.

## Stack
| Layer | Choice |
| --- | --- |
| API / orchestration | FastAPI (Python 3.11), one WebSocket per call |
| STT | Deepgram Python SDK v7 (`listen.v1.connect`), Nova-3, interim results |
| LLM | Google Gemini via `langchain-google-genai`, tool calling |
| TTS | ElevenLabs streaming HTTP API (`pcm_16000`) |
| RAG | ChromaDB (persistent, local) + LangChain text splitter |
| Embeddings | Chroma's built-in ONNX `all-MiniLM-L6-v2` (free, local, no torch) |
| CRM | SQLite via SQLAlchemy (`leads` table) |
| Calendar | Google Calendar API, service account, mock fallback |

## Project structure
```
sales_voice_agent/
├── app/
│   ├── main.py                      # FastAPI app + /ws/call/{session_id}
│   ├── config.py                    # .env settings
│   ├── stt/deepgram_client.py       # streaming STT wrapper
│   ├── tts/elevenlabs_client.py     # streaming TTS wrapper
│   ├── agent/
│   │   ├── gemini_agent.py          # Gemini + tool loop
│   │   ├── tools.py                 # the five LangChain tools
│   │   ├── prompts.py               # system + summary prompts
│   │   └── memory.py                # per-call memory (prototype: dict)
│   ├── rag/
│   │   ├── ingest.py                # build the Chroma index
│   │   └── retriever.py             # knowledge_search backend
│   ├── crm/
│   │   ├── models.py                # Lead model + engine
│   │   └── crud.py                  # upsert / read / attach summary
│   ├── calendar_integration/google_calendar.py
│   └── summary/call_summary.py      # end-of-call structured summary
├── data/knowledge_base/             # pricing.txt, features.txt, faq.txt, integrations.txt
├── scripts/test_client.py           # text / wav / microphone test client
├── chroma_db/                       # persisted vector store (gitignored)
├── sales_agent.db                   # sqlite CRM (gitignored)
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

### 1. Install
```bash
python -m venv .venv
# Windows (Git Bash / PowerShell):
.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```
`sounddevice` is only needed for the microphone test client; everything else
works without it.

### 2. Configure

```bash
cp .env.example .env    # then edit
```

| Variable | Purpose |
| --- | --- |
| `DEEPGRAM_API_KEY` | streaming STT |
| `GEMINI_API_KEY` | conversational brain + summaries (required for real replies) |
| `ELEVENLABS_API_KEY` | streaming TTS |
| `ELEVENLABS_VOICE_ID` | voice to use (default is the premade "Sarah") |
| `GOOGLE_CALENDAR_CREDENTIALS_PATH` | service-account JSON path |
| `GOOGLE_CALENDAR_ID` | calendar to read/book (e.g. `primary` or an email) |

Optional: `GEMINI_MODEL`, `DEEPGRAM_MODEL`, `AUDIO_SAMPLE_RATE`,
`ELEVENLABS_OUTPUT_FORMAT`, `GOOGLE_CALENDAR_TIMEZONE`,
`GOOGLE_CALENDAR_DELEGATED_USER`, `COMPANY_NAME`, `COMPANY_PRODUCT`,
`DATABASE_URL`, `CHROMA_DIR`.

Notes:
- Values already present in the process environment win over `.env`
  (`load_dotenv(override=False)`), which is handy in CI.
- ElevenLabs free plans reject *library* voices with HTTP 402; the default
  voice id is a premade voice that works on free plans.

### 3. Build the knowledge base
```bash
python -m app.rag.ingest
```
This reads `data/knowledge_base/*.txt|*.md|*.pdf`, splits with
`RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)`, deletes the
existing Chroma collection and re-indexes. First run downloads the ~80 MB ONNX
embedding model to `~/.cache/chroma`.

### 4. Run the server
```bash
.venv/Scripts/python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- `GET /health` — configuration status
- `GET /leads` — read-only mock CRM view (useful in the demo)
- `WS /ws/call/{session_id}` — the voice call

## WebSocket protocol

Client → server:
- **binary frames**: raw audio (16-bit PCM, 16 kHz, mono by default)
- `{"type": "text", "text": "..."}` — typed turn (handy without a microphone)
- `{"type": "end_call"}` — finish, get the summary
- `{"type": "ping"}` → `{"type": "pong"}`

Server → client:
- `{"type": "ready", ...}` including `stt_connected` and `tts_format`
- `{"type": "partial_transcript", "text": ...}` / `{"type": "final_transcript", "text": ...}`
- `{"type": "thinking"}`
- `{"type": "assistant_text", "text": ...}`
- `{"type": "audio_start", "format": "pcm_16000"}`, then **binary audio frames**, then `{"type": "audio_end"}`
- `{"type": "call_summary", "summary": {...}, "lead_id": 2 | null}`
- `{"type": "warning" | "error", "message": ...}` for degraded paths

## Test it
```bash
# 1) No microphone needed - type turns, get spoken replies:
python scripts/test_client.py --mode text

# 2) Stream a pre-recorded WAV (must be 16 kHz mono 16-bit):
ffmpeg -i input.m4a -ar 16000 -ac 1 -sample_fmt s16 sample.wav
python scripts/test_client.py --mode wav --wav sample.wav

# 3) Live microphone (needs sounddevice + an input device):
python scripts/test_client.py --mode mic

# Add --no-audio if you only want the transcripts.
```

The client prints partial/final transcripts, the agent's reply, plays the
streamed audio, and prints the structured call summary when the call ends.

Try: *"Hi, I'm Rahul, we run four stores and I want to know your pricing."*
then *"That sounds reasonable, can we do a demo next week?"*

## Google Calendar (service account)

1. In Google Cloud Console: create a project, enable **Google Calendar API**.
2. Create a **service account**, then create a **JSON key** for it.
3. Save the key as `credentials/google-service-account.json` (gitignored).
4. Copy the service account's email (`...@....iam.gserviceaccount.com`).
5. In Google Calendar → the target calendar → *Settings and sharing* → *Share
   with specific people*, add that email with **Make changes to events**.
6. Set `GOOGLE_CALENDAR_ID` to that calendar's address (`primary` works only
   for the service account's own empty calendar).
7. Workspace only: to act as a real user instead of sharing a calendar, enable
   domain-wide delegation and set `GOOGLE_CALENDAR_DELEGATED_USER`.

If the credentials file is missing, availability returns clearly-labelled mock
slots and `book_appointment` returns a `[MOCK BOOKING]` confirmation, so the
demo never breaks. If the API call itself fails, the tool returns a graceful
"sorry, I'm having trouble..." message instead.

## RAG tradeoff
Embeddings use Chroma's default ONNX MiniLM model: free, local, no API cost and
no torch dependency, at some accuracy cost on niche vocabulary. To trade cost
for quality, replace `LocalMiniLMEmbeddings` in `app/rag/retriever.py` with
`GoogleGenerativeAIEmbeddings`, then re-run `python -m app.rag.ingest`.

## Error handling / degradation
| Failure | Behaviour |
| --- | --- |
| Deepgram unavailable | `ready` reports `stt_connected: false`; the client can still send `{"type": "text"}` turns |
| Gemini unavailable / no key | Spoken fallback ("Sorry, I'm having trouble thinking right now...") |
| A tool raises | Tool returns a speakable error; the turn continues |
| ElevenLabs fails | Reply text is already delivered; a `warning` is sent, WebSocket stays open |
| Google Calendar absent/fails | Mock availability/booking or a graceful message |
| Summariser fails | Fallback summary dict with an `error` field; the call still closes cleanly |

Every external call is wrapped in `try/except` with a logged traceback.

## Prototype limits (deliberately simple)
- **Memory is in-process** (`app/agent/memory.py`): `session_id -> CallSession`
  dict. Production would use Redis with a TTL, so multiple API workers share
  state and a restart does not drop calls.
- **Turn taking** is final-transcript driven (Deepgram endpointing), with one
  agent turn at a time per session. Production voice agents add barge-in /
  VAD-driven interruption and cancel in-flight TTS.
- **No authentication** on the WebSocket or `/leads`; add auth before exposing
  this publicly.
- **Service-account calendars** can be awkward; the README above covers the
  share-with-service-account workaround.
- **Audio format** is fixed at PCM 16 kHz mono for the demo. For a browser
  client, set `ELEVENLABS_OUTPUT_FORMAT=mp3_44100_128` and decode/play with
  WebAudio, and resample the microphone input before sending.
- **`/leads` and the summary** are the seed of the "business insights" view;
  they are read-only in this prototype.
