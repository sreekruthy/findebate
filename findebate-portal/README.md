# FinDebate Portal

Full-stack research demo for the FinDebate multi-agent financial analysis pipeline.

## Run

```bash
bash start.sh
```

Backend: `http://localhost:8000`  
Frontend: `http://localhost:5173`

## Modes

- `Precomputed`: streams real P5 JSON outputs from `backend/data/p5_outputs`. This works immediately.
- `Live`: runs RAG, P3 agents, P4 agents, P5 debate, and P6 judge. It is enabled only when the local ChromaDB/agent folders and API keys are available.

## API Keys

Add keys to `backend/.env`:

```bash
GROQ_API_KEY=...
GEMINI_API_KEY=...
NVIDIA_API_KEY=...
```

The app deliberately keeps precomputed mode independent from live pipeline dependencies.
