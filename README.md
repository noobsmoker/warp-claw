# 🤖 Warp-Claw

Hybrid infrastructure combining Warp-Cortex with OpenAI-compatible API for local Apple Silicon deployment.

## Features

- **Multi-Agent Councils**: Research, Code, Creative, and Meta councils that collaborate in real-time
- **OpenAI-Compatible API**: Use the `openai` Python client to interact with local models
- **M1/MPS Optimization**: Running on Apple Silicon with Metal Performance Shaders
- **Tool Integration**: Code execution, web search, file system, and knowledge graph tools
- **MCP Bridge**: Connect external Model Context Protocol servers
- **WebSocket Streaming**: Real-time agent activity streaming
- **Optional Dashboard**: Streamlit UI for monitoring and interaction

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Client (openai-python)                 │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              OpenAI API Server (:8000)                 │
│  ┌──────────────────────────────────────────────┐    │
│  │  POST /v1/chat/completions                   │    │
│  │  POST /v1/completions                       │    │
│  │  GET  /v1/models                          │    │
│  │  POST /v1/agents/spawn                    │    │
│  └──────────────────┬───────────────────────┘    │
└──────────────────────┼─────────────────────────────── ────┘
                       │
         ┌─────────────┼─────────────┐
         ▼           ▼           ▼
┌────────────┐ ┌───────────┐ ┌──────────────┐
│  Council  │ │  Tools    │ │  WebSocket │
│Orchestrator│ │ Executor │ │  Stream    │
└─────┬────┘ └────┬─────┘ └─────────────┘
      │            │
      ▼            ▼
┌─────────────────────────┐
│   M1 Cortex Bridge      │
│   (MPS/CPU)            │
└─────────────────────────┘
```

## Quick Start

### 1. Install

```bash
# Clone and install
git clone https://github.com/yourusername/warp-claw.git
cd warp-claw
pip install -r requirements.txt

# Or use M1 setup script
bash scripts/setup_m1.sh
```

### 2. Download a Model

```bash
python scripts/download_models.py qwen-0.5b
```

### 3. Run the API Server

```bash
python -m src.interfaces.openai_api
# Or: make run
```

Server starts on `http://localhost:8000`

### 4. Use with OpenAI Client

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Standard chat
response = client.chat.completions.create(
    model="qwen-0.5b",
    messages=[{"role": "user", "content": "Explain quantum computing [VERIFY]"}]
)
print(response.choices[0].message.content)

# With tools
response = client.chat.completions.create(
    model="qwen-0.5b",
    messages=[{"role": "user", "content": "Calculate fibonacci(100)"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Run Python code"
        }
    }]
)
```

### 5. Spawn a Council

```python
import requests

r = requests.post("http://localhost:8000/v1/agents/spawn", json={
    "prompt": "Design a distributed system",
    "council_types": ["research", "creative", "code"],
    "agent_count": 10
})
council_id = r.json()["council_id"]

# Check status
r = requests.get(f"http://localhost:8000/v1/agents/status/{council_id}")
print(r.json())
```

### 6. Start Dashboard (Optional)

```bash
make dashboard
# Or: streamlit run src/dashboard/app.py
```

Dashboard at `http://localhost:8501`

## Configuration

### Models (`config/models.yaml`)

```yaml
models:
  qwen-0.5b:
    repo: "Qwen/Qwen2.5-0.5B-Instruct"
    device: "mps"
    max_agents: 100
    
  qwen-1.5b:
    repo: "Qwen/Qwen2.5-1.5B-Instruct"
    device: "mps"
    max_agents: 50

default_model: "qwen-0.5b"
```

### Agent Councils (`config/agents.yaml`)

```yaml
councils:
  research:
    agent_count: 3
    system_prompt: "You are a fact-checking sub-agent..."
    triggers: ["[SEARCH]", "[VERIFY]"]
    
  code:
    agent_count: 2
    system_prompt: "You are a code review sub-agent..."
    triggers: ["[CODE]", "[REVIEW]"]
```

## API Endpoints

| Endpoint | Method | Description |
|----------|-------|-------------|
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Chat completion |
| `/v1/completions` | POST | Text completion |
| `/v1/agents/spawn` | POST | Spawn council |
| `/v1/agents/status/{id}` | GET | Council status |
| `/v1/agents` | GET | List agents |

## Tool Reference

| Tool | Description |
|------|-------------|
| `execute_python` | Run Python code in sandbox |
| `web_search` | Search the web |
| `web_fetch` | Fetch URL content |
| `file_system` | Read/write files |
| `knowledge_graph` | RAG memory store |

## Docker

```bash
# Build
make docker-build

# Run
make docker-run
```

## Requirements

- Python 3.11+
- Apple Silicon (M1/M2/M3) for MPS support
- Or x86_64 with CPU fallback

## License

MIT