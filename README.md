# CALM — Adaptive Multimodal Wildfire Monitoring

Architecture base: **URSA** (Universal Research and Scientific Agent, LANL)  
Source paper: "Adaptive Multimodal Wildfire Monitoring with Autonomous Agentic AI"  
Reference: [GitHub](https://github.com/viethungvu1998/calm)

## Features

- **LangGraph StateGraph** — 3-node structure (generator → reflector → formalizer)
- **RSEN** — Reflexive Structured Experts Network (parallel weather + geo analysts)
- **ChromaDB memory** — Reflexion framework, separate collections per agent
- **Safety checks** — Before every tool call (GEE, CDS, DuckDuckGo, ArXiv)
- **Evidence Evaluator** — Anti-hallucination gate in QA pipeline

## Install

```bash
cd calm
pip install -e ".[dev]"
```

Set `OPENAI_API_KEY` (or configure Ollama). Optional: `GEE_PROJECT_ID`, `COPERNICUS_API_KEY`.

## Usage

```bash
calm plan "Wildfire risk assessment for Amazon region next 7 days"
calm version
```

```python
from calm.agents.planning_agent import PlanningAgent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
agent = PlanningAgent(llm=llm, config={})
result = agent.invoke("Wildfire risk California 14 days")
```

## Tests

```bash
pytest tests/ -v
```

## Structure

```
calm/
├── configs/example.yaml
├── src/calm/
│   ├── agents/       # PlanningAgent, RSENModule, WildfireQAAgent, etc.
│   ├── tools/        # earth_engine, copernicus, web_search, arxiv, safety_check
│   ├── prompt_library/  # Tables A.1–A.8 from paper
│   ├── memory/       # ChromaMemoryStore
│   └── models/       # LSTM, ConvLSTM, UTAE, FireCastNet stubs
├── examples/
└── tests/
```
