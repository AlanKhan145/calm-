# CALM — Adaptive Multimodal Wildfire Monitoring

**Hệ thống AI giám sát cháy rừng thích ứng đa phương thức**, sử dụng kiến trúc agentic (URSA) để đánh giá rủi ro, thu thập dữ liệu đa nguồn và trả lời câu hỏi dựa trên bằng chứng.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Tổng quan

CALM kết hợp **plan-driven execution**, **RSEN** (Reflexive Structured Experts Network) và **Evidence Evaluator** để:

- **Planning** → Phân rã query thành kế hoạch JSON có cấu trúc
- **Routing** → RouterAgent xác định task type (QA / Prediction)
- **Execution** → DataKnowledgeAgent, PredictionReasoningAgent, WildfireQAAgent chạy theo plan
- **Validation** → RSEN xác thực dự đoán bằng chuyên gia thời tiết + địa lý
- **Evaluation** → LLM-as-a-Judge chấm 5 tiêu chí chất lượng

### Điểm mạnh

| Tính năng | Mô tả |
|-----------|--------|
| **Plan-driven** | PlanningAgent → RouterAgent → ExecutionAgent; không còn pipeline cứng |
| **SeasFire integration** | Load seasfire-ml checkpoint; heuristic fallback từ met_data khi không có features |
| **Geocoding** | Chỉ cần địa chỉ văn bản (California, Amazon); tool tự tìm tọa độ |
| **Time default** | Không chỉ định ngày → mặc định hôm nay |
| **Step prompt** | Mỗi step có prompt riêng cho executor |
| **Dedup** | Tránh thu thập trùng theo location + time_range |
| **Safety check** | Mọi lệnh tool qua SafetyChecker trước khi gọi |

---

## Cài đặt nhanh

### 1. Clone và cài đặt

```bash
git clone https://github.com/viethungvu1998/calm.git
cd calm
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Biến môi trường

Tạo `.env` hoặc export:

```bash
# Bắt buộc: chọn một
OPENROUTER_API_KEY=sk-or-v1-...
# hoặc
OPENAI_API_KEY=sk-...

# Tùy chọn
GEE_PROJECT_ID=your-project
COPERNICUS_API_KEY=your-key
SEASFIRE_ML_PATH=/path/to/seasfire-ml    # Cho SeasFire model
SEASFIRE_DATASET_PATH=/path/to/data/train # Cho feature builder (optional)
```

### 3. Chạy demo

```bash
python main.py
# hoặc
calm run "What causes wildfires in the Amazon?"
calm run "Predict wildfire risk for California next 7 days"
```

---

## Luồng thực thi

```
Query
  │
  ▼
PlanningAgent (Generator → Reflector → Formalizer)
  │  → plan JSON: steps với action, agent, prompt, parameters
  ▼
RouterAgent (task_type: qa | prediction | hybrid)
  │
  ▼
ExecutionAgent (chạy từng step)
  │
  ├─ data_knowledge → DataKnowledgeAgent.retrieve()
  │    GEE, Copernicus CDS, Web, ArXiv; GeocodingTool cho địa chỉ
  │
  ├─ prediction → PredictionReasoningAgent.predict()
  │    SeasFireModelRunner (checkpoint) hoặc heuristic (met_data)
  │
  ├─ qa → WildfireQAAgent.invoke()
  │    Evidence Evaluator → ChromaDB retrieval → trả lời
  │
  └─ rsen → RSENModule.validate()
       Weather Analyst + Geo Analyst (song song) → Plausible/Implausible
```

---

## Cấu trúc dự án

```
calm/
├── configs/
│   └── example.yaml              # Cấu hình mẫu
├── src/calm/
│   ├── orchestrator.py           # CALMOrchestrator — điểm vào chính
│   ├── agents/
│   │   ├── planning_agent.py     # Phân rã query → plan JSON
│   │   ├── router_agent.py       # Task routing (LLM + keyword fallback)
│   │   ├── execution_agent.py    # Thực thi từng step
│   │   ├── data_knowledge_agent.py # GEE, CDS, Web, ArXiv
│   │   ├── prediction_reasoning_agent.py
│   │   ├── qa_agent.py           # WildfireQAAgent + Evidence Evaluator
│   │   ├── rsen_module.py        # Weather + Geo analysts
│   │   └── evaluator_agent.py    # LLM-as-a-Judge
│   ├── artifact/
│   │   ├── seasfire_runner.py    # Load SeasFire checkpoint, heuristic
│   │   ├── feature_builder.py    # (lat,lon,time) → tensor
│   │   ├── model_runner.py       # SeasFireModelRunner adapter
│   │   └── prediction_store.py   # Cache predictions
│   ├── tools/
│   │   ├── geocoding.py          # Địa chỉ → lat, lon
│   │   ├── earth_engine.py
│   │   ├── copernicus.py
│   │   ├── web_search.py
│   │   └── safety_check.py
│   ├── schemas/
│   │   └── contracts.py          # PlanStep, TaskRouting, ...
│   ├── utils/
│   │   └── time_utils.py         # resolve_time_range, default today
│   └── memory/
│       └── chroma_store.py
├── examples/
│   ├── 01_planning.py
│   ├── 02_prediction_rsen.py
│   ├── 03_wildfire_qa.py
│   └── 04_full_pipeline.py
├── calm_demo.ipynb               # Demo đầy đủ
└── main.py
```

---

## Ví dụ sử dụng

### Orchestrator — auto-routing

```python
from calm.utils.env_loader import load_env
load_env()

from calm.memory.chroma_store import ChromaMemoryStore
from calm.orchestrator import CALMOrchestrator
from langchain_openai import ChatOpenAI  # hoặc ChatOpenRouter

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
memory = ChromaMemoryStore(collection_name="calm", persist_directory=".chroma")
orc = CALMOrchestrator.from_llm(llm=llm, memory_store=memory)

# QA
result = orc.run("What causes wildfires in the Amazon?")
print(result["answer"], result["citations"])

# Prediction
result = orc.run("Predict wildfire risk for California next 7 days")
print(result["risk_level"], result["decision"], result["rationale"])
```

### Dùng SeasFire model

```bash
# 1. Clone seasfire-ml, train model
git clone https://github.com/SeasFire/seasfire-ml.git
cd seasfire-ml
pip install -r requirements.txt
./create_dataset.py --split=train --positive-samples-size=10000 --negative-samples-size=10000
./train_gru.py --target-week=4 --batch-size=16 --timesteps=6 --no-include-oci-variables

# 2. Cấu hình CALM
export SEASFIRE_ML_PATH=/path/to/seasfire-ml
# Trong configs/example.yaml: agent_config.prediction.checkpoint = runs/xxx.best_model.pt
```

```python
import yaml
with open("configs/example.yaml") as f:
    cfg = yaml.safe_load(f)
orc = CALMOrchestrator.from_llm(llm, memory, config=cfg)
result = orc.run("Predict wildfire risk for Amazon next 14 days")
```

Khi không có checkpoint hoặc features: hệ thống dùng **heuristic** từ met_data (temperature, humidity) từ GEE/CDS.

### Planning Agent

```python
from calm.agents.planning_agent import PlanningAgent

agent = PlanningAgent(llm=llm, config={}, n_max=3, f_max=3)
result = agent.invoke("Wildfire risk assessment for Amazon region next 7 days")
plan = result["final_output"]
for step in plan:
    print(step["step_id"], step["action"], step["agent"], step.get("prompt"))
```

### Evaluator — LLM-as-a-Judge

```python
from calm.agents.evaluator_agent import EvaluatorAgent

evaluator = EvaluatorAgent(llm=llm, config={"passing_score": 75.0})
eval_result = evaluator.evaluate(response=orc_result, query=query)
print(eval_result["average_score"], eval_result["passed"], eval_result["scores"])
```

---

## Cấu hình

| Thành phần | Config key | Mô tả |
|------------|------------|--------|
| Planning | `agent_config.planning` | n_max, f_max |
| Data | `agent_config.data_knowledge` | dedup_check, max_news_results |
| Prediction | `agent_config.prediction` | checkpoint, seasfire_variables, timesteps |
| RSEN | `agent_config.rsen` | memory_k, parallel_execution |
| QA | `agent_config.qa` | evidence_threshold, max_search_iterations |
| Evaluator | `agent_config.evaluator` | passing_score (default 75) |

---

## Technology Stack

| Hạng mục | Công nghệ |
|----------|-----------|
| Ngôn ngữ | Python ≥ 3.10 |
| LLM | LangChain, OpenAI, OpenRouter |
| Vector DB | Chroma |
| Embedding | OpenAI text-embedding-3-small / sentence-transformers all-MiniLM-L6-v2 |
| Tools | GEE, Copernicus CDS, DuckDuckGo, ArXiv |
| Model | SeasFire GRU (seasfire-ml), PyTorch |

---

## API Reference (tóm tắt)

| Thành phần | Mô tả |
|------------|--------|
| `CALMOrchestrator` | `run(query)` tự route; `from_llm(llm, memory_store, tools, model_runner, config)` |
| `PlanningAgent` | `invoke(query)` → plan JSON |
| `RouterAgent` | `route(query, plan_steps)` → TaskRouting |
| `ExecutionAgent` | `execute_step(step, context)` |
| `DataKnowledgeAgent` | `retrieve(query, params)` — collect, extract, ChromaDB |
| `PredictionReasoningAgent` | `predict(params)` — SeasFireModelRunner hoặc heuristic |
| `WildfireQAAgent` | `invoke(query, pre_retrieved)` |
| `RSENModule` | `validate(prediction, met_data, spatial_data)` |
| `EvaluatorAgent` | `evaluate(response, query)` → scores, passed |
| `SeasFireModelRunner` | `predict(params)` — adapter cho SeasFire |

---

## Phát triển

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

MIT. Chi tiết xem `LICENSE`.
