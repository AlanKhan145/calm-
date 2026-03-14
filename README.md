# CALM — Adaptive Multimodal Wildfire Monitoring

**Giám sát cháy rừng thích ứng đa phương thức** — Hệ thống AI tự trị sử dụng kiến trúc agentic để đánh giá rủi ro cháy rừng, thu thập dữ liệu đa nguồn và trả lời câu hỏi dựa trên bằng chứng.

> **Kiến trúc gốc:** URSA (Universal Research and Scientific Agent, LANL)  
> **Bài báo tham chiếu:** "Adaptive Multimodal Wildfire Monitoring with Autonomous Agentic AI"  
> **Nguồn:** [GitHub](https://github.com/viethungvu1998/calm)

---

## Mục lục

- [Tổng quan](#tổng-quan)
- [Tính năng chính](#tính-năng-chính)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Cấu hình](#cấu-hình)
- [Sử dụng](#sử-dụng)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [API Reference](#api-reference)
- [Tests](#tests)
- [Docker](#docker)
- [Đóng góp](#đóng-góp)

---

## Tổng quan

CALM là một hệ thống AI đa tác nhân (multi-agent) được thiết kế để:

1. **Lên kế hoạch** — Phân rã câu hỏi về cháy rừng thành các bước thực thi có cấu trúc JSON
2. **Thu thập dữ liệu** — Tổng hợp dữ liệu từ vệ tinh (GEE), khí tượng (Copernicus CDS), tin tức (DuckDuckGo), tài liệu khoa học (ArXiv)
3. **Dự đoán** — Sử dụng mô hình FireCastNet, LSTM, ConvLSTM, UTAE để dự báo rủi ro
4. **Xác thực** — RSEN (Reflexive Structured Experts Network) với chuyên gia thời tiết và địa lý song song
5. **Hỏi đáp** — Trả lời câu hỏi với Evidence Evaluator chống hallucination
6. **Đánh giá** — LLM-as-a-Judge với 5 tiêu chí chất lượng

---

## Tính năng chính

### 1. LangGraph StateGraph — Kiến trúc 3 node (URSA)

Mọi agent tuân theo luồng **generator → reflector → formalizer**:

- **Generator:** Tạo output ban đầu từ query
- **Reflector:** Phản ánh, chỉnh sửa, quyết định có chấp nhận (`[APPROVED]`) hay lặp lại
- **Formalizer:** Chuyển đổi sang JSON hợp lệ với tối đa `f_max` lần thử

### 2. RSEN — Reflexive Structured Experts Network

Mạng chuyên gia song song, **cách ly** (mỗi chuyên gia độc lập):

- **Weather Analyst:** Phân tích dữ liệu khí tượng (nhiệt độ, độ ẩm, gió)
- **Geo Analyst:** Phân tích dữ liệu không gian (loại nhiên liệu, độ dốc, độ cao)
- **Ops Coordinator:** Tổng hợp báo cáo từ cả hai, đưa ra quyết định cuối cùng (Plausible/Implausible)

### 3. ChromaDB Memory — Reflexion Framework

- Bộ nhớ vector riêng biệt cho từng agent: `calm_rsen_memory`, `calm_qa_memory`, `calm_plan_memory`
- Embedding mặc định: `text-embedding-3-small` (OpenAI) hoặc HuggingFace fallback
- Top-k retrieval với ngưỡng tương đồng (`similarity_threshold`)

### 4. Safety Checks — Kiểm tra trước mọi lệnh

- **URSA pattern:** Kiểm tra an toàn trước **mỗi** lệnh gọi tool
- **CALM extension:** Các tiêu chí đặc thù cháy rừng:
  - Không gửi cảnh báo khẩn cấp mà không xác minh
  - Không xóa/ghi đè dữ liệu vệ tinh hoặc checkpoint
  - Không truy cập API mà không có credentials hợp lệ
  - Không thực thi shell thay đổi trạng thái hệ thống

### 5. Evidence Evaluator — Cổng chống hallucination

- Trong QA pipeline: đánh giá bằng chứng trước khi trả lời
- Nếu bằng chứng không đủ: kích hoạt tìm kiếm trực tuyến
- Ngưỡng: `evidence_threshold` (mặc định 0.65)

### 6. Evaluator Agent — LLM-as-a-Judge

Đánh giá chất lượng output theo 5 tiêu chí (0–100):

| Tiêu chí | Mô tả |
|----------|-------|
| Data-Accuracy | Độ chính xác dữ liệu, trích dẫn phù hợp |
| Explainability | Giải thích rõ ràng, lập luận nhân quả |
| Jargon-Avoidance | Tránh thuật ngữ khó hiểu, dễ tiếp cận |
| Redundancy-Avoidance | Không lặp lại, câu trả lời súc tích |
| Citation-Quality | Chất lượng nguồn trích dẫn |

**Ngưỡng đạt:** `passing_score >= 75` (mặc định)

---

## Yêu cầu hệ thống

- **Python:** >= 3.10
- **LLM:** OpenAI API (gpt-4o) hoặc Ollama (local)
- **Tuỳ chọn:**
  - Google Earth Engine: `GEE_PROJECT_ID`
  - Copernicus CDS: `COPERNICUS_API_KEY`
  - GPU: cho mô hình dự đoán (FireCastNet, LSTM, v.v.)

---

## Cài đặt

### 1. Clone và cài đặt package

```bash
cd calm
pip install -e ".[dev]"
```

**Optional dependencies:**

- `.[dev]` — pytest, pytest-cov, ruff, mypy
- `.[otel]` — OpenTelemetry cho tracing

### 2. Biến môi trường

```bash
# Bắt buộc (hoặc dùng Ollama)
export OPENAI_API_KEY="sk-..."

# Tuỳ chọn
export GEE_PROJECT_ID="your-gee-project"
export COPERNICUS_API_KEY="your-cds-api-key"
```

### 3. Xác thực Earth Engine (nếu dùng GEE)

```bash
earthengine authenticate
```

### 4. Cấu hình Copernicus CDS (nếu dùng ERA5)

Tạo file `~/.cdsapirc` với nội dung:

```
url: https://cds.climate.copernicus.eu/api/v2
key: <your-uid>:<your-api-key>
```

---

## Cấu hình

File cấu hình mẫu: `configs/example.yaml`

```yaml
llm_config:
  model: openai:gpt-4o
  temperature: 0.0

agent_config:
  planning:
    n_max: 3        # Số lần tối đa reflector có thể lặp
    f_max: 3        # Số lần tối đa formalizer thử JSON

  data_knowledge:
    gee_project: ${GEE_PROJECT_ID}
    cds_api_key: ${COPERNICUS_API_KEY}
    cache_dir: .cache/wildfire_data
    dedup_check: true      # FR-D05: kiểm tra trùng lặp trước khi crawl
    max_news_results: 10
    max_arxiv_papers: 3

  prediction:
    model: firecastnet
    checkpoint: checkpoints/firecastnet_seasfire.pt
    device: cuda

  rsen:
    memory_collection: calm_rsen_memory
    memory_k: 3
    parallel_execution: true

  qa:
    n_max: 3
    memory_collection: calm_qa_memory
    evidence_threshold: 0.65
    max_search_iterations: 3

  evaluator:
    passing_score: 75.0

  memory:
    type: chroma
    persist_directory: .chroma
    embedding_model: text-embedding-3-small
```

---

## Sử dụng

### CLI

```bash
# Chạy Planning Agent
calm plan "Wildfire risk assessment for Amazon region next 7 days"

# Hiển thị phiên bản
calm version
```

### Python API

#### 1. Planning Agent — Phân rã query thành kế hoạch JSON

```python
from calm.agents.planning_agent import PlanningAgent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
agent = PlanningAgent(llm=llm, config={}, n_max=3, f_max=3)
result = agent.invoke("Wildfire risk California 14 days")

plan_steps = result["final_output"]
print(plan_steps)
```

#### 2. RSEN — Xác thực dự đoán với chuyên gia song song

```python
from calm.agents.rsen_module import RSENModule
from calm.memory.chroma_store import ChromaMemoryStore
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
memory = ChromaMemoryStore(
    collection_name="calm_rsen_memory",
    persist_directory=".chroma",
    k=3,
)
rsen = RSENModule(llm=llm, memory_store=memory, k=3)

result = rsen.validate(
    prediction={"risk_level": "High", "confidence": 0.8},
    met_data={"temperature": 35.0, "humidity": 0.2, "wind_speed": 15},
    spatial_data={"fuel_type": "Shrubland", "slope": 25, "elevation": 500},
)
print(result["validation_decision"])
print(result["final_rationale"])
```

#### 3. Wildfire QA Agent — Hỏi đáp với Evidence Evaluator

```python
from calm.agents.data_knowledge_agent import DataKnowledgeAgent
from calm.agents.qa_agent import WildfireQAAgent
from calm.memory.chroma_store import ChromaMemoryStore
from calm.tools.safety_check import SafetyChecker
from calm.tools.web_search import WebSearchTool
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
safety = SafetyChecker(llm=llm)
web = WebSearchTool(safety_checker=safety, config={"max_news_results": 5})
memory = ChromaMemoryStore(collection_name="calm_qa_memory", persist_directory=".chroma", k=3)

tools = {"earth_engine": None, "copernicus": None, "web_search": web, "arxiv": None}
data_agent = DataKnowledgeAgent(llm=llm, tools=tools, memory_store=memory, config={"dedup_check": True})

qa = WildfireQAAgent(
    llm=llm,
    data_agent=data_agent,
    web_search_tool=web,
    memory_store=memory,
    config={"evidence_threshold": 0.65},
)
result = qa.invoke("What caused the 2023 Canadian wildfires?")
print(result["final_output"])
```

#### 4. Full Pipeline — Kế hoạch + Đánh giá

```python
from calm.agents.evaluator_agent import EvaluatorAgent
from calm.agents.planning_agent import PlanningAgent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
planner = PlanningAgent(llm=llm, config={})
evaluator = EvaluatorAgent(llm=llm, config={"passing_score": 75.0})

query = "Assess wildfire risk for California Central Valley next 14 days"
plan_result = planner.invoke(query)
plan = plan_result.get("final_output") or []

eval_result = evaluator.evaluate(response={"plan": plan, "query": query}, query=query)
print("Evaluation passed:", eval_result["passed"])
print("Average score:", eval_result["average_score"])
```

### Chạy examples

```bash
cd calm
python examples/01_planning.py
python examples/02_prediction_rsen.py
python examples/03_wildfire_qa.py
python examples/04_full_pipeline.py
```

---

## Cấu trúc dự án

```
calm/
├── configs/
│   └── example.yaml           # Cấu hình mẫu
├── src/calm/
│   ├── __init__.py
│   ├── agents/                # Các agent
│   │   ├── base_agent.py      # BaseCALMAgent, AgentState, URSA 3-node
│   │   ├── planning_agent.py  # PlanningAgent
│   │   ├── data_knowledge_agent.py  # Thu thập GEE, CDS, Web, ArXiv
│   │   ├── execution_agent.py # Thực thi kế hoạch
│   │   ├── rsen_module.py     # RSEN (weather + geo analysts)
│   │   ├── prediction_reasoning_agent.py
│   │   ├── qa_agent.py        # WildfireQAAgent + Evidence Evaluator
│   │   └── evaluator_agent.py # LLM-as-a-Judge
│   ├── tools/                 # Công cụ với safety check
│   │   ├── earth_engine.py
│   │   ├── copernicus.py
│   │   ├── web_search.py
│   │   ├── arxiv_tool.py
│   │   ├── safety_check.py
│   │   └── wildfire_models.py
│   ├── prompt_library/        # Prompts (Tables A.1–A.8)
│   │   ├── planning_prompts.py
│   │   ├── rsen_prompts.py
│   │   ├── qa_prompts.py
│   │   └── data_prompts.py
│   ├── memory/
│   │   └── chroma_store.py    # ChromaMemoryStore
│   ├── models/                # Mô hình dự đoán (stubs)
│   │   ├── lstm_model.py
│   │   ├── convlstm_model.py
│   │   ├── utae_model.py
│   │   └── firecastnet_model.py
│   └── cli/
│       └── __init__.py        # Typer CLI: calm plan, calm version
├── examples/
│   ├── 01_planning.py
│   ├── 02_prediction_rsen.py
│   ├── 03_wildfire_qa.py
│   └── 04_full_pipeline.py
├── tests/
│   ├── conftest.py
│   ├── test_planning_agent.py
│   ├── test_rsen_module.py
│   ├── test_qa_agent.py
│   ├── test_safety_check.py
│   └── test_data_knowledge_agent.py
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## API Reference

| Component | Mô tả |
|-----------|-------|
| `PlanningAgent` | Phân rã query → JSON plan (URSA 3-node) |
| `DataKnowledgeAgent` | collect → extract → retrieve (GEE, CDS, Web, ArXiv) |
| `RSENModule` | validate(prediction, met_data, spatial_data) → Plausible/Implausible |
| `WildfireQAAgent` | QA pipeline với Evidence Evaluator |
| `EvaluatorAgent` | evaluate(response, query) → scores, passed |
| `ExecutionAgent` | execute_step(step, context) → result |
| `SafetyChecker` | is_safe(action), check_or_raise(action) |
| `ChromaMemoryStore` | add_texts(), similarity_search() |

---

## Tests

```bash
pytest tests/ -v
```

Tất cả API calls được mock (NP-7.2). Fixtures trong `conftest.py`:

- `mock_llm` — LLM giả
- `mock_llm_plan_approved` — Luồng planning được duyệt
- `mock_rsen_plausible` — RSEN validation → Plausible

---

## Docker

```bash
docker build -t calm .
docker run calm calm --help
```

---

## Đóng góp

- Báo lỗi: mở Issue trên GitHub
- Cải tiến: Fork → Branch → Pull Request
- Tuân thủ kiến trúc URSA (3-node) và các Non-Functional Requirements (NFR) trong bài báo CALM

---

## License

Xem file LICENSE trong repository.
