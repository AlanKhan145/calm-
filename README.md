# CALM — Adaptive Multimodal Wildfire Monitoring

**CALM** là hệ thống agentic cho giám sát và phân tích cháy rừng: trả lời câu hỏi có trích dẫn (QA), ước lượng rủi ro / dự báo (prediction) với kiểm chứng RSEN, và thu thập dữ liệu đa nguồn (Google Earth Engine, Copernicus CDS, web, arXiv).

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## Tổng quan

Điểm vào chính là **`CALMOrchestrator.run(query)`** (hoặc CLI `calm run`, hoặc `python main.py`). Hệ thống:

1. **Chuẩn hoá ngữ cảnh** từ câu hỏi (địa điểm gợi ý, tọa độ trong text, khoảng thời gian; geocode nếu có tool; `resolve_time_range` — không ghi ngày thì mặc định hôm nay).
2. **Lập kế hoạch** — `PlanningAgent` (URSA) sinh plan dạng các bước JSON.
3. **Định tuyến** — `RouterAgent`: `qa` | `prediction` | `hybrid`. `hybrid` được **gom vào pipeline prediction** (cần cả bằng chứng và output mô hình).
4. **Thực thi** theo nhánh:
   - **Prediction:** pipeline riêng `normalize → retrieve → build_features → predict → validate (RSEN) → refine` (refine khi dữ liệu / validation yếu); **không** chạy prediction qua `ExecutionAgent` để tránh thiếu bướcs và refinement.
   - **QA:** hoặc `_qa_pipeline` (retrieve + `WildfireQAAgent`), hoặc `_run_plan_driven` qua `ExecutionAgent` khi có plan và executor.
5. **Bộ nhớ** — lưu episode / short-term qua `MemoryAgent` + Chroma (nếu cấu hình).

Heuristic định tuyến khi LLM thiếu ổn định nằm ở `calm/utils/intent_hints.py` (tránh nhầm câu hỏi “risk” chung chung với tác vụ dự báo).

### Điểm mạnh

| Tính năng | Mô tả |
|-----------|--------|
| Hai pipeline rõ ràng | QA vs prediction tách luồng; prediction có validate + refine |
| SeasFire (tùy chọn) | Checkpoint GRU qua `SeasFireModelRunner`; thiếu model / feature → heuristic có kiểm soát |
| Đa nguồn dữ liệu | GEE, Copernicus, tìm kiếm web, arXiv; geocoding cho địa chỉ tự nhiên |
| RSEN | Hai chuyên gia (khí tượng / địa lý) song song → điều phối → Plausible / Implausible |
| Safety | `SafetyChecker` trước khi gọi tool (tùy luồng) |
| Đánh giá chất lượng | `EvaluatorAgent` — LLM chấm nhiều tiêu chí (tùy dùng) |

---

## Cài đặt

### 1. Clone và môi trường

```bash
git clone <url-repo-calms của bạn>
cd <thư-mục-repo>    # thường là thư mục chứa pyproject.toml và src/calm/
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 2. PyTorch CPU-only (máy không GPU)

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt
```

### 3. Biến môi trường

Sao chép `.env.example` → `.env` và điền tối thiểu:

```bash
# Một trong hai (bắt buộc cho LLM)
OPENROUTER_API_KEY=...
# hoặc
OPENAI_API_KEY=...

# Tùy chọn — bật tool tương ứng
GEE_PROJECT_ID=...
COPERNICUS_API_KEY=...
SEASFIRE_ML_PATH=/path/to/seasfire-ml
SEASFIRE_DATASET_PATH=/path/to/datacube   # nếu dùng feature builder đầy đủ
```

Chi tiết thêm trong `.env.example`.

### 4. Chạy nhanh

```bash
python main.py
# hoặc (sau pip install -e .)
calm run "What causes wildfires in the Amazon?"
calm run "Predict wildfire risk for California over the next 7 days"
```

Notebook kiểm thử end-to-end: **`test.ipynb`** (mở kernel từ thư mục repo để cell setup tự nhận `CALM_DIR`).

---

## Luồng `CALMOrchestrator.run` (thực tế trong code)

```
Query
  │
  ▼
_normalize_query_context (text, location hint, coordinates, time_range)
  │
  ▼
PlanningAgent → plan_steps (JSON)
  │
  ▼
RouterAgent → task_type: qa | prediction | hybrid
              (hybrid → xử lý như prediction)
  │
  ├─ prediction ──► Prediction pipeline
  │                   retrieve (DataKnowledgeAgent)
  │                   → met_data / spatial_data
  │                   → PredictionReasoningAgent.predict
  │                   → RSEN.validate
  │                   → refine (re-query / mở rộng time / downgrade confidence nếu cần)
  │
  └─ qa ──────────► Nếu có ExecutionAgent + plan_steps: plan-driven execution
        hoặc       Ngược lại: QA pipeline (retrieve + WildfireQAAgent)
  │
  ▼
MemoryAgent (episode + short-term) nếu có
```

`ExecutionAgent` vẫn dùng cho **plan-driven QA** và các bước tổng quát; **không** thay thế pipeline prediction ở trên.

---

## Cấu trúc repo (rút gọn)

```
├── configs/
│   └── example.yaml          # Cấu hình mẫu (agent_config, prediction checkpoint, …)
├── src/calm/
│   ├── orchestrator.py       # CALMOrchestrator
│   ├── cli/                  # Typer: calm run | calm plan | calm version
│   ├── agents/               # Planning, Router, Execution, Data, QA, Prediction, RSEN, Memory, Evaluator, …
│   ├── artifact/             # SeasFire runner, feature builder, model_runner, …
│   ├── tools/                # earth_engine, copernicus, web_search, geocoding, arxiv, safety_check, …
│   ├── schemas/              # contracts, query_context, …
│   ├── prompt_library/       # Prompt theo module
│   ├── memory/               # chroma_store, reranker
│   └── utils/                # env_loader, time_utils, intent_hints, …
├── test.ipynb                  # Kiểm tra từng agent + luồng orchestrator
├── main.py                   # Demo 2 query: QA + prediction
├── Dockerfile
├── requirements.txt
└── pyproject.toml            # package tên `calm`, script entry: calm
```

---

## Ví dụ Python

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path("src").resolve()))

from calm.utils.env_loader import load_env
load_env(Path(".env"))

from langchain_openai import ChatOpenAI
from calm.memory.chroma_store import ChromaMemoryStore
from calm.orchestrator import CALMOrchestrator

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
memory = ChromaMemoryStore(collection_name="calm", persist_directory=".chroma")

# from_llm gắn sẵn tool mặc định (GEE, Copernicus, web, arXiv, geocoding) nếu không truyền
orc = CALMOrchestrator.from_llm(llm=llm, memory_store=memory, tools={}, config={})

r = orc.run("What causes wildfires in the Amazon?")
print(r["task_type"], r.get("answer", "")[:200], r.get("citations", [])[:1])

r2 = orc.run("Predict wildfire risk for California next 7 days")
print(r2["task_type"], r2.get("risk_level"), r2.get("decision"), r2.get("rationale", "")[:200])
```

### Gắn checkpoint SeasFire

Trong `configs/example.yaml` (hoặc `config` dict) đặt `agent_config.prediction.checkpoint` trỏ tới file `.pt` sau khi train (xem comment trong file). Nếu không có checkpoint hoặc build runner lỗi, orchestrator log rõ và prediction agent có thể dùng chế độ fallback (tùy `allow_prediction_fallback`).

---

## Cấu hình (tham chiếu `configs/example.yaml`)

| Thành phần | Khóa gợi ý | Ghi chú |
|------------|------------|---------|
| Planning | `agent_config.planning` | `n_max`, `f_max` |
| Data | `agent_config.data_knowledge` | GEE/CDS, dedup, giới hạn web/arXiv |
| Prediction | `agent_config.prediction` | `checkpoint`, `timesteps`, `target_week`, … |
| RSEN | `agent_config.rsen` | `memory_k`, v.v. |
| QA | `agent_config.qa` | `evidence_threshold`, `max_search_iterations` |
| Evaluator | `agent_config.evaluator` | `passing_score`, tiêu chí |

`CALMOrchestrator` cũng đọc nhánh `prediction` ở top-level config nếu bạn truyền dict tương thích (xem `_resolve_prediction_config` trong code).

---

## Công nghệ

| Lớp | Công nghệ |
|-----|-----------|
| Ngôn ngữ | Python ≥ 3.10 |
| LLM / agent | LangChain, LangGraph (tùy agent), OpenAI / OpenRouter / Ollama (tùy cài) |
| Vector store | Chroma |
| Spatial / climate | Earth Engine API, cdsapi |
| Tìm kiếm | ddgs / DuckDuckGo, arXiv |
| Model | PyTorch; tích hợp SeasFire GRU (optional) |

---

## API tóm tắt

| Thành phần | Vai trò |
|------------|---------|
| `CALMOrchestrator` | `run(query)`; `from_llm(...)` khởi tạo đủ agent + executor |
| `PlanningAgent` | `invoke(query)` → plan |
| `RouterAgent` | `route(query, plan_steps)` → `TaskRouting` |
| `DataKnowledgeAgent` | `retrieve(query, params)` |
| `PredictionReasoningAgent` | `predict(params)` |
| `WildfireQAAgent` | `invoke(query, pre_retrieved=…)` |
| `RSENModule` | `validate(prediction, met_data, spatial_data)` |
| `EvaluatorAgent` | `evaluate(response, query)` |

---

## Phát triển

```bash
pip install -e ".[dev]"   # pytest, ruff, mypy (xem pyproject.toml)
ruff check src
pytest    # khi có thư mục tests/
```

---

## License

MIT (theo định hướng dự án). Khi công khai repo, nên thêm file `LICENSE` rõ ràng.
