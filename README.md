# CALM — Adaptive Multimodal Wildfire Monitoring

**Hệ thống AI nghiên cứu giám sát cháy rừng thích ứng đa phương thức**, sử dụng kiến trúc agentic (URSA) để đánh giá rủi ro, thu thập dữ liệu đa nguồn và trả lời câu hỏi dựa trên bằng chứng.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Đặt vấn đề (Problem Statement)

### Vấn đề cần giải quyết

- **Giám sát cháy rừng** đòi hỏi tổng hợp dữ liệu từ nhiều nguồn (vệ tinh, khí tượng, tin tức, tài liệu khoa học) và đưa ra quyết định dựa trên bằng chứng. Các hệ thống hiện tại thường:
  - Thiếu **kế hoạch có cấu trúc** (structured planning) để phân rã câu hỏi phức tạp thành các bước thực thi.
  - Không **xác thực dự đoán** bằng chuyên gia độc lập (thời tiết vs. địa lý), dẫn đến sai lệch.
  - Dễ **hallucination** khi trả lời QA mà không gắn chặt với nguồn dữ liệu.
  - Thiếu **safety checks** trước khi gọi tool (API, vệ tinh, shell), gây rủi ro vận hành.

### Hạn chế của các phương pháp hiện tại

- Pipeline cố định, không phản xạ (reflect) và formalize output → khó đảm bảo JSON/format hợp lệ.
- Thiếu memory có cấu trúc (vector store theo từng agent) để tái sử dụng kinh nghiệm.
- Đánh giá chất lượng output thường thủ công; thiếu LLM-as-a-Judge có tiêu chí rõ ràng.

### Tại sao cần hệ thống này

- CALM kế thừa kiến trúc **URSA** (Universal Research and Scientific Agent, LANL): mỗi agent theo luồng **generator → reflector → formalizer**, đảm bảo output được phản ánh và chuẩn hóa.
- **RSEN** (Reflexive Structured Experts Network) tách biệt chuyên gia thời tiết và địa lý, tổng hợp qua Ops Coordinator → quyết định Plausible/Implausible có căn cứ.
- **Evidence Evaluator** và **Safety Check** trước mỗi lệnh tool giảm hallucination và rủi ro vận hành.

### Ứng dụng thực tế

- Đánh giá rủi ro cháy rừng theo vùng và khung thời gian (ví dụ: Amazon, California, 7–14 ngày).
- Tổng hợp báo cáo từ GEE, Copernicus CDS, tin tức, ArXiv để trả lời câu hỏi dạng “Nguyên nhân cháy Canada 2023?”.
- Xác thực dự đoán từ FireCastNet/LSTM/ConvLSTM/UTAE bằng RSEN trước khi đưa ra cảnh báo.

---

## 2. Phương pháp đề xuất (Proposed Method)

### Tổng quan kiến trúc

- **Luồng 3 node (URSA):** Mọi agent đều có **Generator** (tạo output ban đầu) → **Reflector** (phản ánh, chỉnh sửa, quyết định `[APPROVED]` hoặc lặp) → **Formalizer** (chuyển sang JSON hợp lệ, tối đa `f_max` lần thử). Implement bằng **LangGraph** `StateGraph(AgentState)`.
- **Memory:** ChromaDB với collection riêng cho từng module (`calm_rsen_memory`, `calm_qa_memory`, `calm_plan_memory`), embedding OpenAI `text-embedding-3-small` hoặc HuggingFace `sentence-transformers/all-MiniLM-L6-v2`, top-k retrieval và `similarity_threshold`.
- **Reasoning:** RSEN gồm Weather Analyst, Geo Analyst (chạy song song, cách ly), Ops Coordinator tổng hợp → Plausible/Implausible.
- **Agent system:** PlanningAgent, DataKnowledgeAgent, RSENModule, WildfireQAAgent, ExecutionAgent, EvaluatorAgent; mỗi agent nhận instance LLM (ChatOpenRouter hoặc ChatOpenAI tạo từ env).

### Thuật toán chính

- **Planning:** Phân rã query → danh sách bước JSON (step type, params); Reflector kiểm tra tính đầy đủ và hợp lệ; Formalizer xuất JSON cuối.
- **Retrieval:** `ChromaMemoryStore.add_texts()` + `similarity_search()` với `k` và `similarity_threshold`; dùng cho reflexion (lưu failure/success để tránh lặp).
- **Evidence evaluation:** Điểm bằng chứng so với `evidence_threshold`; nếu dưới ngưỡng → kích hoạt tìm kiếm bổ sung (web/ArXiv) rồi mới trả lời.

### Mô hình AI sử dụng

- **LLM:** OpenAI (gpt-4o) hoặc OpenRouter (Claude, Gemini, v.v.) — tạo trực tiếp ChatOpenAI/ChatOpenRouter từ `OPENAI_API_KEY` hoặc `OPENROUTER_API_KEY`; tùy chọn Ollama (local).
- **Embedding:** `text-embedding-3-small` (OpenAI) hoặc `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace).
- **Dự đoán cháy:** FireCastNet, LSTM, ConvLSTM, UTAE (stubs trong `models/`), checkpoint ví dụ `firecastnet_seasfire.pt`.

### Pipeline / Workflow

1. **Planning:** Query → PlanningAgent (URSA 3-node) → `final_output` là plan JSON.
2. **Data:** DataKnowledgeAgent gọi tools (Earth Engine, Copernicus, Web Search, ArXiv) với safety check; collect → extract → retrieve; lưu vào cache và memory.
3. **Prediction:** Mô hình dự báo (FireCastNet, v.v.) → RSEN validate (met_data + spatial_data) → Plausible/Implausible.
4. **QA:** WildfireQAAgent dùng Evidence Evaluator; nếu đủ bằng chứng → trả lời; không đủ → search thêm.
5. **Evaluation:** EvaluatorAgent (LLM-as-a-Judge) chấm 5 tiêu chí (0–100), so sánh với `passing_score`.

---

## 3. Kiến trúc hệ thống (System Architecture)

| Thành phần | Mô tả |
|------------|--------|
| **LLM** | Tạo trực tiếp: ưu tiên `OPENROUTER_API_KEY` → ChatOpenRouter, không có thì `OPENAI_API_KEY` → ChatOpenAI; có thể set `OPENROUTER_MODEL` / `OPENAI_MODEL`. |
| **Planning agent** | `PlanningAgent(BaseCALMAgent)`: phân rã query thành plan JSON; 3 node generator/reflector/formalizer; `n_max` (số lần reflector lặp), `f_max` (số lần formalizer thử). |
| **Memory store** | `ChromaMemoryStore`: collection name, `persist_directory`, embedding model, `k`, `similarity_threshold`; `add_texts()`, `similarity_search()`; hỗ trợ OpenAI hoặc HuggingFace embeddings. |
| **Embedding** | Mặc định HuggingFace `all-MiniLM-L6-v2`; tùy chọn OpenAI `text-embedding-3-small` khi `use_openai_embeddings=True` và có `OPENAI_API_KEY`. |
| **Retrieval** | Top-k vector search trong Chroma; lọc theo `similarity_threshold`; dùng trong RSEN, QA, Planning để reflexion. |
| **Tool integration** | Earth Engine, Copernicus CDS, DuckDuckGo (web), ArXiv; mỗi lệnh qua `SafetyChecker` (URSA pattern); cấu hình `dedup_check`, `max_news_results`, `max_arxiv_papers`. |
| **RSEN** | Weather Analyst + Geo Analyst (song song), Ops Coordinator; memory collection `calm_rsen_memory`; output `validation_decision`, `final_rationale`. |
| **Evidence Evaluator** | Trong QA pipeline; ngưỡng `evidence_threshold` (mặc định 0.65); trigger search khi không đủ bằng chứng. |
| **Evaluator** | LLM-as-a-Judge; 5 tiêu chí: Data-Accuracy, Explainability, Jargon-Avoidance, Redundancy-Avoidance, Citation-Quality; `passing_score` (mặc định 75). |

### Luồng dữ liệu

- **Input:** Câu hỏi/query dạng text (ví dụ: “Wildfire risk California 14 days”).
- **Planning:** Query → PlanningAgent → state `conversation`, `approved`, `final_output` (plan).
- **Data/Memory:** DataKnowledgeAgent đọc config (GEE, CDS, cache_dir), gọi tools → dữ liệu thô → extract → lưu vào Chroma (collection tương ứng) và cache.
- **RSEN:** prediction + met_data + spatial_data → từng analyst → Ops Coordinator → decision + rationale; có thể ghi lại vào `calm_rsen_memory`.
- **QA:** Query → retrieval từ memory → Evidence Evaluator → nếu đủ → LLM trả lời; không đủ → web/ArXiv → lặp tối đa `max_search_iterations`.
- **Output:** Plan JSON, quyết định RSEN, câu trả lời QA, hoặc bảng điểm Evaluator.

---

## 4. Nguyên tắc hoạt động (Operating Principles)

1. **Input processing:** Query được truyền vào agent dưới dạng string; state khởi tạo: `query`, `conversation=[]`, `approved=False`, `iteration=0`, `final_output=None`, `error=None`.
2. **Planning stage:** Node Generator tạo plan từ query; Reflector đọc output, quyết định [APPROVED] hoặc gửi lại Generator (tối đa `n_max` lần); khi [APPROVED] hoặc hết vòng → chuyển Formalizer.
3. **Retrieval / memory:** Trước khi tạo output, agent có thể gọi `memory_store.similarity_search(query, k)` để lấy ngữ cảnh tương tự; sau khi có kết quả (thành công/thất bại), có thể gọi `add_texts()` để cập nhật reflexion.
4. **Reasoning:** Trong RSEN, Weather Analyst và Geo Analyst nhận cùng input nhưng xử lý độc lập; Ops Coordinator tổng hợp báo cáo và đưa ra quyết định cuối. Trong QA, Evidence Evaluator đánh giá mức độ hỗ trợ của tài liệu trước khi trả lời.
5. **Output generation:** Formalizer chuyển nội dung đã được Reflector duyệt sang JSON hợp lệ (ví dụ danh sách bước); nếu parse thất bại thì thử lại tối đa `f_max` lần. Kết quả cuối nằm trong `state["final_output"]`.

---

## 5. Technology Stack

| Hạng mục | Công nghệ |
|----------|-----------|
| **Ngôn ngữ** | Python ≥ 3.10 |
| **Frameworks** | LangChain (≥0.3), LangGraph (≥0.2), LangChain-OpenAI, LangChain-OpenRouter, LangChain-Ollama, LangChain-Community, LangChain-Chroma, LangChain-Text-Splitters |
| **Vector database** | Chroma (chromadb, langchain-chroma); persist trên disk qua `persist_directory` |
| **Embedding** | OpenAI `text-embedding-3-small` (tùy chọn) hoặc sentence-transformers `all-MiniLM-L6-v2` (mặc định) |
| **LLM API** | OpenAI API, OpenRouter API; tùy chọn Ollama (local) |
| **Dữ liệu / tools** | Earth Engine (earthengine-api), Copernicus CDS (cdsapi), DuckDuckGo (duckduckgo-search), ArXiv (arxiv), BeautifulSoup, Trafilatura, PyMuPDF, pypdf |
| **Khoa học dữ liệu** | NumPy, Pandas, xarray, rasterio, pyproj, Pillow, scikit-learn |
| **Deep learning** | PyTorch, torchvision (cho FireCastNet, LSTM, ConvLSTM, UTAE) |
| **CLI / config** | Typer, Rich, PyYAML, jsonargparse |
| **Tùy chọn** | GPU (CUDA) cho mô hình dự đoán; OpenTelemetry (otel) cho tracing; pytest/ruff/mypy (dev) |

---

## 6. Hướng dẫn cài đặt (Installation Guide)

### 6.1 Yêu cầu

- **Python:** ≥ 3.10.
- **CUDA:** Tùy chọn, dùng cho mô hình dự đoán (FireCastNet, v.v.) khi chạy trên GPU.
- **API keys:** Ít nhất một trong hai: `OPENAI_API_KEY` hoặc `OPENROUTER_API_KEY`. Dùng GEE thì cần `GEE_PROJECT_ID` và `earthengine authenticate`; dùng Copernicus CDS thì cần `COPERNICUS_API_KEY` và file `~/.cdsapirc`.

### 6.2 Clone repository

```bash
git clone https://github.com/viethungvu1998/calm.git
cd calm
```

### 6.3 Tạo môi trường ảo

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

### 6.4 Cài đặt dependencies

Dự án dùng `pyproject.toml` (Hatch), không bắt buộc file `requirements.txt`. Cài package có thể chỉnh sửa:

```bash
pip install -e ".[dev]"
```

Tùy chọn: `.[otel]` nếu cần OpenTelemetry.

### 6.5 Biến môi trường

Tạo file `.env` trong thư mục `calm` (hoặc export trong shell):

```bash
# Chọn một trong hai:
OPENROUTER_API_KEY=sk-or-v1-...
# hoặc
OPENAI_API_KEY=sk-...

# Tùy chọn
GEE_PROJECT_ID=your-gee-project
COPERNICUS_API_KEY=your-cds-api-key
```

Xác thực Earth Engine (nếu dùng GEE):

```bash
earthengine authenticate
```

Cấu hình Copernicus CDS (nếu dùng ERA5): tạo `~/.cdsapirc`:

```
url: https://cds.climate.copernicus.eu/api/v2
key: <uid>:<api-key>
```

### 6.6 Chạy demo

Tạo file `.env` từ mẫu (xem `.env.example`) và đặt ít nhất một trong hai:
`OPENAI_API_KEY` hoặc `OPENROUTER_API_KEY`.

```bash
# Cách 1: Điểm vào chính (tự nạp .env, chạy Planning Agent)
python main.py

# Cách 2: CLI (nạp .env trong calm plan)
calm plan "Wildfire risk assessment for Amazon region next 7 days"

# Cách 3: Các script ví dụ (cần pip install -e . trước)
python examples/01_planning.py
python examples/02_prediction_rsen.py
python examples/03_wildfire_qa.py
python examples/04_full_pipeline.py

# Cách 4: Jupyter notebook (demo đầy đủ theo paper: Planning, RSEN, QA, Pipeline, Execution)
jupyter notebook calm_demo.ipynb
```

---

## 7. Cấu trúc dự án (Project Structure)

```
calm/
├── configs/
│   └── example.yaml           # Cấu hình mẫu (LLM, planning, data_knowledge, prediction, rsen, qa, evaluator, memory)
├── src/calm/
│   ├── __init__.py
│   ├── agents/                # Các agent (URSA 3-node)
│   │   ├── base_agent.py      # BaseCALMAgent, AgentState, StateGraph (generator/reflector/formalizer)
│   │   ├── planning_agent.py  # PlanningAgent — phân rã query → plan JSON
│   │   ├── data_knowledge_agent.py  # Thu thập GEE, CDS, Web, ArXiv; collect → extract → retrieve
│   │   ├── execution_agent.py # Thực thi từng bước kế hoạch
│   │   ├── rsen_module.py     # RSEN: Weather Analyst, Geo Analyst, Ops Coordinator
│   │   ├── prediction_reasoning_agent.py
│   │   ├── qa_agent.py        # WildfireQAAgent + Evidence Evaluator
│   │   └── evaluator_agent.py # LLM-as-a-Judge, 5 tiêu chí
│   ├── tools/                 # Công cụ với safety check (URSA pattern)
│   │   ├── earth_engine.py
│   │   ├── copernicus.py
│   │   ├── web_search.py
│   │   ├── arxiv_tool.py
│   │   ├── safety_check.py
│   │   └── wildfire_models.py
│   ├── prompt_library/        # Prompts (planning, rsen, qa, data)
│   ├── memory/
│   │   └── chroma_store.py    # ChromaMemoryStore — Reflexion, top-k, similarity_threshold
│   ├── models/                # Mô hình dự đoán (FireCastNet, LSTM, ConvLSTM, UTAE)
│   ├── cli/                   # Typer CLI: calm plan, calm version
│   └── utils/                 # load_env(), get_env() — nạp biến môi trường từ .env
├── main.py                    # Điểm vào demo: python main.py
├── calm_demo.ipynb            # Notebook demo theo paper (URSA, RSEN, QA, Evaluator, Execution)
├── .env.example               # Mẫu biến môi trường (sao thành .env)
├── examples/
│   ├── 01_planning.py
│   ├── 02_prediction_rsen.py
│   ├── 03_wildfire_qa.py
│   └── 04_full_pipeline.py
├── tests/
│   ├── conftest.py            # Fixtures: mock_llm, mock_rsen_plausible, ...
│   ├── test_planning_agent.py
│   ├── test_rsen_module.py
│   ├── test_qa_agent.py
│   ├── test_safety_check.py
│   └── test_data_knowledge_agent.py
├── pyproject.toml
├── Dockerfile
└── README.md
```

- **agents:** Logic nghiệp vụ chính; mỗi agent kế thừa base 3-node, có thể dùng memory và tools riêng.
- **memory:** Lưu trữ vector (Chroma), dùng cho reflexion và retrieval theo từng collection.
- **retrieval:** Thực hiện trong `ChromaMemoryStore.similarity_search()`; embedding trong chroma_store.
- **LLM:** Tạo ChatOpenRouter hoặc ChatOpenAI từ env tại điểm vào (main, CLI, examples, notebook); agent nhận instance LLM qua tham số.
- **tools:** Giao tiếp API bên ngoài (GEE, CDS, web, ArXiv); mọi lệnh qua SafetyChecker.
- **utils:** Có thể mở rộng; hiện tại config đọc qua YAML/jsonargparse.

---

## 8. Ví dụ sử dụng (Example Usage)

### Load hệ thống và chạy Planning Agent

```python
import os
from calm.utils.env_loader import load_env
load_env()

if os.environ.get("OPENROUTER_API_KEY"):
    from langchain_openrouter import ChatOpenRouter
    llm = ChatOpenRouter(model="openai/gpt-4o", api_key=os.environ["OPENROUTER_API_KEY"], temperature=0.0)
elif os.environ.get("OPENAI_API_KEY"):
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=os.environ["OPENAI_API_KEY"], temperature=0.0)
else:
    raise ValueError("Đặt OPENAI_API_KEY hoặc OPENROUTER_API_KEY trong .env")

from calm.agents.planning_agent import PlanningAgent
agent = PlanningAgent(llm=llm, config={}, n_max=3, f_max=3)
result = agent.invoke("Wildfire risk assessment for Amazon region next 7 days")
print(result["final_output"])
print("Approved:", result["approved"])
```

### RSEN — Xác thực dự đoán

```python
import os
from calm.utils.env_loader import load_env
load_env()
if os.environ.get("OPENROUTER_API_KEY"):
    from langchain_openrouter import ChatOpenRouter
    llm = ChatOpenRouter(model="openai/gpt-4o", api_key=os.environ["OPENROUTER_API_KEY"], temperature=0.0)
elif os.environ.get("OPENAI_API_KEY"):
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=os.environ["OPENAI_API_KEY"], temperature=0.0)
else:
    raise ValueError("Đặt OPENAI_API_KEY hoặc OPENROUTER_API_KEY trong .env")

from calm.agents.rsen_module import RSENModule
from calm.memory.chroma_store import ChromaMemoryStore
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

### Wildfire QA Agent (có Evidence Evaluator)

```python
import os
from calm.utils.env_loader import load_env
load_env()
if os.environ.get("OPENROUTER_API_KEY"):
    from langchain_openrouter import ChatOpenRouter
    llm = ChatOpenRouter(model="openai/gpt-4o", api_key=os.environ["OPENROUTER_API_KEY"], temperature=0.0)
elif os.environ.get("OPENAI_API_KEY"):
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=os.environ["OPENAI_API_KEY"], temperature=0.0)
else:
    raise ValueError("Đặt OPENAI_API_KEY hoặc OPENROUTER_API_KEY trong .env")

from calm.agents.data_knowledge_agent import DataKnowledgeAgent
from calm.agents.qa_agent import WildfireQAAgent
from calm.memory.chroma_store import ChromaMemoryStore
from calm.tools.safety_check import SafetyChecker
from calm.tools.web_search import WebSearchTool
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

---

## 9. Hướng dẫn phát triển (Development Guidelines)

- **Quy ước code:** Tuân thủ kiến trúc URSA (3 node: generator → reflector → formalizer); mỗi agent kế thừa `BaseCALMAgent` và implement `_generator_node`, `_reflector_node`, `_formalizer_node`. Gọi tool luôn qua `SafetyChecker`. Code style: Ruff, type hints (Mypy tùy chọn).
- **Thiết kế modular:** Agent nhận instance LLM (ChatOpenRouter/ChatOpenAI tạo từ env tại điểm vào); memory nhận interface có `add_texts` và `similarity_search`; config đọc từ YAML (ví dụ `configs/example.yaml`).
- **Đóng góp:** Báo lỗi qua GitHub Issues; cải tiến qua Fork → Branch → Pull Request. Giữ nguyên NFR và safety criteria (không cảnh báo khẩn cấp chưa xác minh, không xóa/ghi đè dữ liệu vệ tinh/checkpoint, không gọi API thiếu credentials, không thực thi shell thay đổi trạng thái hệ thống).

---

## 10. Công việc tương lai (Future Work)

- **Offline LLM:** Mở rộng hỗ trợ Ollama / local models đầy đủ cho toàn bộ pipeline (planning, RSEN, QA, evaluator) và cấu hình mặc định ổn định.
- **Multimodal input:** Đầu vào ảnh vệ tinh hoặc bản đồ trực tiếp vào Planning/DataKnowledge; tích hợp encoder đa phương thức cho retrieval.
- **Triển khai phân tán:** Chạy từng agent hoặc từng analyst (RSEN) trên worker riêng; hàng đợi task (Celery/RQ) và đồng bộ state/memory qua database hoặc object store.
- **Mô hình dự đoán:** Hoàn thiện training và checkpoint cho FireCastNet, LSTM, ConvLSTM, UTAE; tích hợp pipeline từ GEE/CDS → input tensor → prediction → RSEN.
- **Observability:** Tích hợp OpenTelemetry (otel) mặc định cho tracing toàn pipeline và log structured.

---

## 11. License

Dự án phân phối theo **MIT License**. Chi tiết xem file `LICENSE` trong repository.

---

## 12. Citation (Optional)

Nếu sử dụng CALM trong nghiên cứu, có thể trích dẫn theo định dạng:

```bibtex
@software{calm2026,
  title = {CALM: Adaptive Multimodal Wildfire Monitoring with Autonomous Agentic AI},
  author = {CALM Team},
  year = {2026},
  url = {https://github.com/viethungvu1998/calm},
  note = {Based on URSA (Universal Research and Scientific Agent) architecture}
}
```

---

## API Reference (tóm tắt)

| Thành phần | Mô tả |
|------------|--------|
| `PlanningAgent` | Phân rã query → JSON plan (URSA 3-node). |
| `DataKnowledgeAgent` | collect → extract → retrieve (GEE, CDS, Web, ArXiv). |
| `RSENModule` | `validate(prediction, met_data, spatial_data)` → Plausible/Implausible. |
| `WildfireQAAgent` | QA pipeline với Evidence Evaluator. |
| `EvaluatorAgent` | `evaluate(response, query)` → scores, passed. |
| `ExecutionAgent` | `execute_step(step, context)` → result. |
| `SafetyChecker` | `is_safe(action)`, `check_or_raise(action)`. |
| `ChromaMemoryStore` | `add_texts()`, `similarity_search()`. |
