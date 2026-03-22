# CALM Refactor Blueprint

Refactor theo phản hồi: bỏ pipeline cứng, chuyển sang plan-driven execution, chuẩn hóa contract, artifact store.

## Đã triển khai

### 1. Pydantic Schemas (`calm/schemas/`)
- `PlanStep`, `TaskPlan` — plan chuẩn
- `TaskRouting` — RouterAgent output (task_type, confidence, required_artifacts)
- `RetrievedEvidence`, `PredictionOutput`, `ValidationOutput`, `QAResponse` — contract giữa các module

### 2. RouterAgent (`calm/agents/router_agent.py`)
- LLM-based routing thay keyword fallback
- Trả về `TaskRouting` với task_type, confidence, required_artifacts
- Fallback keyword khi LLM fail

### 3. Plan-driven Execution
- `CALMOrchestrator.run()`: PlanningAgent → RouterAgent → ExecutionAgent chạy từng step
- ExecutionAgent đã có routing table; orchestrator dùng nó thay vì _qa_pipeline / _prediction_pipeline
- Khi không có executor: fallback về pipeline cũ (backward compatible)

### 4. Fix duplicate retrieval
- `WildfireQAAgent.invoke(query, pre_retrieved=None)` — nhận pre-retrieved từ orchestrator
- Orchestrator gọi `data_agent.retrieve` một lần, truyền `pre_retrieved` vào QA
- ExecutionAgent gọi QA với `context["data_result"]` khi có

### 5. RSEN field mismatch
- RSEN trả `validation_decision`; orchestrator đọc cả `validation_decision` và `decision` (fallback)

### 6. Dedup chuẩn hóa
- `DataKnowledgeAgent._dedup_key(query, location, time_range)` — key có location + time
- Tránh dedup sai khi cùng wording nhưng khác vùng/thời gian

### 7. Artifact Store + SeasFireRunner stubs
- `calm/artifact/prediction_store.py` — PredictionArtifactStore (cache predictions theo location+time+model)
- `calm/artifact/seasfire_runner.py` — SeasFireRunner (stub, chưa nối seasfire-ml)

## Cấu trúc thư mục

```
calm/src/calm/
├── schemas/
│   ├── __init__.py
│   └── contracts.py      # PlanStep, TaskRouting, ValidationOutput, ...
├── artifact/
│   ├── __init__.py
│   ├── prediction_store.py   # PredictionArtifactStore
│   └── seasfire_runner.py    # SeasFireRunner (stub)
├── agents/
│   ├── router_agent.py       # RouterAgent
│   ├── ...
```

## SeasFire integration (đã triển khai)

1. **SeasFireRunner** — load seasfire-ml GRU checkpoint (SEASFIRE_ML_PATH), predict_batch với tensor hoặc heuristic fallback từ met_data
2. **SeasFireFeatureBuilder** — (lat, lon, time) → tensor từ dataset (SEASFIRE_DATASET_PATH, optional)
3. **SeasFireModelRunner** — adapter predict(params) cho PredictionReasoningAgent
4. **Orchestrator** — từ config `agent_config.prediction.checkpoint` tự tạo SeasFireModelRunner; truyền met_data/spatial_data vào predict
5. **Fallback heuristic** — khi không có model/features: dùng temp, humidity từ GEE/CDS
