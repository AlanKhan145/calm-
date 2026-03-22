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

## Việc còn lại (khi có seasfire-ml)

1. **SeasFireFeatureBuilder** — đọc zarr, tạo features
2. **SeasFireRunner.predict_batch()** — load checkpoint, chạy inference, lưu parquet
3. **PredictionReasoningAgent** — đọc từ ArtifactStore thay vì gọi model trực tiếp
4. **Nối data → features → model** — prediction pipeline truyền features thật vào model
