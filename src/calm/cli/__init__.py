"""
CALM CLI — giao diện dòng lệnh dựa trên Typer.

Lệnh:
  calm run   <query>   — Tự động định tuyến (QA / Prediction) qua CALMOrchestrator
  calm plan  <query>   — Chỉ chạy Planning Agent, in plan JSON
  calm version         — In phiên bản
"""

import typer

from calm.utils.env_loader import load_env

app = typer.Typer(help="CALM — Adaptive Multimodal Wildfire Monitoring")


def main() -> None:
    """Điểm vào lệnh calm: nạp .env rồi chạy Typer."""
    load_env()
    app()


# ─────────────────────────────────────────────────────────────
# calm run — điểm vào chính: tự định tuyến QA / Prediction
# ─────────────────────────────────────────────────────────────

@app.command()
def run(
    query: str = typer.Argument(..., help="Wildfire monitoring query (QA hoặc Prediction)"),
    model: str = typer.Option(
        None,
        "--model",
        "-m",
        help="Model ID (e.g. gpt-4o, openai/gpt-4o cho OpenRouter)",
    ),
    chroma_dir: str = typer.Option(
        ".chroma",
        "--chroma-dir",
        help="Thư mục lưu ChromaDB",
    ),
) -> None:
    """
    Tự động định tuyến câu truy vấn sang đúng pipeline.

    Câu hỏi  → QA Pipeline   (DataAgent → ChromaDB → WildfireQAAgent)\n
    Dự đoán  → Prediction Pipeline (DataAgent → PredictionAgent → RSEN)
    """
    import os

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    try:
        # ── LLM ─────────────────────────────────────────────────────────
        if os.environ.get("OPENROUTER_API_KEY"):
            from langchain_openrouter import ChatOpenRouter

            llm = ChatOpenRouter(
                model=model or os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o"),
                api_key=os.environ["OPENROUTER_API_KEY"],
                temperature=0.0,
            )
        elif os.environ.get("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=model or os.environ.get("OPENAI_MODEL", "gpt-4o"),
                openai_api_key=os.environ["OPENAI_API_KEY"],
                temperature=0.0,
            )
        else:
            raise ValueError(
                "Đặt OPENAI_API_KEY hoặc OPENROUTER_API_KEY trong .env"
            )

        # ── Memory + Tools ───────────────────────────────────────────────
        from calm.memory.chroma_store import ChromaMemoryStore
        from calm.orchestrator import CALMOrchestrator
        from calm.tools.web_search import WebSearchTool

        memory = ChromaMemoryStore(
            collection_name="calm_cli",
            persist_directory=chroma_dir,
            use_openai_embeddings=bool(os.environ.get("OPENAI_API_KEY")),
        )
        tools: dict = {}
        try:
            tools["web_search"] = WebSearchTool()
        except Exception:
            pass

        # ── Orchestrator ─────────────────────────────────────────────────
        orchestrator = CALMOrchestrator.from_llm(
            llm=llm,
            memory_store=memory,
            tools=tools,
        )

        console.print(f"\n[bold cyan]Query:[/bold cyan] {query}")
        console.print("[dim]Đang phân tích và định tuyến...[/dim]\n")

        result = orchestrator.run(query)
        task_type = result.get("task_type", "?")

        # ── Hiển thị kết quả ─────────────────────────────────────────────
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("key", style="bold green", width=18)
        table.add_column("value")

        table.add_row("Pipeline", task_type.upper())
        table.add_row("Plan steps", str(len(result.get("plan_steps", []))))

        if task_type == "qa":
            table.add_row(
                "Answer",
                str(result.get("answer", ""))[:400],
            )
            table.add_row(
                "Confidence",
                f"{result.get('confidence', 0.0):.2f}",
            )
            citations = result.get("citations", [])
            if citations:
                table.add_row("Citations", citations[0])
        else:
            table.add_row("Risk level", result.get("risk_level", "?"))
            table.add_row("Confidence", f"{result.get('confidence', 0.0):.2f}")
            table.add_row("RSEN decision", result.get("decision", "?"))
            rationale = str(result.get("rationale", ""))
            if rationale:
                table.add_row("Rationale", rationale[:400])

        if result.get("error"):
            table.add_row("[red]Error[/red]", str(result["error"]))

        console.print(
            Panel(table, title=f"[bold]CALM — {task_type.upper()} Result[/bold]",
                  border_style="green")
        )

    except ValueError as e:
        console.print(f"[red]Lỗi cấu hình: {e}[/red]")
        raise typer.Exit(1)
    except ImportError as e:
        console.print(f"[red]Import error: {e}[/red]")
        raise typer.Exit(1)


# ─────────────────────────────────────────────────────────────
# calm plan — chỉ chạy Planning Agent, in plan JSON
# ─────────────────────────────────────────────────────────────

@app.command()
def plan(
    query: str = typer.Argument(..., help="Wildfire monitoring query"),
    model: str = typer.Option(
        None,
        "--model",
        "-m",
        help="Model ID (e.g. gpt-4o, openai/gpt-4o for OpenRouter)",
    ),
) -> None:
    """Chạy Planning Agent với câu truy vấn giám sát cháy rừng."""
    import os

    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    try:
        from calm.agents.planning_agent import PlanningAgent

        if os.environ.get("OPENROUTER_API_KEY"):
            from langchain_openrouter import ChatOpenRouter

            llm = ChatOpenRouter(
                model=model or os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o"),
                api_key=os.environ["OPENROUTER_API_KEY"],
                temperature=0.0,
            )
        elif os.environ.get("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=model or os.environ.get("OPENAI_MODEL", "gpt-4o"),
                openai_api_key=os.environ["OPENAI_API_KEY"],
                temperature=0.0,
            )
        else:
            raise ValueError("Đặt OPENAI_API_KEY hoặc OPENROUTER_API_KEY trong .env")

        agent = PlanningAgent(llm=llm, config={})
        result = agent.invoke(query)
        plan_steps = result.get("final_output") or []
        console.print(
            Panel(str(plan_steps), title="Plan", border_style="green")
        )
    except ValueError as e:
        console.print(
            "[red]Lỗi cấu hình: Thiếu OPENAI_API_KEY hoặc OPENROUTER_API_KEY.[/red]"
        )
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except ImportError as e:
        console.print(f"[red]Import error: {e}[/red]")
        raise typer.Exit(1)


# ─────────────────────────────────────────────────────────────
# calm version
# ─────────────────────────────────────────────────────────────

@app.command()
def version() -> None:
    """Show CALM version."""
    from calm import __version__

    typer.echo(f"calm {__version__}")
