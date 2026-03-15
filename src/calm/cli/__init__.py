"""
CALM CLI — giao diện dòng lệnh dựa trên Typer.

Cung cấp lệnh: calm plan <query>, calm version.
Nạp biến môi trường từ .env trước khi chạy agent.
"""

import typer

from calm.utils.env_loader import load_env

app = typer.Typer(help="CALM — Adaptive Multimodal Wildfire Monitoring")


def main() -> None:
    """Điểm vào lệnh calm: nạp .env rồi chạy Typer."""
    load_env()
    app()


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
        console.print("[red]Lỗi cấu hình: Thiếu OPENAI_API_KEY hoặc OPENROUTER_API_KEY.[/red]")
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except ImportError as e:
        console.print(f"[red]Import error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Show CALM version."""
    from calm import __version__

    typer.echo(f"calm {__version__}")
