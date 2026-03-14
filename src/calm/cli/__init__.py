"""
File: __init__.py
Description: CALM CLI — typer-based (NP-6.6).
Author: CALM Team
Created: 2026-03-13
"""

import typer

app = typer.Typer(help="CALM — Adaptive Multimodal Wildfire Monitoring")


def main() -> None:
    """Entry point for calm command."""
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
    """Run planning agent on a query."""
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    try:
        from calm.agents.planning_agent import PlanningAgent
        from calm.llm_factory import get_llm

        llm = get_llm(model=model)
        agent = PlanningAgent(llm=llm, config={})
        result = agent.invoke(query)
        plan_steps = result.get("final_output") or []
        console.print(
            Panel(str(plan_steps), title="Plan", border_style="green")
        )
    except ImportError as e:
        console.print(f"[red]Import error: {e}[/red]")


@app.command()
def version() -> None:
    """Show CALM version."""
    from calm import __version__

    typer.echo(f"calm {__version__}")
