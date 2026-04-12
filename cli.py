import typer

app = typer.Typer(help="FTIR sports ML — CLI")


@app.command()
def train(
    config: str = typer.Argument(..., help="Path to experiment JSON config"),
):
    """Run training pipeline defined by a JSON experiment config."""
    from ftir.models.training import run_experiment
    run_experiment(config)


@app.command()
def streamlit():
    """Launch the Streamlit dashboard."""
    import subprocess
    import sys
    from pathlib import Path
    app_path = Path(__file__).parent / "app" / "main.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


if __name__ == "__main__":
    app()
