import typer

from .hello import app as hello_app

app = typer.Typer()
app.add_typer(hello_app)
