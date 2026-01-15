import logging
import typer
from pathlib import Path

from perscit_model.xml_processing.tagger import CitationTagger

EXTRACTION_MODEL_PATH = model_path = (
    Path(__file__).parent.parent.parent / "outputs/models/extraction/"
)

app = typer.Typer()


@app.command()
def tag(
    input: Path,
    overwrite_ok: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Modify XML files in place without asking permission",
    ),
    copy: bool = typer.Option(
        False,
        "--copy",
        "-c",
        help="Instead of overwriting XML files, edits copies in processed/ dir",
    ),
):
    if overwrite_ok and copy:
        raise typer.BadParameter(
            "Can't authorize editing in place and request editing copies at the same time"
        )
    tagger = CitationTagger(EXTRACTION_MODEL_PATH)
    if copy:
        typer.echo(f"Processing {input} and saving to {input.parent / 'processed'}")
        tagger.process_xml(input, overwrite=False)
    else:
        if overwrite_ok:
            typer.echo(f"Processing {input} in place")
            tagger.process_xml(input, overwrite=True)
        else:
            go_ahead = typer.confirm(f"Overwrite XML file(s) at {input}?")
            if go_ahead:
                typer.echo(f"Processing {input} in place")
                tagger.process_xml(input, overwrite=True)
            else:
                typer.echo("Aborted by user")
                raise typer.Exit(code=1)


@app.command()
def resolve():
    typer.echo("Resolve not implemented yet")


@app.callback()
def setup_cli(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable INFO output"),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Disable non-critical output"
    ),
):
    if verbose and quiet:
        raise typer.BadParameter("Cannot use --verbose and --quiet together")
    if verbose:
        level = logging.INFO
    elif quiet:
        level = logging.CRITICAL
    else:
        level = logging.WARNING

    # Clear existing handler
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )


def main():
    app()


# To enable running app with cli.py directly rather than via __main__.py during development
if __name__ == "__main__":
    main()
