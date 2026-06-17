"""Command-line interface (Typer): train / forecast / evaluate.

A thin orchestration layer over the compute library. Typer (and TensorFlow, on
the train path) are imported LAZILY inside :func:`build_app` / the command
bodies, so importing :mod:`lstmforecast.cli` registers no commands and does no
I/O. The module-level ``app`` is a lazily-built singleton consumed by the
``lstm-forecast`` console-script entry point.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import typer


def build_app() -> typer.Typer:
    """Construct and return the Typer application.

    Registers ``train``, ``forecast``, and ``evaluate`` on a fresh
    ``typer.Typer``. Typer is imported lazily inside this function so importing
    :mod:`lstmforecast.cli` does not import Typer or register any commands.

    Returns
    -------
    typer.Typer
        The configured Typer application.
    """
    raise NotImplementedError


def main() -> None:
    """Console-script entry point: build the app and invoke it.

    Wired to the ``lstm-forecast`` script in ``pyproject.toml``. Builds the Typer
    app via :func:`build_app` and runs it.
    """
    raise NotImplementedError


if __name__ == "__main__":  # pragma: no cover - manual invocation only
    main()
