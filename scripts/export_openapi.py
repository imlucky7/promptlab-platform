"""Export the OpenAPI specification to a file.

The FastAPI app already serves the live spec at ``/openapi.json`` (and renders
interactive docs at ``/docs`` and ``/redoc``). This script additionally writes a
static spec to disk so it can be published/deployed independently (e.g. to an API
portal, a docs site, or a contract repository) without running the service.

Usage:
    python -m scripts.export_openapi                 # writes openapi.json
    python -m scripts.export_openapi --format yaml   # writes openapi.yaml
    python -m scripts.export_openapi --output docs/openapi.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from app.main import create_app


def generate_openapi() -> dict[str, Any]:
    """Build the OpenAPI document from the FastAPI app.

    Returns:
        The OpenAPI specification as a dictionary.
    """
    app = create_app()
    return app.openapi()


def export(output: Path, fmt: str) -> None:
    """Write the OpenAPI document to ``output`` in the requested format.

    Args:
        output: Destination file path.
        fmt: Either ``"json"`` or ``"yaml"``.
    """
    spec = generate_openapi()
    output.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "yaml":
        output.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    else:
        output.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"OpenAPI spec written to {output} ({fmt})")


def main() -> None:
    """CLI entry point for exporting the OpenAPI spec."""
    parser = argparse.ArgumentParser(description="Export the OpenAPI specification.")
    parser.add_argument(
        "--format",
        choices=["json", "yaml"],
        default="json",
        help="Output format (default: json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: openapi.json or openapi.yaml).",
    )
    args = parser.parse_args()

    output = args.output or Path(f"openapi.{ 'yaml' if args.format == 'yaml' else 'json' }")
    export(output, args.format)


if __name__ == "__main__":
    main()
