#!/usr/bin/env python3
"""Download MMPose model weights from OpenMMLab.

This script downloads pre-trained MMPose models without requiring openmim.
Models are downloaded directly from download.openmmlab.com and stored in
the assets directory following the cslrtools2 convention.

Usage:
    python download_models.py --model rtmpose-m --task body --resolution 256x192
    python download_models.py --model rtmw-l --task wholebody --resolution 384x288
    python download_models.py --list  # Show available models
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tomllib
from pathlib import Path
from typing import NamedTuple

import requests
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

# Get the assets directory path (src/cslrtools2/assets/mmpose/)
DEFAULT_ASSETS_DIR = Path(__file__).parent.parent.parent.parent / "assets" / "mmpose"

console = Console()


class ModelInfo(NamedTuple):
    """Model information."""

    name: str
    task: str
    resolution: str
    url: str
    md5: str | None = None
    description: str = ""


def load_models() -> dict[str, ModelInfo]:
    """Load model definitions from models.toml.

    Returns:
        Dictionary mapping model IDs to ModelInfo instances
    """
    toml_path = Path(__file__).parent / "models.toml"
    if not toml_path.exists():
        console.print(f"[red]Error: models.toml not found at {toml_path}[/red]")
        sys.exit(1)

    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    models_dict: dict[str, ModelInfo] = {}
    for model_id, model_data in data.get("models", {}).items():
        models_dict[model_id] = ModelInfo(
            name=model_data["name"],
            task=model_data["task"],
            resolution=model_data["resolution"],
            url=model_data["url"],
            md5=model_data.get("md5"),
            description=model_data.get("description", ""),
        )

    return models_dict


# Load models from TOML
MODELS = load_models()


def compute_md5(filepath: Path, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of a file.

    Args:
        filepath: Path to file
        chunk_size: Size of chunks to read

    Returns:
        MD5 hash as hex string
    """
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def download_file(url: str, output_path: Path) -> bool:
    """Download a file with progress bar.

    Args:
        url: URL to download from
        output_path: Path to save file

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Start download
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        # Get file size
        total_size = int(response.headers.get("content-length", 0))

        # Setup progress bar
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Downloading {output_path.name}", total=total_size
            )

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))

        return True

    except requests.exceptions.RequestException as e:
        console.print(f"[red]Error downloading file: {e}[/red]")
        return False
    except OSError as e:
        console.print(f"[red]Error saving file: {e}[/red]")
        return False


def get_model_path(model_info: ModelInfo, assets_dir: Path | None = None) -> Path:
    """Get the output path for a model following cslrtools2 convention.

    Args:
        model_info: Model information
        assets_dir: Assets directory (defaults to DEFAULT_ASSETS_DIR)

    Returns:
        Path to save model
    """
    if assets_dir is None:
        assets_dir = DEFAULT_ASSETS_DIR

    # Pattern: assets/mmpose/{task}/{name}/{resolution}.pth
    model_dir = assets_dir / model_info.task / model_info.name
    filename = f"{model_info.resolution}.pth"
    return model_dir / filename


def list_models(assets_dir: Path | None = None) -> None:
    """List all available models.

    Args:
        assets_dir: Assets directory (defaults to DEFAULT_ASSETS_DIR)
    """
    console.print("\n[bold cyan]Available MMPose Models:[/bold cyan]\n")

    # Group by task
    tasks: dict[str, list[tuple[str, ModelInfo]]] = {}
    for model_id, model_info in MODELS.items():
        if model_info.task not in tasks:
            tasks[model_info.task] = []
        tasks[model_info.task].append((model_id, model_info))

    # Print grouped models
    for task_name, models in sorted(tasks.items()):
        console.print(f"[bold yellow]{task_name.upper()}:[/bold yellow]")
        for model_id, model_info in sorted(models):
            path = get_model_path(model_info, assets_dir)
            exists = path.exists()
            status = "[green]✓[/green]" if exists else "[dim]○[/dim]"
            console.print(
                f"  {status} [bold]{model_id:25s}[/bold] - {model_info.description}"
            )
        console.print()


def download_model(
    model_id: str, force: bool = False, assets_dir: Path | None = None
) -> bool:
    """Download a specific model.

    Args:
        model_id: Model identifier
        force: Force re-download even if file exists
        assets_dir: Assets directory (defaults to DEFAULT_ASSETS_DIR)

    Returns:
        True if successful, False otherwise
    """
    if model_id not in MODELS:
        console.print(f"[red]Error: Model '{model_id}' not found.[/red]")
        console.print("[yellow]Use --list to see available models.[/yellow]")
        return False

    model_info = MODELS[model_id]
    output_path = get_model_path(model_info, assets_dir)

    # Check if already exists
    if output_path.exists() and not force:
        console.print(f"[yellow]Model already exists: {output_path}[/yellow]")
        console.print("[dim]Use --force to re-download.[/dim]")
        return True

    console.print(f"\n[bold cyan]Downloading {model_id}...[/bold cyan]")
    console.print(f"[dim]Task: {model_info.task}[/dim]")
    console.print(f"[dim]Resolution: {model_info.resolution}[/dim]")
    console.print(f"[dim]URL: {model_info.url}[/dim]")
    console.print(f"[dim]Output: {output_path}[/dim]\n")

    # Download
    if not download_file(model_info.url, output_path):
        return False

    # Verify MD5 if provided
    if model_info.md5:
        console.print("\n[cyan]Verifying MD5 checksum...[/cyan]")
        actual_md5 = compute_md5(output_path)
        if actual_md5 == model_info.md5:
            console.print("[green]✓ MD5 checksum verified[/green]")
        else:
            console.print("[red]✗ MD5 checksum mismatch![/red]")
            console.print(f"  Expected: {model_info.md5}")
            console.print(f"  Actual:   {actual_md5}")
            return False

    console.print(f"\n[green]✓ Successfully downloaded {model_id}[/green]")
    console.print(f"[dim]Saved to: {output_path}[/dim]\n")
    return True


def download_all(force: bool = False, assets_dir: Path | None = None) -> None:
    """Download all models.

    Args:
        force: Force re-download even if files exist
        assets_dir: Assets directory (defaults to DEFAULT_ASSETS_DIR)
    """
    console.print("\n[bold cyan]Downloading all models...[/bold cyan]\n")

    success_count = 0
    fail_count = 0

    for model_id in MODELS:
        if download_model(model_id, force=force, assets_dir=assets_dir):
            success_count += 1
        else:
            fail_count += 1
        console.print()

    console.print("[bold]Summary:[/bold]")
    console.print(f"  [green]✓ {success_count} succeeded[/green]")
    if fail_count > 0:
        console.print(f"  [red]✗ {fail_count} failed[/red]")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download MMPose model weights from OpenMMLab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available models
  python download_models.py --list

  # Download a specific model
  python download_models.py --model rtmpose-m-body8

  # Download wholebody model
  python download_models.py --model rtmw-l

  # Download all models
  python download_models.py --all

  # Force re-download
  python download_models.py --model rtmpose-m-body8 --force
        """,
    )

    parser.add_argument(
        "--list", action="store_true", help="List all available models and exit"
    )
    parser.add_argument("--model", type=str, help="Model ID to download")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument(
        "--force", action="store_true", help="Force re-download even if file exists"
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=DEFAULT_ASSETS_DIR,
        help=f"Assets directory (default: {DEFAULT_ASSETS_DIR})",
    )

    args = parser.parse_args()

    # Use provided assets directory
    assets_dir = args.assets_dir

    # List models
    if args.list:
        list_models(assets_dir)
        return 0

    # Download all models
    if args.all:
        download_all(force=args.force, assets_dir=assets_dir)
        return 0

    # Download specific model
    if args.model:
        success = download_model(args.model, force=args.force, assets_dir=assets_dir)
        return 0 if success else 1

    # No action specified
    parser.print_help()
    console.print("\n[yellow]Hint: Use --list to see available models[/yellow]")
    return 1


if __name__ == "__main__":
    sys.exit(main())
