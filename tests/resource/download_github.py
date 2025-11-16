"""Download landmark data from GitHub repositories."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

from .config import GITHUB_REPOS, LANDMARKS_DIR


def download_github_file(
    owner: str,
    repo: str,
    file_path: str,
    output_dir: Path = LANDMARKS_DIR,
) -> Path | None:
    """Download a single file from GitHub repository.

    Args:
        owner: Repository owner
        repo: Repository name
        file_path: Path to file in repository
        output_dir: Output directory

    Returns:
        Path to downloaded file or None if failed
    """
    if requests is None:
        print(
            "ERROR: requests library not installed. Install with: "
            "uv pip install requests"
        )
        return None

    # Create subdirectory for this repo
    repo_dir = output_dir / repo
    repo_dir.mkdir(parents=True, exist_ok=True)

    # Preserve directory structure
    file_name = Path(file_path).name
    output_path = repo_dir / file_name

    if output_path.exists():
        print(f"✓ Already exists: {output_path.relative_to(LANDMARKS_DIR)}")
        return output_path

    # GitHub raw content URL
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{file_path}"
    print(f"Downloading: {file_path}")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        output_path.write_bytes(response.content)
        size_kb = len(response.content) / 1024
        print(f"✓ Downloaded: {output_path.name} ({size_kb:.1f} KB)")
        return output_path

    except requests.RequestException as e:
        print(f"ERROR: Failed to download {file_path}: {e}")
        return None


def download_github_repo_files(repo_info: dict[str, Any]) -> list[Path]:
    """Download all files from a GitHub repository.

    Args:
        repo_info: Repository information dict

    Returns:
        List of downloaded file paths
    """
    print(
        f"\n--- {repo_info['owner']}/{repo_info['repo']} ({repo_info['license']}) ---\n"
    )

    downloaded = []
    for file_path in repo_info["files"]:
        result = download_github_file(
            owner=repo_info["owner"],
            repo=repo_info["repo"],
            file_path=file_path,
        )
        if result:
            downloaded.append(result)

    return downloaded


def download_all_github_landmarks() -> list[Path]:
    """Download all configured GitHub landmark data.

    Returns:
        List of downloaded file paths
    """
    print("\n=== Downloading GitHub Landmark Data ===\n")

    all_downloaded = []
    for repo_info in GITHUB_REPOS:
        downloaded = download_github_repo_files(repo_info)
        all_downloaded.extend(downloaded)

    total_files = sum(len(repo["files"]) for repo in GITHUB_REPOS)
    print(f"\nDownloaded {len(all_downloaded)}/{total_files} files")
    return all_downloaded


if __name__ == "__main__":
    download_all_github_landmarks()
