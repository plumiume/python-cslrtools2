"""Download datasets from Zenodo."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

from .config import DATASETS_DIR, ZENODO_DATASETS


def download_zenodo_file(
    record_id: str,
    filename: str,
    output_dir: Path = DATASETS_DIR,
) -> Path | None:
    """Download a file from Zenodo.

    Args:
        record_id: Zenodo record ID
        filename: Name of file to download
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

    output_path = output_dir / filename
    if output_path.exists():
        print(f"✓ Already exists: {output_path.name}")
        return output_path

    # Zenodo API URL for file download
    api_url = f"https://zenodo.org/api/records/{record_id}"
    print(f"Fetching Zenodo record: {record_id}")

    try:
        # Get record metadata to find file download URL
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        record_data = response.json()

        # Find the file in the record
        files = record_data.get("files", [])
        file_info = next((f for f in files if f["key"] == filename), None)

        if not file_info:
            print(f"ERROR: File {filename} not found in record {record_id}")
            return None

        download_url = file_info["links"]["self"]
        size_mb = file_info["size"] / (1024 * 1024)
        print(f"Downloading: {filename} ({size_mb:.2f} MB)")

        # Download the file
        file_response = requests.get(download_url, timeout=60, stream=True)
        file_response.raise_for_status()

        with output_path.open("wb") as f:
            for chunk in file_response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"✓ Downloaded: {output_path.name}")
        return output_path

    except requests.RequestException as e:
        print(f"ERROR: Failed to download {filename}: {e}")
        return None
    except (KeyError, ValueError) as e:
        print(f"ERROR: Failed to parse Zenodo response: {e}")
        return None


def download_zenodo_dataset(dataset_info: dict[str, Any]) -> Path | None:
    """Download a Zenodo dataset.

    Args:
        dataset_info: Dataset information dict

    Returns:
        Path to downloaded file or None if failed
    """
    print(f"\n--- {dataset_info['name']} ({dataset_info['license']}) ---")
    print(f"Description: {dataset_info['description']}\n")

    return download_zenodo_file(
        record_id=dataset_info["record_id"],
        filename=dataset_info["file"],
    )


def download_all_zenodo_datasets() -> list[Path]:
    """Download all configured Zenodo datasets.

    Returns:
        List of downloaded file paths
    """
    print("\n=== Downloading Zenodo Datasets ===\n")

    downloaded: list[Path] = []
    for dataset_info in ZENODO_DATASETS:
        result = download_zenodo_dataset(dataset_info)
        if result:
            downloaded.append(result)

    print(f"\nDownloaded {len(downloaded)}/{len(ZENODO_DATASETS)} datasets")
    return downloaded


if __name__ == "__main__":
    download_all_zenodo_datasets()
