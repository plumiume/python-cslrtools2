# Copyright 2025 cslrtools2 contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Download videos from Pexels."""

from __future__ import annotations

import re
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

from .config import PEXELS_VIDEOS, VIDEOS_DIR


def extract_video_download_url(html_content: str, resolution: str) -> str | None:
    """Extract video download URL from Pexels HTML.

    Args:
        html_content: HTML content of Pexels video page
        resolution: Resolution string (e.g., 'hd_2048_1080_30fps')

    Returns:
        Download URL or None if not found
    """
    # Pattern to find video file URLs
    pattern = (
        rf"https://videos\.pexels\.com/video-files/\d+/\d+-{re.escape(resolution)}\.mp4"
    )
    match = re.search(pattern, html_content)
    return match.group(0) if match else None


def download_pexels_video(
    video_id: str,
    name: str,
    resolution: str,
    output_dir: Path = VIDEOS_DIR,
) -> Path | None:
    """Download a single Pexels video.

    Args:
        video_id: Pexels video ID
        name: Output filename (without extension)
        resolution: Resolution string
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

    output_path = output_dir / f"{name}.mp4"
    if output_path.exists():
        print(f"✓ Already exists: {output_path.name}")
        return output_path

    # Fetch page to extract download URL
    page_url = f"https://www.pexels.com/video/{video_id}/"
    print(f"Fetching page: {page_url}")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(page_url, headers=headers, timeout=10)
        response.raise_for_status()

        download_url = extract_video_download_url(response.text, resolution)
        if not download_url:
            print(f"ERROR: Could not find download URL for resolution {resolution}")
            return None

        print(f"Downloading: {download_url}")
        video_response = requests.get(
            download_url, headers=headers, timeout=30, stream=True
        )
        video_response.raise_for_status()

        # Check size
        content_length = int(video_response.headers.get("content-length", 0))
        size_mb = content_length / (1024 * 1024)
        print(f"Size: {size_mb:.2f} MB")

        # Download
        with output_path.open("wb") as f:
            for chunk in video_response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"✓ Downloaded: {output_path.name}")
        return output_path

    except requests.RequestException as e:
        print(f"ERROR: Failed to download {name}: {e}")
        return None


def download_all_pexels_videos() -> list[Path]:
    """Download all configured Pexels videos.

    Returns:
        List of downloaded file paths
    """
    print("\n=== Downloading Pexels Videos ===\n")

    downloaded = []
    for video_info in PEXELS_VIDEOS:
        result = download_pexels_video(
            video_id=video_info["video_id"],
            name=video_info["name"],
            resolution=video_info["resolution"],
        )
        if result:
            downloaded.append(result)

    print(f"\nDownloaded {len(downloaded)}/{len(PEXELS_VIDEOS)} videos")
    return downloaded


if __name__ == "__main__":
    download_all_pexels_videos()
