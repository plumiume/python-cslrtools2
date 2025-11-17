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


from __future__ import annotations

from clipar import namespace, mixin
from pathlib import Path

from ....sldataset.app.plugins import Info


@namespace
class FS50Args(mixin.ReprMixin):
    origin: Path
    "Path to the original Fluentsigners50 dataset directory."
    processed: Path
    "Path to the processed Fluentsigners50 dataset directory."

    output: Path
    "Path to the output Zarr dataset directory."
    use_zip: bool = False
    "Whether to use zip store for Zarr dataset."


def processor(args: FS50Args.T):
    import zarr.storage
    from ....sldataset.dataset import dataset_to_zarr
    from ....sldataset.logger import sldataset_logger
    from .main import load

    sldataset_logger.info("Starting FS50 dataset processing")
    sldataset_logger.debug(f"Origin: {args.origin}")
    sldataset_logger.debug(f"Processed: {args.processed}")
    sldataset_logger.debug(f"Output: {args.output}")
    sldataset_logger.debug(f"Use ZIP: {args.use_zip}")

    sldataset_logger.info("Loading dataset from file system...")
    dataset = load(origin=args.origin, processed=args.processed)
    sldataset_logger.info("Loaded dataset successfully")

    if args.use_zip:
        sldataset_logger.info(f"Creating ZIP store at: {args.output}")
        store = zarr.storage.ZipStore(args.output, mode="w")
    else:
        sldataset_logger.info(f"Creating local store at: {args.output}")
        store = zarr.storage.LocalStore(args.output)

    sldataset_logger.info("Converting dataset to Zarr format...")
    group = dataset_to_zarr(
        dataset,
        zarr.create_group(store),
    )
    sldataset_logger.debug(f"zarr store with {len(group)} subgroups/arrays created")
    sldataset_logger.info("Dataset successfully converted to Zarr format")


info: Info[FS50Args.T] = (FS50Args, processor)
