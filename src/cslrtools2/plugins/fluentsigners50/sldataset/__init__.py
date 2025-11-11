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
    from .main import load

    dataset = load(
        origin=args.origin,
        processed=args.processed
    )

    if args.use_zip:
        store = zarr.storage.ZipStore(args.output, mode="w")
    else:
        store = zarr.storage.LocalStore(args.output)

    dataset_to_zarr(
        dataset,
        zarr.create_group(store),
    )

info: Info[FS50Args.T] = (
    FS50Args, processor
)
