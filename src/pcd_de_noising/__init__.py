import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "pcd-de-noising"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from .model import MistNet
from .pcd_dataset import PCDDataset, PointCloudDataModule

# make pep8 happy. src: https://stackoverflow.com/a/31079085/6942666
__all__ = ["Mistnet", "PCDDataset", "PointCloudDataModule"]
