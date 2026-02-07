import os
import typing as t
from pathlib import Path

from .. import __semver__

if t.TYPE_CHECKING:
    from contraqctor.contract import Dataset


def dataset(path: os.PathLike, version: str = __semver__) -> "Dataset":
    """
    Loads the dataset for the Aind VR Foraging project from a specified version.

    Args:
        path (os.PathLike): The path to the dataset root directory.
        version (str): The version of the dataset to load. By default, it uses the package version.

    Returns:
        "Dataset": The loaded dataset.
    """
    from ._dataset import make_dataset

    return make_dataset(Path(path), version=version)


def render_dataset(version: str = __semver__) -> str:
    """Renders the dataset as a tree-like structure for visualization."""
    from contraqctor.contract.utils import print_data_stream_tree_html

    return print_data_stream_tree_html(
        dataset(Path("<RootPath>"), version=version), show_missing_indicator=False, show_type=True
    )
