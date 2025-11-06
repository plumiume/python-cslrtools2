
from typing import Any, Callable, Literal, overload, Iterable, Sequence, Hashable, BinaryIO
from pathlib import Path
from contextlib import AbstractContextManager, ExitStack
from os import PathLike
from io import IOBase, BytesIO, BufferedWriter

import numpy as np
from numpy.typing import ArrayLike

from matplotlib import cm
from matplotlib import style
# from matpliblib import interactive
from matplotlib import cbook
from matplotlib.backend_bases import (
    FigureCanvasBase,
    FigureManagerBase,
    MouseEvent
)
from matplotlib.figure import Figure, FigureBase
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib import rcsetup, rcParamsDefault, rcParamsOrig
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.axes import Subplot
from matplotlib.axes._base import _AxesBase # pyright: ignore[reportPrivateUsage]
from matplotlib.backends.registry import BackendFilter, backend_registry
from matplotlib.projections.polar import PolarAxes
from matplotlib.colorizer import ColorizingArtist, Colorizer
from matplotlib import mlab
from matplotlib.scale import get_scale_names

from matplotlib.cm import ColormapRegistry
from matplotlib.colors import Colormap

import matplotlib.axes
import matplotlib.artist
import matplotlib.backend_bases
from matplotlib.axis import Tick
from matplotlib.backend_bases import Event
from matplotlib.cm import ScalarMappable
from matplotlib.contour import ContourSet, QuadContourSet
from matplotlib.collections import (
    Collection,
    FillBetweenPolyCollection,
    LineCollection,
    PolyCollection,
    PathCollection,
    EventCollection,
    QuadMesh,
)
from matplotlib.colorbar import Colorbar
from matplotlib.container import (
    BarContainer,
    ErrorbarContainer,
    StemContainer,
)
from matplotlib.figure import SubFigure
from matplotlib.legend import Legend
from matplotlib.mlab import GaussianKDE
from matplotlib.image import AxesImage, FigureImage
from matplotlib.patches import FancyArrow, StepPatch, Wedge
from matplotlib.quiver import Barbs, Quiver, QuiverKey
from matplotlib.scale import ScaleBase
from matplotlib.typing import (
    ColorType,
    CoordsType,
    HashableList,
    LineStyleType,
    MarkerType,
)
from matplotlib.widgets import SubplotTool

from matplotlib.colors import Normalize
from matplotlib.lines import Line2D, AxLine
from matplotlib.text import Text, Annotation
from matplotlib.patches import Arrow, Circle, Rectangle
from matplotlib.patches import Polygon
from matplotlib.widgets import Button, Slider, Widget
from matplotlib.ticker import (
    TickHelper, Formatter, FixedFormatter, NullFormatter, FuncFormatter,
    FormatStrFormatter, ScalarFormatter, LogFormatter, LogFormatterExponent,
    LogFormatterMathtext, Locator, IndexLocator, FixedLocator, NullLocator,
    LinearLocator, LogLocator, AutoLocator, MultipleLocator, MaxNLocator
)

colormap: ColormapRegistry
color_sequences: dict[str, list[ColorType]]

def install_repl_displayhook() -> None: ...

def uninstall_repl_displayhook() -> None: ...

def findobj(
    o: Artist | None = None,
    match: Callable[[Artist], bool] | type[Artist] | None = None,
    include_self: bool = True,
    ) -> list[Artist]: ...

def switch_backend(newbackend: str) -> None: ...

def new_figure_manager(
    num: int | str,
    *args: object,
    **kwargs: object,
    ) -> FigureCanvasBase: ...

def draw_if_interactive(
    *args: object,
    **kwargs: object
    ) -> None: ...

def show(
    *args: object,
    **kwargs: object
    ) -> None: ...

def isinteractive() -> bool: ...

def ioff() -> ExitStack: ...

def ion() -> ExitStack: ...

def pause(interval: float) -> None: ...

def rc(group: str, **kwargs: object) -> AbstractContextManager[None]: ...

def rc_context(
    rc: dict[str, object] | None = None,
    fname: str | Path | PathLike[str] | None = None,
    ) -> AbstractContextManager[None]: ...

def rcdefaults() -> None: ...

def getp(obj: Artist, property: str | None = None) -> object: ...

def get(obj: Artist, property: str | None = None) -> object: ...

def setp(obj: Artist, *args: object, file: IOBase = ..., **kwargs: object) -> list[object] | None: ...

def xkcd(
    scale: float = 1, length: float = 100, randomness: float = 2
) -> ExitStack: ...

def figure(
    # autoincrement if None, else integer from 1-N
    num: int | str | Figure | SubFigure | None = None,
    # defaults to rc figure.figsize
    figsize: ArrayLike | None = None,
    # defaults to rc figure.dpi
    dpi: float | None = None,
    *,
    # defaults to rc figure.facecolor
    facecolor: ColorType | None = None,
    # defaults to rc figure.edgecolor
    edgecolor: ColorType | None = None,
    frameon: bool = True,
    FigureClass: type[Figure] = Figure,
    clear: bool = False,
    **kwargs: object
) -> Figure: ...

def gcf() -> Figure: ...

def fignum_exists(num: int | str) -> bool: ...

def get_fignums() -> list[int]: ...

def get_figlabels() -> list[Any]: ...

def get_current_fig_manager() -> FigureManagerBase | None: ...

def connect(s: str, func: Callable[[Event], Any]) -> int: ...

def disconnect(cid: int) -> None: ...

def close(fig: None | int | str | Figure | Literal["all"] = None) -> None: ...

def clf() -> None: ...

def draw() -> None: ...

def savefig(
    fname: str | PathLike[str] | IOBase | BytesIO | BufferedWriter,
    *,
    transparent: bool = ...,
    **kwargs: object
    ) -> None: ...

def figlegend(*args: Any, **kwargs: Any) -> Legend: ...

def axes(arg: tuple[float, float, float, float] | None = None, **kwargs: Any) -> Axes: ...

def delaxes(ax: Axes | None = None) -> None: ...

def sca(ax: Axes) -> None: ...

def cla() -> None: ...

def subplot(*args: Any, **kwargs: Any) -> Axes: ...

@overload
def subplots(
    nrows: Literal[1] = ...,
    ncols: Literal[1] = ...,
    *,
    sharex: bool | Literal["none", "all", "row", "col"] = ...,
    sharey: bool | Literal["none", "all", "row", "col"] = ...,
    squeeze: Literal[True] = ...,
    width_ratios: Sequence[float] | None = ...,
    height_ratios: Sequence[float] | None = ...,
    subplot_kw: dict[str, Any] | None = ...,
    gridspec_kw: dict[str, Any] | None = ...,
    **fig_kw: Any
    ) -> tuple[Figure, Axes]: ...

@overload
def subplots(
    nrows: int = ...,
    ncols: int = ...,
    *,
    sharex: bool | Literal["none", "all", "row", "col"] = ...,
    sharey: bool | Literal["none", "all", "row", "col"] = ...,
    squeeze: Literal[False],
    width_ratios: Sequence[float] | None = ...,
    height_ratios: Sequence[float] | None = ...,
    subplot_kw: dict[str, Any] | None = ...,
    gridspec_kw: dict[str, Any] | None = ...,
    **fig_kw: Any
    ) -> tuple[Figure, np.ndarray]: ...

@overload
def subplots(
    nrows: int = ...,
    ncols: int = ...,
    *,
    sharex: bool | Literal["none", "all", "row", "col"] = ...,
    sharey: bool | Literal["none", "all", "row", "col"] = ...,
    squeeze: bool = ...,
    width_ratios: Sequence[float] | None = ...,
    height_ratios: Sequence[float] | None = ...,
    subplot_kw: dict[str, Any] | None = ...,
    gridspec_kw: dict[str, Any] | None = ...,
    **fig_kw: Any
    ) -> tuple[Figure, Any]: ...

@overload
def subplot_mosaic(
    mosaic: str,
    *,
    sharex: bool = ...,
    sharey: bool = ...,
    width_ratios: ArrayLike | None = ...,
    height_ratios: ArrayLike | None = ...,
    empty_sentinel: str = ...,
    subplot_kw: dict[str, Any] | None = ...,
    gridspec_kw: dict[str, Any] | None = ...,
    per_subplot_kw: dict[str | tuple[str, ...], dict[str, Any]] | None = ...,
    **fig_kw: Any
    ) -> tuple[Figure, dict[str, matplotlib.axes.Axes]]: ...

@overload
def subplot_mosaic[_T]( # pyright: ignore[reportOverlappingOverload]
    mosaic: list[HashableList[_T]],
    *,
    sharex: bool = ...,
    sharey: bool = ...,
    width_ratios: ArrayLike | None = ...,
    height_ratios: ArrayLike | None = ...,
    empty_sentinel: _T = ...,
    subplot_kw: dict[str, Any] | None = ...,
    gridspec_kw: dict[str, Any] | None = ...,
    per_subplot_kw: dict[_T | tuple[_T, ...], dict[str, Any]] | None = ...,
    **fig_kw: Any
) -> tuple[Figure, dict[_T, matplotlib.axes.Axes]]: ...


@overload
def subplot_mosaic(
    mosaic: list[HashableList[Hashable]],
    *,
    sharex: bool = ...,
    sharey: bool = ...,
    width_ratios: ArrayLike | None = ...,
    height_ratios: ArrayLike | None = ...,
    empty_sentinel: Any = ...,
    subplot_kw: dict[str, Any] | None = ...,
    gridspec_kw: dict[str, Any] | None = ...,
    per_subplot_kw: dict[Hashable | tuple[Hashable, ...], dict[str, Any]] | None = ...,
    **fig_kw: Any
) -> tuple[Figure, dict[Hashable, matplotlib.axes.Axes]]: ...

def subplot2grid(
    shape: tuple[int, int], loc: tuple[int, int],
    rowspan: int = 1, colspan: int = 1,
    fig: Figure | None = None,
    **kwargs: Any
    ) -> Axes: ...

def twinx(ax: matplotlib.axes.Axes | None = None) -> _AxesBase: ...

def twiny(ax: matplotlib.axes.Axes | None = None) -> _AxesBase: ...

def subplot_tool(targetfig: Figure | None = None) -> SubplotTool | None: ...

def box(on: bool | None = None) -> None: ...

@overload
def xlim(
    left: tuple[float | np.datetime64, float | np.datetime64],
    *,
    emit: bool = ...,
    auto: bool | None = ...,
    xmin: float = ...,
    xmax: float = ...
    ) -> tuple[float, float]: ...
@overload
def xlim(
    left: float | np.datetime64 | None = ...,
    right: float | np.datetime64 | None = ...,
    emit: bool = ...,
    auto: bool | None = ...,
    *,
    xmin: float = ...,
    xmax: float = ...
    ) -> tuple[float, float]: ...

def ylim(
    bottom: float = ...,
    top: float = ...,
    emit: bool = ...,
    auto: bool | None = ...,
    *,
    ymin: float = ...,
    ymax: float = ...
    ) -> tuple[float, float]: ...

def xticks(
    ticks: ArrayLike | None = None,
    labels: Sequence[str] | None = None,
    *,
    minor: bool = False,
    **kwargs: Any
    ) -> tuple[list[Tick] | np.ndarray, list[Text]]: ...

def yticks(
    ticks: ArrayLike | None = None,
    labels: Sequence[str] | None = None,
    *,
    minor: bool = False,
    **kwargs: Any
    ) -> tuple[list[Tick] | np.ndarray, list[Text]]: ...

def rgrids(
    radii: ArrayLike | None = None,
    labels: Sequence[str | Text] | None = None,
    angle: float | None = None,
    fmt: str | None = None,
    **kwargs: Any
    ) -> tuple[list[Line2D], list[Text]]: ...

def thetagrids(
    angles: ArrayLike | None = None,
    labels: Sequence[str | Text] | None = None,
    fmt: str | None = None,
    **kwargs: Any
    ) -> tuple[list[Line2D], list[Text]]: ...

def get_plot_commands() -> list[str]: ...


def colorbar(
    mappable: ScalarMappable | ColorizingArtist | None = None,
    cax: matplotlib.axes.Axes | None = None,
    ax: matplotlib.axes.Axes | Iterable[matplotlib.axes.Axes] | None = None,
    **kwargs: Any
    ) -> Colorbar: ...

def clim(vmin: float | None = None, vmax: float | None = None) -> None: ...

def get_cmap(name: Colormap | str | None = None, lut: int | None = None) -> Colormap: ...

def set_cmap(cmap: Colormap | str) -> None: ...

def imread(
    fname: str | Path | BinaryIO, format: str | None = None
    ) -> np.ndarray: ...

def imsave(
    fname: str | PathLike[str] | BinaryIO, arr: ArrayLike, **kwargs: Any
    ) -> None: ...

def matshow(A: ArrayLike, fignum: None | int = None, **kwargs: Any) -> AxesImage: ...