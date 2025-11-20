
from typing import (
    Any, Callable, Literal, overload,
    Iterable, Sequence, Hashable, BinaryIO
)
from pathlib import Path
from contextlib import AbstractContextManager, ExitStack
from os import PathLike
from io import IOBase, BytesIO, BufferedWriter
import datetime

import numpy as np
from numpy.typing import ArrayLike
import PIL.Image

from matplotlib import cm
from matplotlib import style
# from matpliblib import interactive
from matplotlib import cbook
from matplotlib.backend_bases import (
    FigureCanvasBase,
    FigureManagerBase,
    MouseButton
)
from matplotlib.figure import Figure, FigureBase
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib import rcsetup, rcParamsDefault, rcParamsOrig
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.axes import Subplot
from matplotlib.axes._base import _AxesBase  # pyright: ignore[reportPrivateUsage]
from matplotlib.backends.registry import BackendFilter, backend_registry
from matplotlib.projections.polar import PolarAxes
from matplotlib.colorizer import ColorizingArtist, Colorizer
from matplotlib import mlab

from matplotlib.cm import ColormapRegistry
from matplotlib.colors import Colormap

import matplotlib.axes
import matplotlib.artist # pyright: ignore[reportUnusedImport] # noqa: #261
import matplotlib.backend_bases # pyright: ignore[reportUnusedImport] # noqa: #261
from matplotlib.axis import Tick
from matplotlib.backend_bases import Event
from matplotlib.cm import ScalarMappable
from matplotlib.contour import ContourSet, QuadContourSet
from matplotlib.collections import (
    Collection,
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
from matplotlib.lines import Line2D
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

from matplotlib import rcParams as rcParams
from matplotlib import get_backend as get_backend

__all__ = [
    # Module-level variables
    "colormap",
    "color_sequences",
    # Re-exported from matplotlib
    "rcParams",
    "get_backend",
    "cm",
    "style",
    "cbook",
    "mlab",
    "get_scale_names",
    # Re-exported types and classes
    "Figure",
    "FigureBase",
    "GridSpec",
    "SubplotSpec",
    "rcsetup",
    "rcParamsDefault",
    "rcParamsOrig",
    "Artist",
    "Axes",
    "Subplot",
    "BackendFilter",
    "backend_registry",
    "PolarAxes",
    "Colorizer",
    "ColorizingArtist",
    "Normalize",
    "Line2D",
    "Text",
    "Annotation",
    "Arrow",
    "Circle",
    "Rectangle",
    "Polygon",
    "Button",
    "Slider",
    "Widget",
    # Ticker classes
    "TickHelper",
    "Formatter",
    "FixedFormatter",
    "NullFormatter",
    "FuncFormatter",
    "FormatStrFormatter",
    "ScalarFormatter",
    "LogFormatter",
    "LogFormatterExponent",
    "LogFormatterMathtext",
    "Locator",
    "IndexLocator",
    "FixedLocator",
    "NullLocator",
    "LinearLocator",
    "LogLocator",
    "AutoLocator",
    "MultipleLocator",
    "MaxNLocator",
    # Module imports for type checking
    "matplotlib",
    # Figure management
    "figure",
    "gcf",
    "fignum_exists",
    "get_fignums",
    "get_figlabels",
    "get_current_fig_manager",
    "connect",
    "disconnect",
    "close",
    "clf",
    "draw",
    "savefig",
    "figlegend",
    "figimage",
    "figtext",
    # Axes management
    "axes",
    "delaxes",
    "sca",
    "gca",
    "cla",
    "subplot",
    "subplots",
    "subplot_mosaic",
    "subplot2grid",
    "twinx",
    "twiny",
    "subplot_tool",
    "box",
    # Axis functions
    "xlim",
    "ylim",
    "xticks",
    "yticks",
    "rgrids",
    "thetagrids",
    "xlabel",
    "ylabel",
    "title",
    "xscale",
    "yscale",
    # Grid and ticks
    "grid",
    "minorticks_on",
    "minorticks_off",
    "tick_params",
    "ticklabel_format",
    "locator_params",
    # Plotting functions
    "plot",
    "plot_date",
    "scatter",
    "bar",
    "barh",
    "bar_label",
    "stem",
    "step",
    "fill",
    "fill_between",
    "fill_betweenx",
    "stackplot",
    "stairs",
    "errorbar",
    "boxplot",
    "violinplot",
    "eventplot",
    "hist",
    "hist2d",
    "pie",
    "polar",
    "loglog",
    "semilogx",
    "semilogy",
    "acorr",
    "xcorr",
    "ecdf",
    # Contour and mesh plots
    "contour",
    "contourf",
    "tricontour",
    "tricontourf",
    "tripcolor",
    "triplot",
    # Image and color plots
    "imshow",
    "matshow",
    "pcolor",
    "pcolormesh",
    "spy",
    "hexbin",
    # Vector field plots
    "quiver",
    "quiverkey",
    "barbs",
    "streamplot",
    # Spectrum plots
    "magnitude_spectrum",
    "angle_spectrum",
    "phase_spectrum",
    "cohere",
    "csd",
    "psd",
    "specgram",
    # Annotations and shapes
    "annotate",
    "arrow",
    "axhline",
    "axhspan",
    "axvline",
    "axvspan",
    "axline",
    "text",
    "table",
    # Lines and markers
    "hlines",
    "vlines",
    # Colorbar and colormap
    "colorbar",
    "clim",
    "get_cmap",
    "set_cmap",
    "sci",
    "gci",
    # Colormap setter functions
    "autumn",
    "bone",
    "cool",
    "copper",
    "flag",
    "gray",
    "hot",
    "hsv",
    "jet",
    "pink",
    "prism",
    "spring",
    "summer",
    "winter",
    "magma",
    "inferno",
    "plasma",
    "viridis",
    "nipy_spectral",
    # Legend and labels
    "legend",
    "clabel",
    # Axis and view control
    "axis",
    "autoscale",
    "margins",
    # Image I/O
    "imread",
    "imsave",
    # Interactive control
    "install_repl_displayhook",
    "uninstall_repl_displayhook",
    "isinteractive",
    "ioff",
    "ion",
    "pause",
    "show",
    "draw_if_interactive",
    # Backend control
    "switch_backend",
    "new_figure_manager",
    # Configuration
    "rc",
    "rc_context",
    "rcdefaults",
    "xkcd",
    # Object properties
    "getp",
    "get",
    "setp",
    "findobj",
    # Figure layout
    "subplots_adjust",
    "suptitle",
    "tight_layout",
    # User interaction
    "ginput",
    "waitforbuttonpress",
    # Miscellaneous
    "broken_barh",
]


def get_scale_names() -> list[str]: ...


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


def setp(
    obj: Artist,
    *args: object,
    file: IOBase = ...,
    **kwargs: object
) -> list[object] | None: ...


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


def axes(
    arg: tuple[float, float, float, float] | None = None,
    **kwargs: Any
) -> Axes: ...


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
def subplot_mosaic[_T](  # pyright: ignore[reportOverlappingOverload]
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


def get_cmap(
    name: Colormap | str | None = None,
    lut: int | None = None
) -> Colormap: ...


def set_cmap(cmap: Colormap | str) -> None: ...


def imread(
    fname: str | Path | BinaryIO, format: str | None = None
    ) -> np.ndarray: ...


def imsave(
    fname: str | PathLike[str] | BinaryIO, arr: ArrayLike, **kwargs: Any
    ) -> None: ...


def matshow(A: ArrayLike, fignum: None | int = None, **kwargs: Any) -> AxesImage: ...


def polar(*args: Any, **kwargs: Any) -> list[Line2D]: ...


def figimage(
    X: ArrayLike,
    xo: int = 0,
    yo: int = 0,
    alpha: float | None = None,
    norm: str | Normalize | None = None,
    cmap: str | Colormap | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    origin: Literal["upper", "lower"] | None = None,
    resize: bool = False,
    *,
    colorizer: Colorizer | None = None,
    **kwargs: Any,
) -> FigureImage: ...


def figtext(
    x: float, y: float, s: str, fontdict: dict[str, Any] | None = None, **kwargs: Any
    ) -> Text: ...


def gca() -> Axes: ...


def gci() -> ColorizingArtist | None: ...


def ginput(
    n: int = 1,
    timeout: float = 30,
    show_clicks: bool = True,
    mouse_add: MouseButton = MouseButton.LEFT,
    mouse_pop: MouseButton = MouseButton.RIGHT,
    mouse_stop: MouseButton = MouseButton.MIDDLE,
    ) -> list[tuple[int, int]]: ...


def subplots_adjust(
    left: float | None = None,
    bottom: float | None = None,
    right: float | None = None,
    top: float | None = None,
    wspace: float | None = None,
    hspace: float | None = None,
    ) -> None: ...


def suptitle(t: str, **kwargs: Any) -> Text: ...


def tight_layout(
    *,
    pad: float = 1.08,
    h_pad: float | None = None,
    w_pad: float | None = None,
    rect: tuple[float, float, float, float] | None = None,
    ) -> None: ...


def waitforbuttonpress(timeout: float = -1) -> None | bool: ...


def acorr(
    x: ArrayLike, *, data: Any = None, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray, LineCollection | Line2D, Line2D | None]: ...


def angle_spectrum(
    x: ArrayLike,
    Fs: float | None = None,
    Fc: int | None = None,
    window: Callable[[ArrayLike], ArrayLike] | ArrayLike | None = None,
    pad_to: int | None = None,
    sides: Literal["default", "onesided", "twosided"] | None = None,
    *,
    data: Any = None,
    **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, Line2D]: ...


def annotate(
    text: str,
    xy: tuple[float, float],
    xytext: tuple[float, float] | None = None,
    xycoords: CoordsType = "data",
    textcoords: CoordsType | None = None,
    arrowprops: dict[str, Any] | None = None,
    annotation_clip: bool | None = None,
    **kwargs: Any,
    ) -> Annotation: ...


def arrow(x: float, y: float, dx: float, dy: float, **kwargs: Any) -> FancyArrow: ...


def autoscale(
    enable: bool = True,
    axis: Literal["both", "x", "y"] = "both",
    tight: bool | None = None,
    ) -> None: ...


def axhline(
    y: float = 0,
    xmin: float = 0,
    xmax: float = 1,
    **kwargs: Any
) -> Line2D: ...


def axhspan(
    ymin: float, ymax: float, xmin: float = 0, xmax: float = 1, **kwargs: Any
    ) -> Rectangle: ...


def axis(
    arg: tuple[float, float, float, float] | bool | str | None = None,
    /,
    *,
    emit: bool = True,
    **kwargs: Any,
    ) -> tuple[float, float, float, float]: ...


def axline(
    xy1: tuple[float, float],
    xy2: tuple[float, float] | None = None,
    *,
    slope: float | None = None,
    **kwargs: Any,
    ) -> Line2D: ...


def axvline(
    x: float = 0,
    ymin: float = 0,
    ymax: float = 1,
    **kwargs: Any
) -> Line2D: ...


def axvspan(
    xmin: float, xmax: float, ymin: float = 0, ymax: float = 1, **kwargs: Any
    ) -> Rectangle: ...


def bar(
    x: float | ArrayLike,
    height: float | ArrayLike,
    width: float | ArrayLike = 0.8,
    bottom: float | ArrayLike | None = None,
    *,
    align: Literal["center", "edge"] = "center",
    data: Any = None,
    **kwargs: Any,
    ) -> BarContainer: ...


def barbs(*args: Any, data: Any = None, **kwargs: Any) -> Barbs: ...


def barh(
    y: float | ArrayLike,
    width: float | ArrayLike,
    height: float | ArrayLike = 0.8,
    left: float | ArrayLike | None = None,
    *,
    align: Literal["center", "edge"] = "center",
    data: Any = None,
    **kwargs: Any,
    ) -> BarContainer: ...


def bar_label(
    container: BarContainer,
    labels: ArrayLike | None = None,
    *,
    fmt: str | Callable[[float], str] = "%g",
    label_type: Literal["center", "edge"] = "edge",
    padding: float = 0,
    **kwargs: Any,
    ) -> list[Annotation]: ...


def boxplot(
    x: ArrayLike | Sequence[ArrayLike],
    notch: bool | None = None,
    sym: str | None = None,
    vert: bool | None = None,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    whis: float | tuple[float, float] | None = None,
    positions: ArrayLike | None = None,
    widths: float | ArrayLike | None = None,
    patch_artist: bool | None = None,
    bootstrap: int | None = None,
    usermedians: ArrayLike | None = None,
    conf_intervals: ArrayLike | None = None,
    meanline: bool | None = None,
    showmeans: bool | None = None,
    showcaps: bool | None = None,
    showbox: bool | None = None,
    showfliers: bool | None = None,
    boxprops: dict[str, Any] | None = None,
    tick_labels: Sequence[str] | None = None,
    flierprops: dict[str, Any] | None = None,
    medianprops: dict[str, Any] | None = None,
    meanprops: dict[str, Any] | None = None,
    capprops: dict[str, Any] | None = None,
    whiskerprops: dict[str, Any] | None = None,
    manage_ticks: bool = True,
    autorange: bool = False,
    zorder: float | None = None,
    capwidths: float | ArrayLike | None = None,
    label: Sequence[str] | None = None,
    *,
    data: Any = None,
    ) -> dict[str, Any]: ...


def broken_barh(
    xranges: Sequence[tuple[float, float]],
    yrange: tuple[float, float],
    *,
    data: Any = None,
    **kwargs: Any,
    ) -> PolyCollection: ...


def clabel(
    CS: ContourSet,
    levels: ArrayLike | None = None,
    **kwargs: Any
) -> list[Text]: ...


def cohere(
    x: ArrayLike,
    y: ArrayLike,
    NFFT: int = 256,
    Fs: float = 2,
    Fc: int = 0,
    detrend: (
        Literal["none", "mean", "linear"] | Callable[[ArrayLike], ArrayLike]
    ) = ...,
    window: Callable[[ArrayLike], ArrayLike] | ArrayLike = ...,
    noverlap: int = 0,
    pad_to: int | None = None,
    sides: Literal["default", "onesided", "twosided"] = "default",
    scale_by_freq: bool | None = None,
    *,
    data: Any = None,
    **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]: ...


def contour(*args: Any, data: Any = None, **kwargs: Any) -> QuadContourSet: ...


def contourf(*args: Any, data: Any = None, **kwargs: Any) -> QuadContourSet: ...


def csd(
    x: ArrayLike,
    y: ArrayLike,
    NFFT: int | None = None,
    Fs: float | None = None,
    Fc: int | None = None,
    detrend: (
        Literal["none", "mean", "linear"] | Callable[[ArrayLike], ArrayLike] | None
    ) = None,
    window: Callable[[ArrayLike], ArrayLike] | ArrayLike | None = None,
    noverlap: int | None = None,
    pad_to: int | None = None,
    sides: Literal["default", "onesided", "twosided"] | None = None,
    scale_by_freq: bool | None = None,
    return_line: bool | None = None,
    *,
    data: Any = None,
    **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, Line2D]: ...


def ecdf(
    x: ArrayLike,
    weights: ArrayLike | None = None,
    *,
    complementary: bool = False,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    compress: bool = False,
    data: Any = None,
    **kwargs: Any,
) -> Line2D: ...


def errorbar(
    x: float | ArrayLike,
    y: float | ArrayLike,
    yerr: float | ArrayLike | None = None,
    xerr: float | ArrayLike | None = None,
    fmt: str = "",
    ecolor: ColorType | None = None,
    elinewidth: float | None = None,
    capsize: float | None = None,
    barsabove: bool = False,
    lolims: bool | ArrayLike = False,
    uplims: bool | ArrayLike = False,
    xlolims: bool | ArrayLike = False,
    xuplims: bool | ArrayLike = False,
    errorevery: int | tuple[int, int] = 1,
    capthick: float | None = None,
    *,
    data: Any = None,
    **kwargs: Any,
) -> ErrorbarContainer: ...


def eventplot(
    positions: ArrayLike | Sequence[ArrayLike],
    orientation: Literal["horizontal", "vertical"] = "horizontal",
    lineoffsets: float | Sequence[float] = 1,
    linelengths: float | Sequence[float] = 1,
    linewidths: float | Sequence[float] | None = None,
    colors: ColorType | Sequence[ColorType] | None = None,
    alpha: float | Sequence[float] | None = None,
    linestyles: LineStyleType | Sequence[LineStyleType] = "solid",
    *,
    data: Any = None,
    **kwargs: Any,
) -> EventCollection: ...


def fill(*args: Any, data: Any = None, **kwargs: Any) -> list[Polygon]: ...


def fill_between(
    x: ArrayLike,
    y1: ArrayLike | float,
    y2: ArrayLike | float = 0,
    where: Sequence[bool] | None = None,
    interpolate: bool = False,
    step: Literal["pre", "post", "mid"] | None = None,
    *,
    data: Any = None,
    **kwargs: Any,
) -> PolyCollection: ...


def fill_betweenx(
    y: ArrayLike,
    x1: ArrayLike | float,
    x2: ArrayLike | float = 0,
    where: Sequence[bool] | None = None,
    step: Literal["pre", "post", "mid"] | None = None,
    interpolate: bool = False,
    *,
    data: Any = None,
    **kwargs: Any,
) -> PolyCollection: ...


def grid(
    visible: bool | None = None,
    which: Literal["major", "minor", "both"] = "major",
    axis: Literal["both", "x", "y"] = "both",
    **kwargs: Any,
) -> None: ...


def hexbin(
    x: ArrayLike,
    y: ArrayLike,
    C: ArrayLike | None = None,
    gridsize: int | tuple[int, int] = 100,
    bins: Literal["log"] | int | Sequence[float] | None = None,
    xscale: Literal["linear", "log"] = "linear",
    yscale: Literal["linear", "log"] = "linear",
    extent: tuple[float, float, float, float] | None = None,
    cmap: str | Colormap | None = None,
    norm: str | Normalize | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: float | None = None,
    linewidths: float | None = None,
    edgecolors: Literal["face", "none"] | ColorType = "face",
    reduce_C_function: Callable[[np.ndarray | list[float]], float] = ...,
    mincnt: int | None = None,
    marginals: bool = False,
    colorizer: Colorizer | None = None,
    *,
    data: Any = None,
    **kwargs: Any,
) -> PolyCollection: ...


def hist(
    x: ArrayLike | Sequence[ArrayLike],
    bins: int | Sequence[float] | str | None = None,
    range: tuple[float, float] | None = None,
    density: bool = False,
    weights: ArrayLike | None = None,
    cumulative: bool | float = False,
    bottom: ArrayLike | float | None = None,
    histtype: Literal["bar", "barstacked", "step", "stepfilled"] = "bar",
    align: Literal["left", "mid", "right"] = "mid",
    orientation: Literal["vertical", "horizontal"] = "vertical",
    rwidth: float | None = None,
    log: bool = False,
    color: ColorType | Sequence[ColorType] | None = None,
    label: str | Sequence[str] | None = None,
    stacked: bool = False,
    *,
    data: Any = None,
    **kwargs: Any,
) -> tuple[
    np.ndarray | list[np.ndarray],
    np.ndarray,
    BarContainer | Polygon | list[BarContainer | Polygon],
]: ...


def stairs(
    values: ArrayLike,
    edges: ArrayLike | None = None,
    *,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    baseline: float | ArrayLike | None = 0,
    fill: bool = False,
    data: Any = None,
    **kwargs: Any,
) -> StepPatch: ...


def hist2d(
    x: ArrayLike,
    y: ArrayLike,
    bins: None | int | tuple[int, int] | ArrayLike | tuple[ArrayLike, ArrayLike] = 10,
    range: ArrayLike | None = None,
    density: bool = False,
    weights: ArrayLike | None = None,
    cmin: float | None = None,
    cmax: float | None = None,
    *,
    data: Any = None,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, QuadMesh]: ...


def hlines(
    y: float | ArrayLike,
    xmin: float | ArrayLike,
    xmax: float | ArrayLike,
    colors: ColorType | Sequence[ColorType] | None = None,
    linestyles: LineStyleType = "solid",
    label: str = "",
    *,
    data: Any = None,
    **kwargs: Any,
) -> LineCollection: ...


def imshow(
    X: ArrayLike | PIL.Image.Image,
    cmap: str | Colormap | None = None,
    norm: str | Normalize | None = None,
    *,
    aspect: Literal["equal", "auto"] | float | None = None,
    interpolation: str | None = None,
    alpha: float | ArrayLike | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    colorizer: Colorizer | None = None,
    origin: Literal["upper", "lower"] | None = None,
    extent: tuple[float, float, float, float] | None = None,
    interpolation_stage: Literal["data", "rgba", "auto"] | None = None,
    filternorm: bool = True,
    filterrad: float = 4.0,
    resample: bool | None = None,
    url: str | None = None,
    data: Any = None,
    **kwargs: Any,
) -> AxesImage: ...


def legend(*args: Any, **kwargs: Any) -> Legend: ...


def locator_params(
    axis: Literal["both", "x", "y"] = "both", tight: bool | None = None, **kwargs: Any
) -> None: ...


def loglog(*args: Any, **kwargs: Any) -> list[Line2D]: ...


def magnitude_spectrum(
    x: ArrayLike,
    Fs: float | None = None,
    Fc: int | None = None,
    window: Callable[[ArrayLike], ArrayLike] | ArrayLike | None = None,
    pad_to: int | None = None,
    sides: Literal["default", "onesided", "twosided"] | None = None,
    scale: Literal["default", "linear", "dB"] | None = None,
    *,
    data: Any = None,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, Line2D]: ...


def margins(
    *margins: float,
    x: float | None = None,
    y: float | None = None,
    tight: bool | None = True,
) -> tuple[float, float] | None: ...


def minorticks_off() -> None: ...


def minorticks_on() -> None: ...


def pcolor(
    *args: ArrayLike,
    shading: Literal["flat", "nearest", "auto"] | None = None,
    alpha: float | None = None,
    norm: str | Normalize | None = None,
    cmap: str | Colormap | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    colorizer: Colorizer | None = None,
    data: Any = None,
    **kwargs: Any,
) -> Collection: ...


def pcolormesh(
    *args: ArrayLike,
    alpha: float | None = None,
    norm: str | Normalize | None = None,
    cmap: str | Colormap | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    colorizer: Colorizer | None = None,
    shading: Literal["flat", "nearest", "gouraud", "auto"] | None = None,
    antialiased: bool = False,
    data: Any = None,
    **kwargs: Any,
) -> QuadMesh: ...


def phase_spectrum(
    x: ArrayLike,
    Fs: float | None = None,
    Fc: int | None = None,
    window: Callable[[ArrayLike], ArrayLike] | ArrayLike | None = None,
    pad_to: int | None = None,
    sides: Literal["default", "onesided", "twosided"] | None = None,
    *,
    data: Any = None,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, Line2D]: ...


def pie(
    x: ArrayLike,
    explode: ArrayLike | None = None,
    labels: Sequence[str] | None = None,
    colors: ColorType | Sequence[ColorType] | None = None,
    autopct: str | Callable[[float], str] | None = None,
    pctdistance: float = 0.6,
    shadow: bool = False,
    labeldistance: float | None = 1.1,
    startangle: float = 0,
    radius: float = 1,
    counterclock: bool = True,
    wedgeprops: dict[str, Any] | None = None,
    textprops: dict[str, Any] | None = None,
    center: tuple[float, float] = (0, 0),
    frame: bool = False,
    rotatelabels: bool = False,
    *,
    normalize: bool = True,
    hatch: str | Sequence[str] | None = None,
    data: Any = None,
) -> tuple[list[Wedge], list[Text]] | tuple[list[Wedge], list[Text], list[Text]]: ...


def plot(
    *args: float | ArrayLike | str,
    scalex: bool = True,
    scaley: bool = True,
    data: Any = None,
    **kwargs: Any,
) -> list[Line2D]: ...


def plot_date(
    x: ArrayLike,
    y: ArrayLike,
    fmt: str = "o",
    tz: str | datetime.tzinfo | None = None,
    xdate: bool = True,
    ydate: bool = False,
    *,
    data: Any = None,
    **kwargs: Any,
) -> list[Line2D]: ...


def psd(
    x: ArrayLike,
    NFFT: int | None = None,
    Fs: float | None = None,
    Fc: int | None = None,
    detrend: (
        Literal["none", "mean", "linear"] | Callable[[ArrayLike], ArrayLike] | None
    ) = None,
    window: Callable[[ArrayLike], ArrayLike] | ArrayLike | None = None,
    noverlap: int | None = None,
    pad_to: int | None = None,
    sides: Literal["default", "onesided", "twosided"] | None = None,
    scale_by_freq: bool | None = None,
    return_line: bool | None = None,
    *,
    data: Any = None,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, Line2D]: ...


def quiver(*args: Any, data: Any = None, **kwargs: Any) -> Quiver: ...


def quiverkey(
    Q: Quiver, X: float, Y: float, U: float, label: str, **kwargs: Any
) -> QuiverKey: ...


def scatter(
    x: float | ArrayLike,
    y: float | ArrayLike,
    s: float | ArrayLike | None = None,
    c: ArrayLike | Sequence[ColorType] | ColorType | None = None,
    marker: MarkerType | None = None,
    cmap: str | Colormap | None = None,
    norm: str | Normalize | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: float | None = None,
    linewidths: float | Sequence[float] | None = None,
    *,
    edgecolors: Literal["face", "none"] | ColorType | Sequence[ColorType] | None = None,
    colorizer: Colorizer | None = None,
    plotnonfinite: bool = False,
    data: Any = None,
    **kwargs: Any,
) -> PathCollection: ...


def semilogx(*args: Any, **kwargs: Any) -> list[Line2D]: ...


def semilogy(*args: Any, **kwargs: Any) -> list[Line2D]: ...


def specgram(
    x: ArrayLike,
    NFFT: int | None = None,
    Fs: float | None = None,
    Fc: int | None = None,
    detrend: (
        Literal["none", "mean", "linear"] | Callable[[ArrayLike], ArrayLike] | None
    ) = None,
    window: Callable[[ArrayLike], ArrayLike] | ArrayLike | None = None,
    noverlap: int | None = None,
    cmap: str | Colormap | None = None,
    xextent: tuple[float, float] | None = None,
    pad_to: int | None = None,
    sides: Literal["default", "onesided", "twosided"] | None = None,
    scale_by_freq: bool | None = None,
    mode: Literal["default", "psd", "magnitude", "angle", "phase"] | None = None,
    scale: Literal["default", "linear", "dB"] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    *,
    data: Any = None,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, AxesImage]: ...


def spy(
    Z: ArrayLike,
    precision: float | Literal["present"] = 0,
    marker: str | None = None,
    markersize: float | None = None,
    aspect: Literal["equal", "auto"] | float | None = "equal",
    origin: Literal["upper", "lower"] = "upper",
    **kwargs: Any,
) -> AxesImage: ...


def stackplot(
    x: ArrayLike,
    *args: ArrayLike,
    labels: Sequence[str] = (),
    colors: Sequence[ColorType] | None = None,
    hatch: Sequence[str] | None = None,
    baseline: Literal["zero", "sym", "wiggle", "weighted_wiggle"] = "zero",
    data: Any = None,
    **kwargs: Any,
) -> list[PolyCollection]: ...


def stem(
    *args: ArrayLike | str,
    linefmt: str | None = None,
    markerfmt: str | None = None,
    basefmt: str | None = None,
    bottom: float = 0,
    label: str | None = None,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    data: Any = None,
) -> StemContainer: ...


def step(
    x: ArrayLike,
    y: ArrayLike,
    *args: Any,
    where: Literal["pre", "post", "mid"] = "pre",
    data: Any = None,
    **kwargs: Any,
) -> list[Line2D]: ...


def streamplot(
    x: ArrayLike,
    y: ArrayLike,
    u: ArrayLike,
    v: ArrayLike,
    density: float | tuple[float, float] = 1,
    linewidth: float | ArrayLike | None = None,
    color: ColorType | ArrayLike | None = None,
    cmap: str | Colormap | None = None,
    norm: str | Normalize | None = None,
    arrowsize: float = 1,
    arrowstyle: str = "-|>",
    minlength: float = 0.1,
    transform: Any = None,
    zorder: float | None = None,
    start_points: ArrayLike | None = None,
    maxlength: float = 4.0,
    integration_direction: Literal["forward", "backward", "both"] = "both",
    broken_streamlines: bool = True,
    *,
    data: Any = None,
) -> Any: ...


def table(
    cellText: Sequence[Sequence[str]] | None = None,
    cellColours: Sequence[Sequence[ColorType]] | None = None,
    cellLoc: Literal["left", "center", "right"] = "right",
    colWidths: Sequence[float] | None = None,
    rowLabels: Sequence[str] | None = None,
    rowColours: Sequence[ColorType] | None = None,
    rowLoc: Literal["left", "center", "right"] = "left",
    colLabels: Sequence[str] | None = None,
    colColours: Sequence[ColorType] | None = None,
    colLoc: Literal["left", "center", "right"] = "center",
    loc: str = "bottom",
    bbox: tuple[float, float, float, float] | None = None,
    edges: str = "closed",
    **kwargs: Any,
) -> Any: ...


def text(
    x: float, y: float, s: str, fontdict: dict[str, Any] | None = None, **kwargs: Any
) -> Text: ...


def tick_params(axis: Literal["both", "x", "y"] = "both", **kwargs: Any) -> None: ...


def ticklabel_format(
    *,
    axis: Literal["both", "x", "y"] = "both",
    style: Literal["", "sci", "scientific", "plain"] | None = None,
    scilimits: tuple[int, int] | None = None,
    useOffset: bool | float | None = None,
    useLocale: bool | None = None,
    useMathText: bool | None = None,
) -> None: ...


def tricontour(*args: Any, **kwargs: Any) -> QuadContourSet: ...


def tricontourf(*args: Any, **kwargs: Any) -> QuadContourSet: ...


def tripcolor(
    *args: Any,
    alpha: float = 1.0,
    norm: str | Normalize | None = None,
    cmap: str | Colormap | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    shading: Literal["flat", "gouraud"] = "flat",
    facecolors: ArrayLike | None = None,
    **kwargs: Any,
) -> PolyCollection: ...


def triplot(
    *args: Any,
    **kwargs: Any
) -> list[Line2D] | tuple[list[Line2D], list[Line2D]]: ...


def violinplot(
    dataset: ArrayLike | Sequence[ArrayLike],
    positions: ArrayLike | None = None,
    vert: bool | None = None,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    widths: float | ArrayLike = 0.5,
    showmeans: bool = False,
    showextrema: bool = True,
    showmedians: bool = False,
    quantiles: Sequence[float | Sequence[float]] | None = None,
    points: int = 100,
    bw_method: (
        Literal["scott", "silverman"] | float | Callable[[GaussianKDE], float] | None
    ) = None,
    side: Literal["both", "low", "high"] = "both",
    *,
    data: Any = None,
) -> dict[str, Collection]: ...


def vlines(
    x: float | ArrayLike,
    ymin: float | ArrayLike,
    ymax: float | ArrayLike,
    colors: ColorType | Sequence[ColorType] | None = None,
    linestyles: LineStyleType = "solid",
    label: str = "",
    *,
    data: Any = None,
    **kwargs: Any,
) -> LineCollection: ...


def xcorr(
    x: ArrayLike,
    y: ArrayLike,
    normed: bool = True,
    detrend: Callable[[ArrayLike], ArrayLike] = ...,
    usevlines: bool = True,
    maxlags: int = 10,
    *,
    data: Any = None,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, LineCollection | Line2D, Line2D | None]: ...


def sci(im: ColorizingArtist) -> None: ...


def title(
    label: str,
    fontdict: dict[str, Any] | None = None,
    loc: Literal["left", "center", "right"] | None = None,
    pad: float | None = None,
    *,
    y: float | None = None,
    **kwargs: Any,
) -> Text: ...


def xlabel(
    xlabel: str,
    fontdict: dict[str, Any] | None = None,
    labelpad: float | None = None,
    *,
    loc: Literal["left", "center", "right"] | None = None,
    **kwargs: Any,
) -> Text: ...


def ylabel(
    ylabel: str,
    fontdict: dict[str, Any] | None = None,
    labelpad: float | None = None,
    *,
    loc: Literal["bottom", "center", "top"] | None = None,
    **kwargs: Any,
) -> Text: ...


def xscale(value: str | ScaleBase, **kwargs: Any) -> None: ...


def yscale(value: str | ScaleBase, **kwargs: Any) -> None: ...


# Colormap setter functions
def autumn() -> None: ...
def bone() -> None: ...
def cool() -> None: ...
def copper() -> None: ...
def flag() -> None: ...
def gray() -> None: ...
def hot() -> None: ...
def hsv() -> None: ...
def jet() -> None: ...
def pink() -> None: ...
def prism() -> None: ...
def spring() -> None: ...
def summer() -> None: ...
def winter() -> None: ...
def magma() -> None: ...
def inferno() -> None: ...
def plasma() -> None: ...
def viridis() -> None: ...
def nipy_spectral() -> None: ...
