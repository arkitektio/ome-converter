"""Convert a bioformats-readable file into one ``ArrayDataset`` per series.

Each series becomes a pyramided ``ArrayDataset``, and everything the file says about it
is pinned to a *coordinate*: facts true of the whole series (the OME document, the
microscope body, the stage) hang off the origin anchor; facts true of one channel (its
histogram, name, detector, light path) hang off that channel's anchor. Physical scale is
an edge, not a property — the series register into one micrometer world, and each series
gets a scene of its own over it. A tile scan is the exception in both directions: its
tiles get no scene where the file also holds the stitched image they were merged into, and
one mosaic scene between them where it does not — see `scene_groups`.

Nothing about that scene is bootstrapped: every lens and layer is authored explicitly,
**one layer per channel**. The only exception is a packed RGB(A) photograph, decided from
the reader's sample axis alone — see `packs_one_pixel`, and `tests/test_app.py`,
which tests the rule.

Absent facts stay absent. No objective is asserted where OME names none, and a file with
no pixel calibration is **not** registered into the world: ``scale={x: 1.0}`` is not a
neutral default but the claim that one pixel is one micrometer. Such a dataset is staged
in its own pixel grid instead.
"""

from __future__ import annotations

import json
import logging
import operator
import os
import re
import tempfile
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from arkitekt import App, run
from mikro import Mikro, build_pyramid, canonical, scales_from, space_3d
from mikro import mikro_service
from mikro.api.schema import (
    ArrayDataset,
    AxisAnchorInput,
    AxisType,
    Blending,
    ColorMap,
    CoordinateAnchorInput,
    File,
    LabelInput,
    OmeMetadataInput,
    OptikitStateInput,
    PreferredView,
    ProjectionMode,
    Scene,
    SourceFileInput,
    StageStateInput,
    ValueHistogramInput,
    ValueRelation,
)
from arkitekt import Task
from ome_metadata import (
    COLOR_BY_EMISSION,
    FAR_RED,
    channel_lightpath,
    channel_optikit,
    find_instrument,
    find_objective,
    ome_rgba,
    qty,
    to_micrometer,
    um,
    unit_symbol,
)
from mikro.pyramid import axes_for
from converter_metadata import device, devices, setting, text
from logos import declare_logos, logo_of
from mikro.inputs.render import (
    RGB_PRIMARIES,
    color_for_wavelength,
    fallback_color,
    rgba,
)
from mikro.inputs.render import composite_graph

logger = logging.getLogger(__name__)

# The version of the app. A release is the commit tagged with it (v1.2.3).
__version__ = "0.1.0"

app = App(
    "ome-converter",
    __version__,
    description=(
        "Converts bioformats-readable files (czi, lif, nd2, ome-tiff, ...) into mikro "
        "datasets, one per series, with their OME metadata, light path and device state."
    ),
    author="jhnnsrs",
)
app.service(mikro_service)
declare_logos(app)


# ---------------------------------------------------------------------------
# Scratch space
#
# Where the downloaded source file goes. Its own section because the obvious answer is
# wrong on at least one machine this runs on: `/tmp` there is a 15 GB tmpfs — RAM — and
# the files these converters exist for are routinely larger than that.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The axes the world below is built over. Every scale and offset is derived by
# intersecting a series' own dims with this set — never by reading the OME record's
# fields, and never by reading them back off the CoordinateSystem. `register` raises
# if a scale or an offset names an axis the two spaces do not share, and a 2D series
# routinely carries a populated `position_z`, so building from the intersection is
# what keeps that from being a crash instead of an omission.
WORLD_AXES = frozenset({"z", "y", "x"})

# The axis type for each name this converter can emit. Stated rather than inferred:
# the library's fallback calls every unrecognised name SPACE, which is right for
# z/y/x and silently wrong for anything else — and it matters twice over, because
# the pyramid coarsens exactly the SPACE axes.
AXIS_TYPES: dict[str, AxisType] = {
    "t": AxisType.TIME,
    "c": AxisType.CHANNEL,
    "z": AxisType.SPACE,
    "y": AxisType.SPACE,
    "x": AxisType.SPACE,
}

AXIS_LONG_NAMES = {
    "t": "time",
    "c": "channel",
    "z": "depth",
    "y": "height",
    "x": "width",
}

# `ValueHistogramInput.from_array` calls `np.asarray`, so it is the one place a lazy
# series is forced into memory. Above this many bytes per channel slice the histogram
# is computed from the finest pyramid level that fits instead.
HISTOGRAM_BUDGET_BYTES = 256 * 1024 * 1024

# Past this many layers a scene stops being useful. Scenes are per series now, so only a
# mosaic scene can realistically reach it — 98 four-channel tiles are 392 layers. The
# world still holds every registration and outlives the run, so a subset can be staged
# later.
MAX_SCENE_LAYERS = 48

# How Leica names the stitched image of a tile scan: the scan's own name with "merged"
# appended — "R 1_Merged", but also "…_4merged" with no separator at all.
MERGED_SUFFIX = re.compile(r"[\s_-]*merged$", re.IGNORECASE)

# bioio hands the reader back lazily; the first import pulls a JRE and the bioformats
# jars through cjdk/maven, which is tens of seconds and wants the network once.
_READER: Any = None


# ---------------------------------------------------------------------------
# Units and value coercion
#
# Every quantity field on the schema is a pint-backed scalar that rejects a bare
# float, so a number without a unit cannot be uploaded — and a None that slips
# through string formatting becomes the literal "None nm". Both failure modes are
# closed here rather than at each of the ~60 call sites: these four helpers return
# None whenever they have nothing to say, and a None setting is dropped before it
# reaches an input.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The array
# ---------------------------------------------------------------------------


def load_series(img: Any, index: int) -> tuple[xr.DataArray, dict[str, str], bool]:
    """Series `index` as a named, lazy array, the axis types to state, and is-it-RGB.

    Selected by position, never by name. Scene names are not unique — a LIF tile scan
    calls every tile the same thing (98 of 99 series in the test file are ``"R 1"``) —
    and ``set_scene(name)`` resolves a name to its first match, or does nothing at all
    when the current scene already carries it. Either way every tile after the first
    would be uploaded with the first one's pixels under its own OME record. A name is
    therefore refused outright rather than passed on.

    Returns the array with its dims already reduced to the ones this converter is
    willing to claim, a ``{name: TYPE}`` mapping for any axis whose type the naming
    convention would get wrong, and whether the ``c`` axis is a *sample* axis — three
    or four samples packed into one pixel, i.e. a photograph.

    That last flag is the reader's own answer to "is this RGB", and it is the only
    trustworthy one. It is decided here, where the ``"S"`` axis is still visible; three
    lines further down that axis has been renamed to ``c`` and is indistinguishable
    from three dyes. See `convert_series` for the heuristics this replaced.
    """
    img.set_scene(operator.index(index))
    array = img.xarray_dask_data

    # RGB samples. Bioformats presents a JPEG-style RGB image as one metadata channel
    # with three samples, and selecting sample 0 (the obvious reading) throws away two
    # thirds of the picture.
    #
    # `packed_samples` is recorded here and nowhere else. Only 3 samples (RGB) or 4
    # (RGBA) are a picture: any other extent behind an "S" axis is some other packing,
    # and the honest reading of it is ordinary channels — one layer each.
    packed_samples = False
    if "S" in array.dims:
        if array.sizes["S"] > 1 and array.sizes.get("C", 1) == 1:
            packed_samples = array.sizes["S"] in (3, 4)
            array = array.squeeze("C").rename({"S": "c"})
        else:
            array = array.isel(S=0)

    array = array.rename({d: str(d).lower() for d in array.dims if str(d) in "TCZYXS"})

    if "c" not in array.dims:
        array = array.expand_dims("c")

    # Keep c, y and x unconditionally; keep t and z only where they have extent. A kept
    # axis is a permanent structural claim: a singleton `t` asserts a time series, and a
    # singleton `z` lets the world accept a z scale a photograph has no thickness to
    # state. `c` is kept at extent 1 for the opposite reason — the per-channel anchors and
    # layers key off it, so without it a single-channel stack has nowhere to hang its
    # histogram, label, detector or light path.
    dropped = [
        d for d in array.dims if str(d) not in ("c", "y", "x") and array.sizes[d] == 1
    ]
    if dropped:
        array = array.squeeze(dropped, drop=True)

    # Anything left that is not t/c/z/y/x is a mosaic or tile index. It has to be
    # declared: the type default calls it SPACE, which would both sort it in among
    # z/y/x and hand it to the pyramid to be halved — quietly turning 12 tiles into 6.
    extra_types = {str(d): "INDEX" for d in array.dims if str(d) not in AXIS_TYPES}

    return (
        xr.DataArray(array.data, dims=[str(d) for d in array.dims]),
        extra_types,
        packed_samples,
    )


# ---------------------------------------------------------------------------
# OME lookup
# ---------------------------------------------------------------------------


def series_ome_json(file: File, index: int, image: Any, instrument: Any | None) -> str:
    """The OME record for this one series, as the JSON string the anchor takes.

    Reduced to this image and its instrument rather than dumping the whole file's
    document: a 40-series slide would otherwise carry 40 identical copies of it, and
    nothing client-side reads the string, so nothing would ever notice the waste. The
    per-series document is also the one that is actually *true of this dataset*.

    The wrapper around it names the source file too. That link is made properly by the
    ``sourceFiles`` entry on the dataset — a ``SourceFileInput`` carrying the file id
    and this series' OME identifier, which is what a query traverses. Repeating it here
    costs nothing and keeps the blob a complete record of where it came from even when
    read on its own; ``series_id`` below is deliberately the same value the link uses.
    """
    from ome_types.model import OME

    payload: dict[str, Any] = {
        "source_file": file.name,
        "source_file_id": str(file.id),
        "series_index": index,
        "series_id": str(image.id),
    }
    try:
        reduced = OME(
            images=[image],
            instruments=[instrument] if instrument is not None else [],
        )
        payload["ome"] = json.loads(reduced.model_dump_json(exclude_unset=True))
    except Exception:
        # A reduced document can fail to revalidate when the image references
        # something left behind (an experimenter, a plate). The metadata blob is not
        # worth failing an upload over, so fall back to the image alone.
        logger.warning("Could not rebuild a reduced OME document for series %s", index)
        payload["ome"] = json.loads(image.model_dump_json(exclude_unset=True))
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# Device state
#
# The layer OME records but the light path has no member for: what the software
# wrote down, what the stage was doing, and every device setting that is a number
# on a named box rather than something the light passed through.
# ---------------------------------------------------------------------------


def series_optikit(
    file: File,
    index: int,
    image: Any,
    instrument: Any | None,
    scale_note: str | None,
) -> OptikitStateInput | None:
    """The hardware state of the whole series."""
    pixels = image.pixels
    planes = list(pixels.planes)
    first = planes[0] if planes else None
    env = getattr(image, "imaging_environment", None)
    microscope = getattr(instrument, "microscope", None) if instrument else None
    objective_settings = getattr(image, "objective_settings", None)
    objective = find_objective(instrument, image)

    stage = None
    if first is not None:
        x = um(first.position_x, first.position_x_unit)
        y = um(first.position_y, first.position_y_unit)
        z = um(first.position_z, first.position_z_unit)
        if any((x, y, z)):
            stage = StageStateInput(x=x, y=y, z=z)

    temperature = None
    if env is not None:
        temperature = qty(env.temperature, env.temperature_unit, "degC")

    state_devices = devices(
        device(
            "microscope",
            "microscope",
            setting("instrument-id", string=text(getattr(instrument, "id", None))),
            setting(
                "manufacturer", string=text(getattr(microscope, "manufacturer", None))
            ),
            setting("model", string=text(getattr(microscope, "model", None))),
            setting(
                "serial-number", string=text(getattr(microscope, "serial_number", None))
            ),
            setting("type", string=text(getattr(microscope, "type", None))),
        ),
        # The run itself. This is the only home the source file's identity has left,
        # so `source-file` and `source-file-id` are load-bearing rather than decorative.
        device(
            "acquisition",
            "software",
            setting("source-file", string=file.name),
            setting("source-file-id", string=str(file.id)),
            setting("series-index", number=float(index)),
            setting("series-id", string=text(image.id)),
            setting("image-name", string=text(image.name)),
            setting("image-description", string=text(image.description)),
            setting(
                "acquisition-date",
                string=image.acquisition_date.isoformat()
                if image.acquisition_date
                else None,
            ),
            setting("dimension-order", string=text(pixels.dimension_order)),
            setting("pixel-type", string=text(pixels.type)),
            setting("significant-bits", number=pixels.significant_bits),
            setting("interleaved", flag=pixels.interleaved),
            setting("big-endian", flag=pixels.big_endian),
            setting("size-t", number=float(pixels.size_t)),
            setting("size-c", number=float(pixels.size_c)),
            setting("size-z", number=float(pixels.size_z)),
            setting("size-y", number=float(pixels.size_y)),
            setting("size-x", number=float(pixels.size_x)),
            setting(
                "physical-size-x",
                quantity=um(pixels.physical_size_x, pixels.physical_size_x_unit),
            ),
            setting(
                "physical-size-y",
                quantity=um(pixels.physical_size_y, pixels.physical_size_y_unit),
            ),
            setting(
                "physical-size-z",
                quantity=um(pixels.physical_size_z, pixels.physical_size_z_unit),
            ),
            # Says out loud what the registration below had to assume. There is no way
            # to express "unknown" through a BY_DIMENSION edge — omitting an axis and
            # passing 1.0 are the same claim — so the assumption is made visible here
            # rather than hidden in the edge.
            setting("scale-note", string=scale_note),
            setting(
                "time-increment",
                quantity=qty(pixels.time_increment, pixels.time_increment_unit, "s"),
            ),
        ),
        device(
            "objective",
            "objective",
            setting("model", string=text(getattr(objective, "model", None))),
            setting(
                "manufacturer", string=text(getattr(objective, "manufacturer", None))
            ),
            setting(
                "magnification",
                number=getattr(objective, "nominal_magnification", None)
                or getattr(objective, "calibrated_magnification", None),
            ),
            setting("numerical-aperture", number=getattr(objective, "lens_na", None)),
            setting(
                "working-distance",
                quantity=um(
                    getattr(objective, "working_distance", None),
                    getattr(objective, "working_distance_unit", None),
                ),
            ),
            setting("immersion", string=text(getattr(objective, "immersion", None))),
            setting("correction", string=text(getattr(objective, "correction", None))),
            setting(
                "correction-collar",
                number=getattr(objective_settings, "correction_collar", None),
            ),
            setting("medium", string=text(getattr(objective_settings, "medium", None))),
            setting(
                "refractive-index",
                number=getattr(objective_settings, "refractive_index", None),
            ),
        ),
        device(
            "environment",
            "environment",
            setting(
                "air-pressure",
                quantity=qty(
                    getattr(env, "air_pressure", None),
                    getattr(env, "air_pressure_unit", None),
                    "hPa",
                ),
            ),
            setting("humidity", number=getattr(env, "humidity", None)),
            setting("co2-percent", number=getattr(env, "co2_percent", None)),
        ),
        # Only reached when the position could not be read as a length — which is the
        # common case, because OME's schema default for a plane position is
        # "reference frame", an arbitrary unit that is not micrometers. The typed
        # StageStateInput above must not claim it, but the numbers are still what the
        # microscope wrote down, so they are kept verbatim rather than dropped.
        device(
            "stage-xy",
            "stage",
            setting(
                "position-x",
                string=f"{first.position_x} {unit_symbol(first.position_x_unit)}"
                if first is not None
                and first.position_x is not None
                and to_micrometer(first.position_x, first.position_x_unit) is None
                else None,
            ),
            setting(
                "position-y",
                string=f"{first.position_y} {unit_symbol(first.position_y_unit)}"
                if first is not None
                and first.position_y is not None
                and to_micrometer(first.position_y, first.position_y_unit) is None
                else None,
            ),
            setting(
                "position-z",
                string=f"{first.position_z} {unit_symbol(first.position_z_unit)}"
                if first is not None
                and first.position_z is not None
                and to_micrometer(first.position_z, first.position_z_unit) is None
                else None,
            ),
        ),
        device(
            "timing",
            "acquisition",
            setting("plane-count", number=float(len(planes)) if planes else None),
            setting(
                "first-delta-t",
                quantity=qty(first.delta_t, first.delta_t_unit, "s") if first else None,
            ),
            setting(
                "last-delta-t",
                quantity=qty(planes[-1].delta_t, planes[-1].delta_t_unit, "s")
                if planes
                else None,
            ),
            setting(
                "exposure-time",
                quantity=qty(first.exposure_time, first.exposure_time_unit, "s")
                if first
                else None,
            ),
        ),
    )

    if stage is None and temperature is None and not state_devices:
        return None
    return OptikitStateInput(
        stage=stage, temperature=temperature, devices=state_devices
    )


# ---------------------------------------------------------------------------
# The light path
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------


def channel_label(channel: Any | None, index: int, implicit_rgb: bool) -> str:
    if implicit_rgb:
        return (
            ("Red", "Green", "Blue", "Alpha")[index] if index < 4 else f"Sample {index}"
        )
    if channel is not None:
        for candidate in (channel.name, channel.fluor, channel.id):
            rendered = text(candidate)
            if rendered:
                return rendered
    return f"Channel {index}"


def origin_anchor(
    data: xr.DataArray,
    file: File,
    index: int,
    image: Any,
    instrument: Any | None,
    scale_note: str | None,
) -> CoordinateAnchorInput:
    """Everything true of the whole series, pinned to voxel (0, ..., 0).

    The axis anchors are built from ``data.dims``, so they follow the squeeze rule
    automatically — an anchor naming an axis the dataset does not have has nowhere
    to land.
    """
    return CoordinateAnchorInput(
        axis_anchors=tuple(AxisAnchorInput(axis=str(d), value=0) for d in data.dims),
        label=LabelInput(label=text(image.name) or f"Series {index}"),
        ome_metadata=OmeMetadataInput(
            metadata_string=series_ome_json(file, index, image, instrument)
        ),
        microscope=series_optikit(file, index, image, instrument, scale_note),
    )


def channel_color(
    channel: Any | None, index: int, implicit_rgb: bool, total: int
) -> tuple[int, int, int, int]:
    """What to draw this channel in.

    OME's own colour first — it is what the acquisition software displayed, and the
    only one of these that is a recorded fact rather than a guess. Then the emission
    wavelength, which at least lands in the right part of the spectrum. Then a palette,
    so that two channels never silently collide.
    """
    if implicit_rgb:
        # Index 3 of an RGBA pixel is alpha — a mask, not a colour. `% 3` would paint it
        # the same red as channel 0, which reads as two identical channels in every
        # listing that shows the anchor colour. White is the honest neutral: the layer
        # composite does not use this colour at all (it hardcodes the three primaries),
        # so this only ever labels the anchor.
        if index >= 3:
            return rgba((255, 255, 255))
        return rgba(RGB_PRIMARIES[index])

    if channel is not None:
        from_file = ome_rgba(channel)
        if from_file is not None:
            return from_file

        emission = to_micrometer(
            channel.emission_wavelength, channel.emission_wavelength_unit
        )
        if emission is not None:
            nanometers = emission * 1000
            return color_for_wavelength(nanometers, COLOR_BY_EMISSION, FAR_RED)

    return fallback_color(index, total)


@dataclass(frozen=True)
class ChannelSpec:
    """One channel, as both the anchors and the layers need it.

    Computed once because the histogram is the expensive part — it is the contrast
    window the layer opens with *and* the render metadata the anchor carries, and
    computing it twice would double the only forced materialisation in the converter.
    """

    index: int
    label: str
    color: tuple[int, int, int, int]
    histogram: ValueHistogramInput
    clim: tuple[float, float]
    ome_channel: Any | None


def display_window(shown: np.ndarray) -> tuple[float, float]:
    """The contrast window a channel's layer opens with: p1/p99 of what it *shows*.

    `shown` is the picture, which is not the voxels wherever the layer projects — see
    `channel_specs`. p1/p99 rather than min/max, because a single hot pixel should not
    decide the display range of the whole channel — except where the percentiles coincide
    (a channel that is over 99 % one value), which would be a window of no width.
    """
    shown = shown[np.isfinite(shown)]
    low, high = (float(v) for v in np.percentile(shown, [1, 99]))
    if high <= low:
        low, high = float(shown.min()), float(shown.max())
    if high <= low:
        high = low + 1.0
    return low, high


def channel_specs(
    data: xr.DataArray,
    pyramid: Sequence[xr.DataArray],
    image: Any,
    implicit_rgb: bool,
    is_volume: bool,
) -> list[ChannelSpec]:
    """One spec per channel of the array.

    A contrast window is per channel by nature — a nuclear stain and a brightfield
    channel share no sensible range — which is why each histogram is computed from that
    channel's slice alone rather than from the stack.
    """
    channels = list(image.pixels.channels)
    specs: list[ChannelSpec] = []
    total = data.sizes["c"]

    for index in range(total):
        # Implicit RGB has three array channels behind one OME channel, so the metadata
        # does not index by array position and only the first is real.
        channel = (
            channels[index] if index < len(channels) and not implicit_rgb else None
        )

        base = data.isel(c=index)
        slice_ = base
        if base.nbytes > HISTOGRAM_BUDGET_BYTES:
            # `from_array` materialises whatever it is handed, so a multi-gigabyte
            # channel is summarised from a pyramid level instead: the finest one that
            # fits, not the coarsest. The artifact is worth naming, and it grows with
            # every level: max downsampling keeps only block maxima, so the distribution
            # shifts *up* and p1/p99 come out brighter than the truth. (Mean had the
            # opposite bias — a distribution narrowed towards the middle.) Hence level 0
            # wherever it fits.
            slice_ = next(
                (
                    level.isel(c=index)
                    for level in pyramid[1:]
                    if level.isel(c=index).nbytes <= HISTOGRAM_BUDGET_BYTES
                ),
                pyramid[-1].isel(c=index),
            )

        # Materialised once and handed to both: `from_array` on the lazy slice would
        # read the channel, and the window would then read it again.
        values = np.asarray(slice_)

        # What the layer draws. A z-stack is drawn as a maximum projection, and the
        # maxima along z are a different population from the voxels: in sparse
        # fluorescence most voxels are background, so the stack's p99 sits barely above
        # it while nearly every projected pixel holds something brighter. Measured on a
        # LIF tile, the voxel window was [0, 2] of 0..255 and 62 % of the projection
        # saturated. The histogram on the anchor stays the voxel one — that is a fact
        # about the data; the window is a choice about a picture.
        shown = values
        if is_volume:
            projection = base.max("z")
            if slice_ is base:
                shown = values.max(axis=slice_.dims.index("z"))
            elif projection.nbytes <= HISTOGRAM_BUDGET_BYTES:
                # A projection is one plane, so it fits where the stack did not — and
                # it is exact, at the cost of one more read of the channel. Worth it:
                # on the 5 GB stitched LIF channel the coarsest level's window was
                # [0, 253] against [0, 79] here, a layer opening three times too dim.
                shown = np.asarray(projection)
            else:
                shown = values.max(axis=slice_.dims.index("z"))

        specs.append(
            ChannelSpec(
                index=index,
                label=channel_label(channel, index, implicit_rgb),
                color=channel_color(channel, index, implicit_rgb, total),
                histogram=ValueHistogramInput.from_array(values),
                clim=display_window(shown),
                ome_channel=channel,
            )
        )

    return specs


def channel_anchors(
    specs: Sequence[ChannelSpec],
    image: Any,
    instrument: Any | None,
) -> list[CoordinateAnchorInput]:
    """One anchor per channel: its contrast window, its identity, its optics."""
    return [
        CoordinateAnchorInput(
            axis_anchors=(AxisAnchorInput(axis="c", value=spec.index),),
            value_histogram=spec.histogram,
            label=LabelInput(label=spec.label),
            microscope=(
                channel_optikit(spec.ome_channel, spec.index, spec.label, instrument)
                if spec.ome_channel is not None
                else None
            ),
            light_graph=(
                channel_lightpath(spec.ome_channel, image, instrument)
                if spec.ome_channel is not None
                else None
            ),
        )
        for spec in specs
    ]


def channel_lens(dataset: Any, index: int) -> Any:
    """The lens that pins channel `index`: what a layer over one channel draws.

    A layer draws its lens and nothing narrower, so a channel is a lens of its own.
    A dataset with no channel axis is one signal already, and its whole lens is it.
    """
    if "c" not in dataset.axis_names:
        return dataset.lens()
    return dataset.lens(c=index)


def author_channel_layers(
    mikro: Mikro,
    dataset: ArrayDataset,
    scene: Any,
    specs: Sequence[ChannelSpec],
    is_volume: bool,
    implicit_rgb: bool,
    order_offset: int = 0,
) -> int:
    """One layer per channel of `dataset`, composited into `scene`.

    Per channel, because one layer over a multi-channel dataset means one recipe standing
    in for every channel: no per-channel colour, no per-channel contrast, nothing to
    toggle. It is also the only place OME's own channel colour can land.

    The one exception is a packed RGB(A) series, one layer over its samples: those
    "channels" are one photograph, and separating them gives toggles that are only ever
    right in one combination. `implicit_rgb` is trusted only as far as it can be checked —
    this function refuses it for any channel count that is not 3 or 4. That refusal is the
    backstop for the whole rule, kept here rather than left to the server because it fires
    first and demotes instead of failing the upload.

    One lens per channel, beside the dataset's whole one: a layer draws its lens and
    nothing narrower, so the channel a layer shows is the one its lens pins.

    **Which kind of layer** — each recipe has a server kind that states it as fields:

    * one channel, flat → ``create_intensity_layer``
    * one channel, through a z-stack → ``create_volume_layer``
    * a packed RGB(A) pixel, flat → ``create_rgb_layer``

    ``create_layer`` and a render graph are left for the one shape none of them can say,
    a packed RGB(A) **z-stack**: an RGB layer carries no projection mode.

    Returns the number of layers created — not ``len(specs)`` in the RGB case.
    """
    lens = dataset.lens()

    if implicit_rgb and len(specs) not in (3, 4):
        logger.warning(
            "Refusing to composite %s channel(s) into one RGB layer; a render graph "
            "covers exactly one channel unless the channels are the 3 or 4 samples of "
            "one pixel. Authoring one layer per channel instead.",
            len(specs),
        )
        implicit_rgb = False

    if implicit_rgb:
        if is_volume:
            # The one shape the flat kinds cannot say. Three primaries even for RGBA:
            # the fourth sample is alpha, a mask rather than a colour. `rgb_graph` says
            # nearly this, but its bare triples are what the server rejects.
            mikro.create_layer(
                lens=lens.id,
                scene=scene.id,
                render_graph=composite_graph(
                    [
                        {
                            "lens": channel_lens(dataset, index).id,
                            "color": rgba(primary),
                            "colormap": ColorMap.INTENSITY,
                            # Already display-referred; stretching it would change the
                            # colours the image was authored with.
                            "clim_min": 0.0,
                            "clim_max": 255.0,
                            "mode": ProjectionMode.MIP,
                        }
                        for index, primary in enumerate(RGB_PRIMARIES)
                    ]
                ),
                blending=Blending.ADDITIVE,
                visible=True,
                order=order_offset,
            )
            return 1

        # Flat: the photograph is exactly what an RGB layer is for, and it says so as
        # three component lenses, one per sample, rather than as a tree a reader has to
        # recognise. One contrast window across all three — they are components of one
        # picture, not three signals with their own scaling — left at the full 8-bit
        # range for the reason above.
        #
        # No `blending` argument here, and none accepted: the server forces NORMAL,
        # because a photograph sits *over* what is beneath it rather than summing with
        # it. That is a change from the ADDITIVE this path used to pass.
        mikro.create_rgb_layer(
            lens=lens.id,
            scene=scene.id,
            red_lens=channel_lens(dataset, 0).id,
            green_lens=channel_lens(dataset, 1).id,
            blue_lens=channel_lens(dataset, 2).id,
            clim_min=0.0,
            clim_max=255.0,
            visible=True,
            order=order_offset,
        )
        return 1

    for spec in specs:
        # `color`, not a colormap: what this converter knows about a channel is an RGBA
        # quad — OME's own channel colour, or the one its emission wavelength implies —
        # and no named map means it. That fact used to force the whole layer onto a
        # render graph; an intensity layer carries it as a field now.
        settings = dict(
            lens=channel_lens(dataset, spec.index).id,
            scene=scene.id,
            color=spec.color,
            # Of what the layer shows, not of the voxels — see `channel_specs`.
            clim_min=spec.clim[0],
            clim_max=spec.clim[1],
            blending=Blending.ADDITIVE,
            visible=True,
            order=order_offset + spec.index,
        )

        if is_volume:
            # A z-stack has no single plane to show, so each channel projects through the
            # volume rather than picking one slice. Same kind of layer, one field more.
            mikro.create_volume_layer(mode=ProjectionMode.MIP, **settings)
        else:
            mikro.create_intensity_layer(**settings)

    return len(specs)


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


def voxel_scale(
    pixels: Any, dims: Sequence[str]
) -> tuple[dict[str, float] | None, str | None]:
    """The per-axis factor from this dataset's indices into micrometers.

    Returns ``(None, reason)`` when the file has no lateral calibration. Registering
    anyway would not be a neutral default — ``scale={"x": 1.0}`` positively claims
    that one pixel is one micrometer, and the dataset would then be composited into a
    shared scene at the wrong size next to genuinely calibrated series.

    A missing *axial* size is different: omitting ``z`` from the mapping and passing
    ``z=1.0`` are the same claim (missing shared axes default to 1.0), so there is no
    way to abstain. The assumption is recorded as a setting instead of hidden.
    """
    shared = [d for d in dims if d in WORLD_AXES]
    sizes = {
        "x": to_micrometer(pixels.physical_size_x, pixels.physical_size_x_unit),
        "y": to_micrometer(pixels.physical_size_y, pixels.physical_size_y_unit),
        "z": to_micrometer(pixels.physical_size_z, pixels.physical_size_z_unit),
    }

    if sizes["x"] is None or sizes["y"] is None:
        return (
            None,
            "no lateral pixel size in the file — not registered into a physical space",
        )

    note = None
    if "z" in shared and sizes["z"] is None:
        note = "physical-size-z unknown, registered as 1 um per plane"

    return {axis: sizes.get(axis) or 1.0 for axis in shared}, note


def stage_offsets(pixels: Any, dims: Sequence[str]) -> dict[str, float]:
    """Where the stage was, in the world's micrometers.

    The same numbers that go into ``StageStateInput`` on the origin anchor, so the
    recorded state and the placement agree by construction.
    """
    planes = list(pixels.planes)
    if not planes:
        return {}
    first = planes[0]
    positions = {
        "x": to_micrometer(first.position_x, first.position_x_unit),
        "y": to_micrometer(first.position_y, first.position_y_unit),
        "z": to_micrometer(first.position_z, first.position_z_unit),
    }
    return {
        axis: positions[axis]
        for axis in dims
        if axis in WORLD_AXES and positions.get(axis) is not None
    }


def compose_scene(
    mikro: Mikro, name: str, coordinate_system: Any, entries: Sequence[Converted]
) -> Scene:
    """A scene over `coordinate_system`, with every layer authored explicitly.

    ``create_scene`` rather than ``stage`` / ``create_scene_from_coordinate_system``:
    the bootstrap decides on your behalf how many registered sources become layers,
    which recipe each one gets, and — for anything multi-channel — collapses a whole
    dataset into a single layer. This converter knows exactly what it uploaded and what
    each channel is, so none of those are decisions worth delegating. It adopts the
    world system, then authors one lens per dataset and one layer per channel, each
    with its own colour, contrast window and projection.

    `entries` is one series, or the tiles of one mosaic — never the whole file; which
    series share a scene is `scene_groups`' decision, not this function's.

    The scene carries only what is genuinely scene-wide. ``blending`` is ADDITIVE
    because that is how fluorescence channels combine — and how three RGB primaries
    recompose a photograph. ``preferred_view`` is a viewer hint and nothing more: the
    projection that actually renders a volume lives on each layer's channel node, so
    this only says which way a viewer should open, and only claims a preference when
    something in the scene has depth to look into.
    """
    scene = mikro.create_scene(
        name=name,
        coordinate_system=coordinate_system.id,
        blending=Blending.ADDITIVE,
        preferred_view=(
            PreferredView.THREE_D
            if any(entry.is_volume for entry in entries)
            else PreferredView.TWO_D
        ),
    )

    # Ordered so a series' channels stay together in the layer stack rather than
    # interleaving with the next series'.
    order = 0
    for entry in entries:
        order += author_channel_layers(
            mikro,
            entry.dataset,
            scene,
            entry.specs,
            entry.is_volume,
            entry.implicit_rgb,
            order,
        )

    logger.info(
        "Scene %s over %s: %s layer(s) from %s dataset(s)",
        scene.id,
        scene.world_coordinate_system.name,
        order,
        len(entries),
    )
    return scene


# ---------------------------------------------------------------------------
# Scenes
#
# Which series are staged, and with what. One scene over the whole file composited
# things that only share a slide: a stack with its RGB snapshot, three positions — and a
# stitched image on top of every tile it was stitched from, which is the same pixels
# twice. Every series is still uploaded and registered; this only decides what is drawn
# together.
# ---------------------------------------------------------------------------


def scene_groups(names: Sequence[str | None]) -> list[list[int]]:
    """The series each scene composes, one list per scene, in order of first series.

    A series is staged on its own unless it is a tile. Bio-Formats splits one LIF tile
    scan into a series per tile, all under the scan's name (``"R 1"`` ×98), and the
    stitched image beside them is that name plus `MERGED_SUFFIX` (``"R 1_Merged"``).
    So:

    * a tile whose stitched image is in the file is in **no** scene — the stitched image
      is the picture, and its own scene shows it;
    * the tiles of a scan with no stitched image are **one** scene between them, the
      mosaic, because there they are the picture;
    * a stitched image with no tiles (an export holding only the merge) is just a series.

    An unnamed series is staged on its own: two missing names are not a shared one.
    """
    stitched = {
        MERGED_SUFFIX.sub("", name)
        for name in names
        if name and MERGED_SUFFIX.search(name)
    } - {""}
    counts: dict[str, int] = {}
    for name in names:
        if name:
            counts[name] = counts.get(name, 0) + 1

    groups: list[list[int]] = []
    mosaics: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        if not name:
            groups.append([index])
        elif name in stitched:
            continue
        elif counts[name] > 1:
            if name not in mosaics:
                mosaics[name] = []
                groups.append(mosaics[name])
            mosaics[name].append(index)
        else:
            groups.append([index])
    return groups


def stage_group(
    mikro: Mikro, title: str, world: Any, entries: Sequence[Converted]
) -> None:
    """Compose the scene(s) for one group out of `scene_groups`.

    The registered series in it share one scene over the world. An uncalibrated series is
    not in the world, so it is staged in its own pixel grid instead — the honest frame for
    it, and one it shares with nothing.
    """
    registered = [entry for entry in entries if entry.registered]
    if registered:
        layer_count = sum(entry.layer_count for entry in registered)
        if layer_count > MAX_SCENE_LAYERS:
            logger.warning(
                "%s would compose %s layers from %s series; not building a scene over "
                "more than %s. The world holds every registration and outlives this run, "
                "so a subset can be composed later.",
                title,
                layer_count,
                len(registered),
                MAX_SCENE_LAYERS,
            )
        else:
            compose_scene(mikro, title, world, registered)

    for entry in entries:
        if entry.registered:
            continue
        if entry.dataset.intrinsic_system is None:
            logger.warning(
                "Series %s has no intrinsic coordinate system to compose over; the "
                "dataset is uploaded but has no scene.",
                entry.series,
            )
            continue
        compose_scene(mikro, entry.name, entry.dataset.intrinsic_system, [entry])


# ---------------------------------------------------------------------------
# One series
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Converted:
    """What the action needs to know about a series after it has landed."""

    series: int
    name: str
    dataset: ArrayDataset
    specs: list[ChannelSpec]
    registered: bool
    is_volume: bool
    implicit_rgb: bool

    @property
    def layer_count(self) -> int:
        """How many layers this series composes into a scene.

        Not ``len(specs)``: a packed RGB(A) series is one layer over its 3 or 4
        samples. Kept here rather than recomputed at each call site so the budget check
        in `stage_group` and what `author_channel_layers` actually creates cannot drift.

        For every other series this is ``len(specs)`` — one layer per channel — and that
        is the invariant, not an implementation detail.
        """
        return 1 if self.implicit_rgb else len(self.specs)


def declared_samples_per_pixel(pixels: Any) -> int | None:
    """What OME says is packed into one pixel of a single-channel series.

    ``SamplesPerPixel`` is a *declaration*, not a count of channels: "this one channel
    is N samples of the same pixel". It is the closest the OME record comes to answering
    "is this a photograph" — and it still does not decide it here. See `packs_one_pixel`
    for why: it can only fire in the collapse direction, and a partial OME record is
    precisely the shape that breaks. This exists to log a disagreement with the reader.

    Only meaningful for a single channel: a series with three channels each of one
    sample is three dyes, whatever any of them declares. ``None`` when the file is
    silent, which ome-types represents as ``None`` rather than defaulting to 1.
    """
    channels = list(pixels.channels)
    if len(channels) != 1:
        return None
    samples = getattr(channels[0], "samples_per_pixel", None)
    return int(samples) if samples is not None else None


def packs_one_pixel(
    packed_samples: bool, declared: int | None, channels: int, dtype: Any
) -> bool:
    """Whether this series' channels are the samples of one pixel — the ONLY exception.

    ONE LAYER PER CHANNEL. A function rather than an inline expression so it can be
    tested without uploading, and so there is one place to read when asking why a series
    got fewer layers than channels.

    Exactly one thing may say yes: the reader's sample axis (``packed_samples`` out of
    `load_series`). Nothing in the OME record decides it, and nothing that *counts*
    channels may either — collapsing three dyes destroys per-channel colour, contrast and
    toggling, which is the error this rule exists to prevent. ``declared`` is taken only
    to log a disagreement; `tests/test_app.py` records why it is not a decider.

    And the samples have to be 8-bit. An RGB layer is for a picture that is already
    display-referred — a snapshot, a scanned slide — and is drawn unstretched over 0..255.
    A 12- or 16-bit sample axis is a colour camera's acquisition: measurements, which want
    a window each and would all saturate under that fixed one. Like every other clause
    here this can only demote, to one layer per sample.
    """
    return packed_samples and channels in (3, 4) and np.dtype(dtype) == np.uint8


def convert_series(
    mikro: Mikro,
    img: Any,
    ome: Any,
    index: int,
    file: File,
    world: Any,
    levels: int,
    method: str,
) -> Converted:
    """Upload one series, and place it if the file says where.

    Never stages it: whether a series gets a scene depends on the series around it, so
    that is decided once, in the action — see `scene_groups`.
    """
    # Refused rather than approximated. Falling back to the last image would attach
    # another series' channels, detector settings and light path to this one — a
    # dataset described as something it is not, which is worse than one described as
    # nothing. The per-series guard in the action turns this into "series N skipped"
    # rather than a failed file.
    if index >= len(ome.images):
        raise IndexError(
            f"Series {index} has no OME image to describe it — the file declares "
            f"{len(ome.images)} image(s) for {index + 1}+ series. Refusing to borrow "
            f"another series' metadata."
        )
    image = ome.images[index]
    pixels = image.pixels
    instrument = find_instrument(ome, image)

    array, extra_types, packed_samples = load_series(img, index)

    # Canonical order — time, then channel, then space — is load-bearing rather than
    # cosmetic: the render axes are derived from the *position* of the spatial axes,
    # so a mis-ordered array does not fail to render, it renders wrongly.
    data = canonical(array, types=extra_types)
    dims = [str(d) for d in data.dims]

    # ONE LAYER PER CHANNEL, and the sample axis out of `load_series` is the only thing
    # allowed to say otherwise — see `packs_one_pixel`. This is the single condition under
    # which `author_channel_layers` may emit fewer layers than channels.
    declared = declared_samples_per_pixel(pixels)
    implicit_rgb = packs_one_pixel(
        packed_samples, declared, data.sizes["c"], data.dtype
    )
    if packed_samples and not implicit_rgb:
        logger.warning(
            "Series %s has a packed sample axis but %s %s sample(s) are no 8-bit RGB(A) "
            "picture; treating them as channels, one layer each.",
            index,
            data.sizes["c"],
            data.dtype,
        )
    if declared is not None and declared != data.sizes["c"]:
        # Logged, never acted on. If a file turns up where OME declares a packed pixel
        # the reader did not expose, this is the line that will say so — and the fix is
        # then to look at the reader, not to let the OME record decide.
        logger.info(
            "Series %s: the reader gave %s channel(s) where OME declares %s sample(s) "
            "per pixel. The reader decides; see `packs_one_pixel`.",
            index,
            data.sizes["c"],
            declared,
        )
    is_volume = data.sizes.get("z", 1) > 1

    # build_pyramid rather than dataset_arrays, because the level arrays themselves
    # are needed: the histogram of a very large channel is taken off the coarsest one.
    # scales_from returns levels 1..N only — `data` is level 0, and listing it in both
    # would upload the base twice.
    pyramid = build_pyramid(data, levels=levels, method=method, types=extra_types)
    scales = scales_from(pyramid, method)

    scale, scale_note = voxel_scale(pixels, dims)
    # Named and coloured as samples wherever the reader packed them, even when the series
    # is not collapsed into one RGB layer: a 16-bit colour camera's three samples are
    # still red, green and blue.
    specs = channel_specs(data, pyramid, image, packed_samples, is_volume)

    name = f"{file.name} - {text(image.name) or f'({index})'}"
    dataset = mikro.create_array_dataset(
        data=pyramid[0],
        scales=scales,
        name=name,
        axes=axes_for(
            data, types=dict.fromkeys(extra_types, "INDEX"), long_names=AXIS_LONG_NAMES
        ),
        anchors=[
            origin_anchor(data, file, index, image, instrument, scale_note),
            *channel_anchors(specs, image, instrument),
        ],
        # Where these arrays came from. Not a coordinate-graph edge — a file has no
        # space, so there is no map to state and `derivedFrom` would be the wrong
        # mechanism — but the link that makes the uploaded dataset traversable back to
        # the bytes it was read out of, and forward from the file to every series it
        # produced. The series identifier is OME's own image id rather than the
        # positional index, because that is what survives a reader that reorders.
        source_files=[
            SourceFileInput(
                file=file.id,
                series_identifier=str(image.id),
                # The pixels are the file's pixels: bioformats decodes, this converter
                # renames and reorders axes, and nothing rescales a value. So the
                # histograms and contrast limits computed here are true of the file too.
                value_relation=ValueRelation.IDENTICAL,
            )
        ],
    )

    if scale is None:
        # Uncalibrated: not placed in the shared world. `stage_group` stages it in its
        # own pixel grid instead.
        logger.info("Series %s: %s", index, scale_note)
        return Converted(index, name, dataset, specs, False, is_volume, implicit_rgb)

    world.register(dataset, scale=scale, **stage_offsets(pixels, dims))
    return Converted(index, name, dataset, specs, True, is_volume, implicit_rgb)


# ---------------------------------------------------------------------------
# The action
# ---------------------------------------------------------------------------


def _open(path: str) -> Any:
    """A BioImage on the bioformats reader.

    Imported here rather than at module scope: the first import pulls a JRE and the
    bioformats jars through cjdk and maven, which takes tens of seconds and wants the
    network once. At module scope that would block app registration and turn a missing
    JVM into a startup crash rather than a per-call error.

    The reader is named explicitly because plugin discovery may otherwise pick a
    thinner one for an OME-TIFF, and thin metadata is exactly what this converter has
    nothing to do with.
    """
    from bioio import BioImage

    return BioImage(path, reader=_start_bioformats())


def _start_bioformats() -> Any:
    """The bioformats reader class, with its JVM started.

    The flavour's Dockerfile makes the same calls at build time, so the Bio-Formats jars
    are baked into the image and a pod never needs the network to read a file. For that to hold, the run has to look where the build wrote:
    ``JGO_CACHE_DIR`` is jgo's own variable, but scyjava hands jgo a cache dir derived
    from ``$HOME`` instead, so it is passed on here. Without it set (a local run) the
    defaults stand.
    """
    global _READER
    if _READER is None:
        cache_dir = os.environ.get("JGO_CACHE_DIR")
        if cache_dir:
            import scyjava.config

            scyjava.config.set_cache_dir(cache_dir)

        # Importing the reader is what registers the Bio-Formats endpoints with scyjava.
        import bioio_bioformats
        import scyjava

        scyjava.start_jvm()
        _READER = bioio_bioformats.Reader
    return _READER


@app.action(
    name="Convert OME file to Datasets",
    description=(
        "Read a bioformats-supported file (czi, lif, nd2, ome-tiff, ...) and upload "
        "every series as a pyramided dataset, with its OME metadata, light path and "
        "device state, placed in a shared micrometer space and staged in a scene of its "
        "own. Tiles of a tile scan whose stitched image is in the file are uploaded but "
        "not staged."
    ),
    collections=["conversion"],
    logo=logo_of("convert_ome_file_to_datasets"),
)
def convert_ome_file_to_datasets(
    file: File,
    mikro: Mikro,
    task: Task,
    levels: int = 6,
    method: str = "max",
    stage_scene: bool = True,
    world_name: str = "",
) -> list[ArrayDataset]:
    """Convert every series of a bioformats file into a dataset.

    Returns a list rather than yielding. A mosaic scene is composed after its last tile,
    and a generator only gets that far if the consumer exhausts it — an abandoned one
    would leave a mosaic silently unstaged, or compose it from a `finally` during garbage
    collection, issuing a network call from a context where the transport may already be
    gone. The list costs nothing real: the arrays are
    uploaded inside ``create_array_dataset`` regardless of when the handle is returned, an
    ``ArrayDataset`` is a handful of bytes, and incremental feedback already comes from
    ``progress``.
    """
    assert file.store, "No file provided"

    task.progress(0, "Downloading file")
    # Under its own name, not `File.temporary()`: bioformats dispatches its reader on the
    # extension, so a .czi arriving as a bare store key would fail to open. Into a scratch
    # dir, though, never the working directory — `store.download(name)` writes relative to
    # the CWD and the `finally` below deletes what it wrote, which destroyed `test.ims` in
    # this directory on 2026-08-19. MIKRO_SCRATCH_DIR when set, TMPDIR otherwise; the
    # caller picks, because a tmpfs /tmp puts the download in RAM.
    workdir = tempfile.mkdtemp(
        prefix="ome-converter-", dir=os.environ.get("MIKRO_SCRATCH_DIR") or None
    )

    # The download itself is inside the try, so a failure part-way through it still
    # takes the scratch directory with it rather than leaving a stray one behind.
    try:
        path = file.store.download(os.path.join(workdir, os.path.basename(file.name)))
        task.progress(5, "Starting the bioformats reader")
        img = _open(path)
        ome = img.ome_metadata

        # A physical space is not something a dataset owns — it is an ordinary
        # coordinate system, belonging to nobody, that outlives every dataset put into
        # it. One per invocation.
        #
        # Known gap against the omero app, which took an optional `Stage` so two files
        # could share one: `CoordinateSystem` is not a registered rekuest structure, so
        # the object cannot be a parameter, and nothing suggests the space is
        # get-or-create by name. Two invocations therefore mint two disjoint worlds and
        # cannot share a scene. `world_name` only controls what the space is called.
        world = space_3d(
            mikro, world_name or f"{file.name} (micrometers)", unit="micrometer"
        )

        converted: dict[int, Converted] = {}
        failures: list[tuple[int, Exception]] = []

        scenes = list(img.scenes)

        # Decided up front, from the names alone, so that each scene can be composed as
        # soon as its last series has landed rather than after the whole file: a crash at
        # series 80 still leaves the scenes before it. A series past the end of the OME
        # record has no name to group by — and fails in `convert_series` anyway.
        names = [
            text(ome.images[index].name) if index < len(ome.images) else None
            for index in range(len(scenes))
        ]
        groups = scene_groups(names)
        last_of = {group[-1]: group for group in groups}
        unstaged = len(scenes) - sum(len(group) for group in groups)
        if stage_scene and unstaged:
            logger.info(
                "%s tile(s) have their stitched image in this file; they are uploaded "
                "and registered, but only the stitched image is staged.",
                unstaged,
            )

        for index, scene in enumerate(scenes):
            percent = 10 + int(85 * index / max(len(scenes), 1))
            task.progress(percent, f"Converting series {index + 1}/{len(scenes)}")
            try:
                converted[index] = convert_series(
                    mikro, img, ome, index, file, world, levels, method
                )
            except Exception as exc:  # one bad series must not cost the other 39
                logger.exception("Series %s (%s) failed", index, scene)
                failures.append((index, exc))

            group = last_of.get(index)
            if not stage_scene or group is None:
                continue
            entries = [converted[i] for i in group if i in converted]
            if not entries:
                continue
            title = (
                entries[0].name
                if len(group) == 1
                else f"{file.name} - {names[group[0]]} ({len(group)} tiles)"
            )
            try:
                stage_group(mikro, title, world, entries)
            except Exception:  # the datasets are uploaded; a scene is not worth them
                logger.exception("Could not stage %s", title)

        datasets = [converted[index].dataset for index in sorted(converted)]

        if failures and not datasets:
            raise RuntimeError(
                f"All {len(failures)} series failed; first: {failures[0][1]}"
            ) from failures[0][1]
        if failures:
            logger.warning(
                "%s of %s series failed: %s",
                len(failures),
                len(scenes),
                [i for i, _ in failures],
            )

        task.progress(100, f"Converted {len(datasets)} series")
        return datasets

    finally:
        # The whole scratch directory, so nothing the download created outlives the run —
        # and only ever that directory, which this converter made and owns.
        shutil.rmtree(workdir, ignore_errors=True)
        if os.path.exists(workdir):
            # Cleanup must never mask whatever actually went wrong above.
            logger.warning("Could not remove the scratch directory %s", workdir)


if __name__ == "__main__":
    run(app)
