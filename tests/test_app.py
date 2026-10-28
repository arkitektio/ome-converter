"""Offline checks of the converter's RGB decision, contrast windows and layer authoring.

No backend, no bioformats, no JRE: the reader is a stand-in for the two attributes
`load_series` touches, and the Mikro client is a recorder, so the assertions land on the
calls that would reach the server rather than on a property that predicts them.

Everything here is about one rule:

    ONE LAYER PER CHANNEL. The only exception is a packed RGB(A) pixel.
"""

import numpy as np
import pytest
import xarray as xr
from ome_types.model import Channel, Image, Pixels

import app as M


class _RecordingMikro:
    """Stands in for the Mikro client: records every mutation the layer author makes."""

    def __init__(self, calls: list) -> None:
        self._calls = calls

    def __getattr__(self, name: str):
        return lambda **kw: self._calls.append((name, kw))


def test_the_app_declares_its_action():
    """One action, taking a mikro file and handed a mikro client."""
    assert M.app.services == ["mikro"]
    declaration = M.app.registry.to_implement_agent_input().model_dump(
        mode="json", exclude_none=True
    )
    [implementation] = declaration["implementations"]
    assert implementation["interface"] == "convert_ome_file_to_datasets"
    args = {arg["key"]: arg for arg in implementation["definition"]["args"]}
    assert args["file"]["identifier"] == "@mikro/file"
    # The client and the task are injected, not ports.
    assert "mikro" not in args and "task" not in args


# --- OME's own declaration ----------------------------------------------------------
#
# SamplesPerPixel is a statement about a pixel; a channel count is not. Only the first
# may decide RGB.


def pixels_with(channel_samples, size_c):
    return Pixels(
        id="Pixels:0",
        dimension_order="XYCZT",
        type="uint8",
        size_x=8,
        size_y=8,
        size_z=1,
        size_c=size_c,
        size_t=1,
        channels=[
            Channel(id=f"Channel:{i}", samples_per_pixel=s)
            for i, s in enumerate(channel_samples)
        ],
    )


@pytest.mark.parametrize(
    ("samples", "size_c", "declared"),
    [
        ([3], 3, 3),
        ([4], 4, 4),
        # The exact false positive of the old heuristic: one OME channel, but it says
        # nothing about samples, while the array carries three real dyes.
        ([None], 3, None),
        ([1], 1, 1),
        # Three channels are three dyes whatever any single one of them declares.
        ([3, 1, 1], 3, None),
    ],
)
def test_declared_samples_per_pixel(samples, size_c, declared):
    assert M.declared_samples_per_pixel(pixels_with(samples, size_c)) == declared


# --- the reader's sample axis -------------------------------------------------------


class FakeImage:
    """Duck-types what load_series touches: .set_scene() and .xarray_dask_data."""

    def __init__(self, dims, shape):
        self._array = xr.DataArray(np.zeros(shape, dtype=np.uint8), dims=dims)

    def set_scene(self, scene):
        pass

    @property
    def xarray_dask_data(self):
        return self._array


def load(dims, shape):
    array, _, packed = M.load_series(FakeImage(dims, shape), 0)
    return int(array.sizes["c"]), packed


TCZYXS = ("T", "C", "Z", "Y", "X", "S")


@pytest.mark.parametrize(
    ("dims", "shape", "expected"),
    [
        # A JPEG-style photograph: one metadata channel, three samples behind it.
        (TCZYXS, (1, 1, 1, 8, 8, 3), (3, True)),
        (TCZYXS, (1, 1, 1, 8, 8, 4), (4, True)),
        # Any other extent behind an "S" axis is some other packing, not a picture.
        (TCZYXS, (1, 1, 1, 8, 8, 2), (2, False)),
        # Three dyes with no sample axis at all.
        (("T", "C", "Z", "Y", "X"), (1, 3, 1, 8, 8), (3, False)),
        # Samples alongside real channels: the reader's rule is to take sample 0, and
        # that is not a photograph either.
        (TCZYXS, (1, 2, 1, 8, 8, 3), (2, False)),
    ],
)
def test_the_sample_axis_decides_packing(dims, shape, expected):
    assert load(dims, shape) == expected


# --- which series, when the names collide --------------------------------------------
#
# A LIF tile scan names every tile alike, and bioio resolves a scene *name* to its first
# match — or does nothing when the current scene already has that name. Modelled here as
# bioio behaves, so a `load_series` that selects by name reads tile 0 three times.


class TileScan:
    """Three series sharing one name, each filled with its own index."""

    scenes = ("R 1", "R 1", "R 1")

    def __init__(self):
        self._current = 0

    def set_scene(self, scene):
        if isinstance(scene, str):
            if self.scenes[self._current] == scene:
                return
            scene = self.scenes.index(scene)
        self._current = scene

    @property
    def xarray_dask_data(self):
        return xr.DataArray(
            np.full((1, 1, 1, 4, 4), self._current, dtype=np.uint8),
            dims=("T", "C", "Z", "Y", "X"),
        )


def test_same_named_series_are_read_by_position():
    tiles = TileScan()
    assert [int(M.load_series(tiles, i)[0].max()) for i in range(3)] == [0, 1, 2]


def test_a_scene_name_is_refused_not_resolved():
    # That is how 98 tiles were lost: the action passed `img.scenes[i]` where it meant `i`.
    with pytest.raises(TypeError):
        M.load_series(TileScan(), "R 1")


# --- which series get a scene --------------------------------------------------------
#
# The names are what Bio-Formats reports for real LIF files, not what the LIF XML says.
# One scene per series — except that a stitched image is the picture its tiles make, so
# those tiles are staged nowhere, and tiles with no stitched image are one mosaic.

JMJ = "JMJ_human_slice_surgery_494Calcein_550EthBr_4"


@pytest.mark.parametrize(
    ("names", "groups"),
    [
        # 98 tiles and their stitch: the stitch alone.
        (["R 1"] * 98 + ["R 1_Merged"], [[98]]),
        # "merged" with no separator at all.
        ([JMJ] * 23 + [JMJ + "merged"], [[23]]),
        # An export holding only the stitch.
        (["R1_Merged"], [[0]]),
        # Positions and their snapshots share a slide, not a scene.
        (
            [f"TileScan 1/Position {i}" for i in (1, 2, 3)]
            + [f"Position {i} Snapshot1" for i in (1, 2, 3)],
            [[0], [1], [2], [3], [4], [5]],
        ),
        # A tile scan with no stitch: the tiles are the picture, one mosaic.
        (["R 1"] * 3, [[0, 1, 2]]),
        # Two missing names are not a shared one.
        ([None, None], [[0], [1]]),
        # Groups come in order of their first series, tiles skipped wherever they sit.
        (["R 1", "Overview", "R 1", "R 1 Merged", "R 2", "R 2"], [[1], [3], [4, 5]]),
    ],
)
def test_scene_groups(names, groups):
    assert M.scene_groups(names) == groups


# --- the decision `convert_series` makes from those two ------------------------------
#
# The real function, not a copy of its expression: `packs_one_pixel` exists as a
# function precisely so this can call it rather than restate it and drift.


@pytest.mark.parametrize(
    ("packed", "declared", "channels", "dtype", "expected"),
    [
        (True, None, 3, np.uint8, True),
        (True, 4, 4, np.uint8, True),
        (False, None, 3, np.uint8, False),
        (True, None, 6, np.uint8, False),
        (False, 1, 3, np.uint8, False),
        # SamplesPerPixel can only ever fire in the collapse direction, and a single OME
        # channel declaring 3 samples over an array of three dyes is the exact
        # partial-OME shape that broke the old heuristic. The reader decides.
        (False, 3, 3, np.uint8, False),
        (False, 4, 4, np.uint8, False),
        # An RGB layer is drawn unstretched over 0..255: a deeper sample axis is a colour
        # camera's acquisition and gets a layer and a window per sample instead.
        (True, 3, 3, np.uint16, False),
        (True, None, 3, np.float32, False),
    ],
)
def test_packs_one_pixel(packed, declared, channels, dtype, expected):
    assert M.packs_one_pixel(packed, declared, channels, dtype) is expected


# --- the window a layer opens with ---------------------------------------------------
#
# Sparse fluorescence: 0.5 % of every z-column is signal, at a different depth per column.
# By voxel count that is background, so the stack's p99 is the background level — while
# the projection the volume layer draws is signal in every pixel.


@pytest.fixture
def sparse_stack():
    stack = np.full((200, 32, 32), 2, dtype=np.uint8)
    depth = np.arange(32 * 32).reshape(32, 32) % 200
    np.put_along_axis(
        stack,
        depth[None],
        np.linspace(100, 200, 32 * 32).reshape(1, 32, 32).astype(np.uint8),
        axis=0,
    )
    assert float(np.percentile(stack, 99)) == 2.0
    return stack


def window_spec(array, dims, levels=1):
    """The spec `channel_specs` gives channel 0 of `array`, through the real function."""
    data = xr.DataArray(array[None], dims=("c", *dims))
    pyramid = [data] + [data.coarsen(y=2**n, x=2**n).max() for n in range(1, levels)]
    image = Image(id="Image:0", pixels=pixels_with([None], 1))
    return M.channel_specs(data, pyramid, image, False, "z" in dims)[0]


def test_a_volume_window_is_taken_from_its_projection(sparse_stack):
    spec = window_spec(sparse_stack, ("z", "y", "x"))
    assert spec.clim[0] > 99 and spec.clim[1] > 190
    # About 1 % of the projection saturated, not all of it.
    assert float((sparse_stack.max(axis=0) >= spec.clim[1]).mean()) < 0.02
    # While the anchor's histogram stays the voxels' own.
    assert spec.histogram.p99 == 2.0


def test_an_over_budget_stack_still_gets_the_exact_projection_window(
    sparse_stack, monkeypatch
):
    exact = window_spec(sparse_stack, ("z", "y", "x"))
    monkeypatch.setattr(M, "HISTOGRAM_BUDGET_BYTES", sparse_stack.nbytes // 2)
    pooled = window_spec(sparse_stack, ("z", "y", "x"), levels=3)
    assert pooled.clim == exact.clim
    # And its histogram from the finest level that fits, not the coarsest.
    assert sum(pooled.histogram.histogram) == sparse_stack.size / 4


def test_display_window():
    assert M.display_window(np.arange(101.0).reshape(1, 101)) == (1.0, 99.0)
    # Over 99 % one value: the percentiles coincide, and a window of no width draws
    # nothing.
    dots = np.zeros((64, 64), dtype=np.uint16)
    dots[0, :8] = 500
    assert M.display_window(dots) == (0.0, 500.0)
    assert M.display_window(np.full((4, 4), 7.0)) == (7.0, 8.0)


# --- the layers themselves, not a property that predicts them -----------------------


class FakeId:
    id = "fake"


class FakeLens:
    def __init__(self, id):
        self.id = id


class FakeDataset:
    """A dataset as far as the layer author reads one: its axes, and its lenses.

    A lens' id spells out what it pins — ``"lens"`` for the whole lens, ``"lens:c=2"``
    for channel 2 — so a recorded call can be asked which channel it draws.
    """

    axis_names = ("c", "z", "y", "x")

    def lens(self, **selections):
        for axis in selections:
            if axis not in self.axis_names:
                raise ValueError(f"Invalid axis {axis!r} for axes {list(self.axis_names)}")
        if not selections:
            return FakeLens("lens")
        return FakeLens("lens:" + ",".join(f"{a}={v}" for a, v in selections.items()))


def pinned_channel(lens_id):
    """The channel a fake lens id pins, or None for the whole lens and for no lens."""
    if not isinstance(lens_id, str) or not lens_id.startswith("lens:"):
        return None
    pins = dict(pin.split("=") for pin in lens_id[len("lens:") :].split(","))
    return int(pins["c"]) if "c" in pins else None


def fake_specs(n):
    return [
        M.ChannelSpec(
            index=i,
            label=f"ch{i}",
            color=(i * 60 % 256, 0, 0, 255),
            histogram=M.ValueHistogramInput(
                histogram=[1.0], bins=[0.0, 1.0], min=0.0, max=1.0, p1=0.0, p99=1.0
            ),
            # Deliberately not the histogram's p1/p99, so a check can tell which of the
            # two a layer was given.
            clim=(10.0 + i, 20.0 + i),
            ome_channel=None,
        )
        for i in range(n)
    ]


def layers_for(specs, implicit_rgb, is_volume=True):
    calls = []
    made = M.author_channel_layers(
        _RecordingMikro(calls), FakeDataset(), FakeId(), specs, is_volume, implicit_rgb
    )
    assert made == len(calls), "layer count must match the calls made"
    return calls


def selected_channels(call):
    """Every channel index this layer covers, whichever kind of layer it is.

    An unrecognised call is an error rather than an empty list: otherwise the converter
    moving to a mutation this walker does not know would make every check below pass
    vacuously against nothing at all.
    """
    name, kw = call
    if name in ("create_intensity_layer", "create_volume_layer"):
        index = pinned_channel(kw.get("lens"))
        assert index is not None, f"{name} call's lens pins no channel: {kw!r}"
        return [index]
    if name == "create_rgb_layer":
        found = [pinned_channel(kw.get(k)) for k in ("red_lens", "green_lens", "blue_lens")]
        assert None not in found, f"{name} call is missing a component lens: {kw!r}"
        return found
    if name == "create_layer":
        found = []

        def walk(node):
            if node is None:
                return
            if node.get("kind") == "channel":
                index = pinned_channel(node.get("lens"))
                assert index is not None, f"channel node's lens pins no channel: {node!r}"
                found.append(index)
            for child in node.get("children") or ():
                walk(child)

        graph = kw.get("render_graph")
        assert graph is not None, f"create_layer call carries no render_graph: {kw!r}"
        walk(graph.model_dump()["root"])
        assert found, f"create_layer graph has no channel node: {kw!r}"
        return found
    raise AssertionError(f"unrecognised layer mutation {name!r}; teach the walker about it")


def test_a_z_stack_is_one_volume_layer_per_channel():
    four = layers_for(fake_specs(4), False)
    assert len(four) == 4
    assert {n for n, _ in four} == {"create_volume_layer"}
    assert sorted(i for c in four for i in selected_channels(c)) == [0, 1, 2, 3]
    # The colour reaches the layer as a field, not a graph.
    assert sorted(tuple(kw["color"]) for _, kw in four) == sorted(
        tuple(s.color) for s in fake_specs(4)
    )
    # Each layer opens on its spec's window, not the histogram's percentiles.
    assert sorted((kw["clim_min"], kw["clim_max"]) for _, kw in four) == [
        (10.0 + i, 20.0 + i) for i in range(4)
    ]


def test_a_packed_rgb_z_stack_is_one_render_graph_layer():
    rgb = layers_for(fake_specs(3), True)
    assert len(rgb) == 1
    # An RGB layer carries no projection mode, so this case keeps its render graph.
    assert rgb[0][0] == "create_layer"
    assert sorted(selected_channels(rgb[0])) == [0, 1, 2]
    assert len(layers_for(fake_specs(4), True)) == 1


@pytest.mark.parametrize("n", [2, 5])
def test_implicit_rgb_over_a_non_pixel_channel_count_is_refused(n):
    # The backstop: a caller asserting implicit_rgb over a channel count that cannot be
    # a pixel must not swallow channels.
    assert len(layers_for(fake_specs(n), True)) == n


def test_flat_channels_are_intensity_layers():
    flat = layers_for(fake_specs(4), False, is_volume=False)
    assert {n for n, _ in flat} == {"create_intensity_layer"}
    assert sorted(i for c in flat for i in selected_channels(c)) == [0, 1, 2, 3]
    # One plane to show: a MIP over it would collapse an axis that is not there.
    assert [kw.get("mode") for _, kw in flat] == [None] * 4


def test_a_flat_packed_rgb_pixel_is_one_rgb_layer():
    flat_rgb = layers_for(fake_specs(3), True, is_volume=False)
    assert [n for n, _ in flat_rgb] == ["create_rgb_layer"]
    assert selected_channels(flat_rgb[0]) == [0, 1, 2]
    # Display-referred already: stretching it would change the colours it was authored
    # with.
    kw = flat_rgb[0][1]
    assert (kw["clim_min"], kw["clim_max"]) == (0.0, 255.0)
    # The server forces NORMAL and takes no `blending`: a photograph sits over what is
    # beneath it.
    assert "blending" not in kw


def test_rgba_labels_and_colours():
    assert [M.channel_label(None, i, True) for i in range(4)] == [
        "Red",
        "Green",
        "Blue",
        "Alpha",
    ]
    # Alpha is not painted the same red as channel 0.
    assert [M.channel_color(None, i, True, 4) for i in range(4)] == [
        (255, 0, 0, 255),
        (0, 255, 0, 255),
        (0, 0, 255, 255),
        (255, 255, 255, 255),
    ]
