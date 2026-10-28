"""OME metadata: reading an OME document into mikro inputs.

The OME-TIFF/CZI/LIF path is an OME document by definition. (The same module serves
the OBF converter in `testing/`, whose files carry one alongside the stacks.)

Everything here walks the `ome_types` object graph, so it lives beside its callers
rather than in `mikro`: the client has no `ome-types` dependency and no business
acquiring one. What *is* generic — the setting/device builders, the colour helpers, the
light-port pair — moved into `mikro.metadata`, `mikro.render` and
`mikro.lightpath` instead, and is imported back here.

Two conventions worth stating once, because both are load-bearing:

* **A missing unit means micrometers.** That is OME's schema default for every physical
  size, so `to_micrometer` returns the value unchanged rather than refusing it.
* **An unreadable unit returns ``None``, not the number.** That is what makes an
  uncalibrated file distinguishable from a genuinely-micrometer one, and that
  distinction is the entire input to the registration decision in `voxel_scale`.
"""

from __future__ import annotations

from typing import Any, Sequence

from mikro.api.schema import (
    BeamSplitterElementInput,
    BeamStateInput,
    ChannelKind,
    DetectorElementInput,
    FilterElementInput,
    FilterKind,
    LampElementInput,
    LaserElementInput,
    LightEdgeInput,
    LightpathGraphInput,
    ObjectiveElementInput,
    ObjectiveImmersion,
    OpticalElementInput,
    OptikitStateInput,
    PinholeElementInput,
    PortRole,
    PulseKind,
    SampleElementInput,
    SpectrumInput,
)
from mikro.inputs.lightpath import in_out as _in_out
from mikro.inputs.lightpath import port as _port
from converter_metadata import device, devices, setting, text
from mikro.inputs.render import rgba

# OME's length vocabulary, in micrometers per unit. Deliberately partial: `pixel` and
# `reference frame` are not lengths, and the exotic members (parsec, light year) have no
# business in a microscope file. Anything absent resolves to None — "not a length I can
# read" — which is what the registration decision keys off.
UM_PER_LENGTH_UNIT: dict[str, float] = {
    "m": 1e6,
    "dm": 1e5,
    "cm": 1e4,
    "mm": 1e3,
    "µm": 1.0,
    "um": 1.0,
    "nm": 1e-3,
    "pm": 1e-6,
    "fm": 1e-9,
    "Å": 1e-4,
    "in": 25400.0,
    "meter": 1e6,
    "centimeter": 1e4,
    "millimeter": 1e3,
    "micrometer": 1.0,
    "micron": 1.0,
    "nanometer": 1e-3,
    "picometer": 1e-6,
    "angstrom": 1e-4,
    "inch": 25400.0,
}


# The quantity scalars are pint-backed, and pint does not read OME's degree symbols or
# its spaced mmHg. Everything else in OME's vocabulary is an SI-prefixed symbol pint
# already knows, so this map is the whole of the translation.
PINT_UNIT_FIXES = {"°C": "degC", "°F": "degF", "°R": "degR", "mm Hg": "mmHg"}


# What a channel is drawn in when the file does not say. Fluorescence emission maps to
# roughly the colour the eye would have seen down the eyepiece, which is the one guess
# about a channel that a microscopist can check at a glance.
COLOR_BY_EMISSION = (
    (400, (150, 0, 255)),
    (450, (75, 0, 255)),
    (490, (0, 130, 255)),
    (520, (0, 255, 200)),
    (560, (0, 255, 0)),
    (590, (255, 255, 0)),
    (625, (255, 150, 0)),
    (700, (255, 0, 0)),
)


FAR_RED = (255, 70, 70)


IMMERSION_BY_OME = {
    "Oil": ObjectiveImmersion.OIL,
    "Water": ObjectiveImmersion.WATER,
    "WaterDipping": ObjectiveImmersion.WATER_DIPPING,
    "Air": ObjectiveImmersion.AIR,
    "Multi": ObjectiveImmersion.MULTI,
    "Glycerol": ObjectiveImmersion.GLYCEROL,
}


FILTER_KIND_BY_OME = {
    "Dichroic": FilterKind.DICHROIC,
    "LongPass": FilterKind.LONG_PASS,
    "ShortPass": FilterKind.SHORT_PASS,
    "BandPass": FilterKind.BAND_PASS,
    "MultiPass": FilterKind.MULTI_PASS,
    "NeutralDensity": FilterKind.NEUTRAL_DENSITY,
    "Tuneable": FilterKind.TUNEABLE,
}


PULSE_KIND_BY_OME = {
    "CW": PulseKind.CW,
    "Single": PulseKind.SINGLE,
    "QSwitched": PulseKind.QSWITCHED,
    "Repetitive": PulseKind.REPETITIVE,
    "ModeLocked": PulseKind.MODE_LOCKED,
}


def unit_symbol(unit: Any) -> str | None:
    """The bare symbol of an OME unit, whether it arrives as an enum or a string."""
    if unit is None:
        return None
    return str(getattr(unit, "value", unit))


def to_micrometer(value: Any, unit: Any) -> float | None:
    """An OME length in micrometers, or None when it is not a length.

    Returning None rather than the number unchanged for an unrecognised unit is what
    makes an uncalibrated file distinguishable from a genuinely-micrometer one. That
    distinction is the whole input to the registration decision in :func:`voxel_scale`.
    """
    if value is None:
        return None
    symbol = unit_symbol(unit)
    if symbol is None:
        # OME's schema default for every physical size is micrometers, and a file that
        # states a size while omitting the unit means that default.
        return float(value)
    factor = UM_PER_LENGTH_UNIT.get(symbol)
    if factor is None:
        return None
    return float(value) * factor


def um(value: Any, unit: Any) -> str | None:
    """An OME length as a Length scalar string, always in micrometers.

    Converting rather than passing the unit through keeps pint out of the picture for
    lengths entirely — OME can spell a length in units pint does not know, and one such
    value would otherwise fail the whole upload.
    """
    micrometers = to_micrometer(value, unit)
    if micrometers is None:
        return None
    return f"{micrometers:g} um"


def nm(value: Any, unit: Any) -> str | None:
    """An OME wavelength as a Length scalar string, in nanometers.

    The same conversion as :func:`um` with a different presentation unit, because a
    wavelength is universally spoken in nanometers: ``"488 nm"`` is a laser line,
    ``"0.488 µm"`` is the same fact written so as to be unrecognisable.
    """
    micrometer_value = to_micrometer(value, unit)
    if micrometer_value is None:
        return None
    return f"{micrometer_value * 1000:g} nm"


def qty(value: Any, unit: Any, default_symbol: str | None = None) -> str | None:
    """A non-length OME quantity as a scalar string.

    OME's power, time, frequency, voltage and pressure vocabularies are SI-prefixed
    symbols pint already reads, so these pass through with only the degree-symbol fixes
    applied.
    """
    if value is None:
        return None
    symbol = unit_symbol(unit) or default_symbol
    if symbol is None:
        return None
    return f"{value:g} {PINT_UNIT_FIXES.get(symbol, symbol)}"


def ome_rgba(channel: Any) -> tuple[int, int, int, int] | None:
    """OME's own colour for a channel, as RGBA, or None if it said nothing useful.

    OME writes RGB as 0-255 but alpha as a 0-1 fraction, so the alpha is rescaled rather
    than passed through — an opaque channel arrives as 1 and would otherwise become very
    nearly transparent.
    """
    color = getattr(channel, "color", None)
    if color is None:
        return None
    try:
        components = color.as_rgb_tuple(alpha=True)
    except Exception:
        return None

    rgb = tuple(int(c) for c in components[:3])
    # White is OME's schema default, so it is what a file says when it was never asked.
    # Treating it as a choice would paint every channel in every such file white.
    if rgb == (255, 255, 255):
        return None

    alpha = components[3] if len(components) > 3 else 1.0
    return rgba(rgb, round(float(alpha) * 255))


def find_instrument(ome: Any | None, image: Any | None) -> Any | None:
    """The instrument this image points at, or None."""
    if ome is None or image is None:
        return None
    ref = getattr(image, "instrument_ref", None)
    if ref is None:
        return None
    found = [i for i in ome.instruments if i.id == ref.id]
    return found[0] if len(found) == 1 else None


def find_objective(instrument: Any | None, image: Any | None) -> Any | None:
    settings_ref = getattr(image, "objective_settings", None) if image else None
    if instrument is None or settings_ref is None:
        return None
    found = [o for o in instrument.objectives if o.id == settings_ref.id]
    return found[0] if found else None


def find_detector(instrument: Any | None, channel: Any) -> Any | None:
    settings_ref = getattr(channel, "detector_settings", None)
    if instrument is None or settings_ref is None:
        return None
    found = [d for d in instrument.detectors if d.id == settings_ref.id]
    return found[0] if found else None


def light_sources(instrument: Any | None) -> list[Any]:
    """Every emitter the instrument declares, in a stable order."""
    if instrument is None:
        return []
    return [
        *instrument.lasers,
        *instrument.arcs,
        *instrument.filaments,
        *instrument.light_emitting_diodes,
        *instrument.generic_excitation_sources,
    ]


def channel_sources(instrument: Any | None, channel: Any) -> list[Any]:
    """The emitters that lit *this* channel.

    ``LightSourceSettings`` names the one source the channel actually used, and using it
    is the difference between a graph that describes the acquisition and one that
    asserts every laser on the bench fired into every channel. Only when the channel
    names none does the instrument's full list stand in — at that point the file
    genuinely does not say which, and listing what was available is the honest reading.
    """
    every = light_sources(instrument)
    settings_ref = getattr(channel, "light_source_settings", None)
    if settings_ref is None:
        return every
    named = [s for s in every if s.id == settings_ref.id]
    return named or every


def resolve_filters(instrument: Any | None, refs: Sequence[Any]) -> list[Any]:
    if instrument is None:
        return []
    by_id = {f.id: f for f in instrument.filters}
    return [by_id[r.id] for r in refs if r.id in by_id]


def resolve_dichroic(instrument: Any | None, light_path: Any | None) -> Any | None:
    if instrument is None or light_path is None or light_path.dichroic_ref is None:
        return None
    found = [d for d in instrument.dichroics if d.id == light_path.dichroic_ref.id]
    return found[0] if found else None


def _band(transmittance_range: Any | None) -> SpectrumInput | None:
    """A filter's passband, as a spectrum on the port it leaves by.

    ``FilterElementInput`` has no band fields at all — a passband is a property of the
    light coming out, not of the glass — so this is where OME's TransmittanceRange lands.
    """
    if transmittance_range is None:
        return None
    low = nm(transmittance_range.cut_in, transmittance_range.cut_in_unit)
    high = nm(transmittance_range.cut_out, transmittance_range.cut_out_unit)
    if low is None or high is None:
        return None
    return SpectrumInput(min=low, max=high)


def _source_element(source: Any, index: int) -> OpticalElementInput:
    """One emitter.

    A laser without a wavelength cannot be a ``LaserElementInput``: ``nominal_wavelength``
    is the member's one required field. Rather than invent one, such a source is modelled
    as a lamp — which says less, and says nothing false.
    """
    element_id = f"source-{index}"
    label = (
        " ".join(
            p
            for p in (text(source.manufacturer), text(source.model), text(source.id))
            if p
        )
        or f"Light source {index}"
    )
    ports = [_port("out", "output", PortRole.OUTPUT)]
    wavelength = nm(
        getattr(source, "wavelength", None), getattr(source, "wavelength_unit", None)
    )

    if wavelength is not None:
        return LaserElementInput(
            id=element_id,
            label=label,
            manufacturer=text(source.manufacturer),
            model=text(source.model),
            serial_number=text(source.serial_number),
            nominal_wavelength=wavelength,
            power=qty(
                getattr(source, "power", None),
                getattr(source, "power_unit", None),
                "mW",
            ),
            channel=ChannelKind.FREE_SPACE,
            laser_medium=text(getattr(source, "laser_medium", None)),
            pulse_kind=PULSE_KIND_BY_OME.get(text(getattr(source, "pulse", None)) or ""),
            repetition_rate=qty(
                getattr(source, "repetition_rate", None),
                getattr(source, "repetition_rate_unit", None),
                "Hz",
            ),
            has_pockels_cell=getattr(source, "pockel_cell", None),
            ports=ports,
        )

    return LampElementInput(
        id=element_id,
        label=label,
        manufacturer=text(source.manufacturer),
        model=text(source.model),
        serial_number=text(source.serial_number),
        channel=ChannelKind.FREE_SPACE,
        lamp_type=text(getattr(source, "type", None)) or type(source).__name__,
        ports=ports,
    )


def channel_optikit(
    channel: Any,
    index: int,
    label: str,
    instrument: Any | None,
) -> OptikitStateInput | None:
    """The hardware state of one channel.

    No stage and no temperature: those are stack facts and already sit on the origin
    anchor. Repeating them here would be a second copy that can only drift.
    """
    detector_settings = getattr(channel, "detector_settings", None)
    detector = find_detector(instrument, channel)
    source_settings = getattr(channel, "light_source_settings", None)

    colour = getattr(channel, "color", None)
    colour_hex = None
    if colour is not None:
        try:
            colour_hex = "#%02X%02X%02X" % colour.as_rgb_tuple()[:3]
        except Exception:
            colour_hex = text(colour)

    state_devices = devices(
        device(
            f"channel-{index}",
            "channel",
            setting("channel-id", string=text(channel.id)),
            setting("name", string=text(channel.name) or label),
            setting("fluor", string=text(channel.fluor)),
            # The colour OME assigns this channel, kept as a fact independently of what
            # is drawn with it. `channel_color` turns it into the layer's transfer
            # function, but that is a rendering decision and can be overridden; this is
            # what the file said.
            setting("color", string=colour_hex),
            setting("illumination-type", string=text(channel.illumination_type)),
            setting("acquisition-mode", string=text(channel.acquisition_mode)),
            setting("contrast-method", string=text(channel.contrast_method)),
            setting(
                "excitation-wavelength",
                quantity=nm(
                    channel.excitation_wavelength, channel.excitation_wavelength_unit
                ),
            ),
            setting(
                "emission-wavelength",
                quantity=nm(
                    channel.emission_wavelength, channel.emission_wavelength_unit
                ),
            ),
            setting("nd-filter", number=channel.nd_filter),
            setting("pockel-cell-setting", number=channel.pockel_cell_setting),
            setting("samples-per-pixel", number=channel.samples_per_pixel),
            setting(
                "pinhole-size",
                quantity=um(channel.pinhole_size, channel.pinhole_size_unit),
            ),
        ),
        device(
            f"detector-{index}",
            "detector",
            setting("detector-id", string=text(getattr(detector_settings, "id", None))),
            setting("model", string=text(getattr(detector, "model", None))),
            setting(
                "manufacturer", string=text(getattr(detector, "manufacturer", None))
            ),
            setting(
                "serial-number", string=text(getattr(detector, "serial_number", None))
            ),
            setting("detector-type", string=text(getattr(detector, "type", None))),
            setting(
                "gain",
                number=getattr(detector_settings, "gain", None)
                if getattr(detector_settings, "gain", None) is not None
                else getattr(detector, "gain", None),
            ),
            setting(
                "voltage",
                quantity=qty(
                    getattr(detector_settings, "voltage", None),
                    getattr(detector_settings, "voltage_unit", None),
                    "V",
                ),
            ),
            setting("offset", number=getattr(detector_settings, "offset", None)),
            setting("zoom", number=getattr(detector_settings, "zoom", None)),
            setting(
                "read-out-rate",
                quantity=qty(
                    getattr(detector_settings, "read_out_rate", None),
                    getattr(detector_settings, "read_out_rate_unit", None),
                    "MHz",
                ),
            ),
            setting(
                "binning", string=text(getattr(detector_settings, "binning", None))
            ),
            setting(
                "amplification-gain",
                number=getattr(detector, "amplification_gain", None),
            ),
        ),
        device(
            f"light-source-{index}",
            "light-source",
            setting("source-id", string=text(getattr(source_settings, "id", None))),
            setting(
                "attenuation", number=getattr(source_settings, "attenuation", None)
            ),
            setting(
                "wavelength",
                quantity=nm(
                    getattr(source_settings, "wavelength", None),
                    getattr(source_settings, "wavelength_unit", None),
                ),
            ),
        ),
    )

    if not state_devices:
        return None
    return OptikitStateInput(devices=state_devices)


def channel_lightpath(
    channel: Any,
    image: Any | None,
    instrument: Any | None,
) -> LightpathGraphInput | None:
    """The light path for one channel, as far as OME describes it.

    Elements and edges are built from one ``hops`` list rather than written out, because
    nothing validates that an edge names a port its element actually declares — a typo
    would only surface server-side. Generating both from the same list is the only local
    guarantee there is.
    """
    if instrument is None or image is None:
        return None

    light_path = getattr(channel, "light_path", None)
    objective = find_objective(instrument, image)
    detector = find_detector(instrument, channel)
    sources = channel_sources(instrument, channel)
    excitation = resolve_filters(
        instrument, getattr(light_path, "excitation_filters", [])
    )
    emission = resolve_filters(instrument, getattr(light_path, "emission_filters", []))
    dichroic = resolve_dichroic(instrument, light_path)

    emission_band = _band(emission[0].transmittance_range if emission else None) or (
        SpectrumInput(
            min=nm(channel.emission_wavelength, channel.emission_wavelength_unit),
            max=nm(channel.emission_wavelength, channel.emission_wavelength_unit),
        )
        if channel.emission_wavelength is not None
        else None
    )

    elements: list[OpticalElementInput] = []
    hops: list[tuple[str, str, str, str, BeamStateInput | None, str]] = []

    excitation_beam = BeamStateInput(
        wavelength=nm(
            channel.excitation_wavelength, channel.excitation_wavelength_unit
        ),
        mode_hint=text(channel.illumination_type),
    )
    emission_beam = BeamStateInput(
        wavelength=nm(channel.emission_wavelength, channel.emission_wavelength_unit),
        mode_hint=text(channel.acquisition_mode),
    )

    # The immersion the excitation and emission cross at the objective's front lens.
    immersion_medium = text(
        getattr(getattr(image, "objective_settings", None), "medium", None)
    )
    immersion = immersion_medium.upper() if immersion_medium else "AIR"

    # --- the excitation arm -------------------------------------------------
    #
    # One source needs no combiner at all — the overwhelmingly common case, and one
    # fewer fictitious element — and several get a beam splitter run backwards, with one
    # declared input port each.
    tail: tuple[str, str] | None = None

    for index, source in enumerate(sources):
        elements.append(_source_element(source, index))

    if len(sources) == 1:
        tail = ("source-0", "out")
    elif len(sources) > 1:
        combiner_ports = [
            _port(f"in-{i}", f"source {i}", PortRole.INPUT) for i in range(len(sources))
        ]
        combiner_ports.append(_port("out", "combined", PortRole.OUTPUT))
        elements.append(
            BeamSplitterElementInput(
                id="combiner",
                label="Source combiner",
                ports=combiner_ports,
            )
        )
        for i in range(len(sources)):
            hops.append(
                (f"source-{i}", "out", "combiner", f"in-{i}", excitation_beam, "AIR")
            )
        tail = ("combiner", "out")

    for index, filter_obj in enumerate(excitation):
        element_id = f"ex-filter-{index}"
        elements.append(
            FilterElementInput(
                id=element_id,
                label=text(filter_obj.model)
                or text(filter_obj.id)
                or f"Excitation filter {index}",
                manufacturer=text(filter_obj.manufacturer),
                model=text(filter_obj.model),
                serial_number=text(filter_obj.serial_number),
                filter_kind=FILTER_KIND_BY_OME.get(text(filter_obj.type) or ""),
                transmittance=getattr(
                    filter_obj.transmittance_range, "transmittance", None
                ),
                description=text(getattr(filter_obj, "filter_wheel", None)),
                ports=_in_out(_band(filter_obj.transmittance_range)),
            )
        )
        if tail is not None:
            hops.append((*tail, element_id, "in", excitation_beam, "AIR"))
        tail = (element_id, "out")

    if dichroic is not None:
        elements.append(
            BeamSplitterElementInput(
                id="dichroic",
                label=text(dichroic.model) or text(dichroic.id) or "Dichroic",
                manufacturer=text(dichroic.manufacturer),
                model=text(dichroic.model),
                serial_number=text(dichroic.serial_number),
                # r_fraction / t_fraction are deliberately omitted: OME records neither,
                # and a plausible number is still an invented one.
                ports=[
                    _port("ex-in", "excitation in", PortRole.INPUT),
                    _port("ex-out", "excitation reflected", PortRole.OUTPUT),
                    _port("det-in", "emission in", PortRole.INPUT),
                    _port(
                        "det-out",
                        "emission transmitted",
                        PortRole.OUTPUT,
                        emission_band,
                    ),
                ],
            )
        )
        if tail is not None:
            hops.append((*tail, "dichroic", "ex-in", excitation_beam, "AIR"))
        tail = ("dichroic", "ex-out")

    # --- the objective, the sample, and back --------------------------------
    if objective is not None:
        elements.append(
            ObjectiveElementInput(
                id="objective",
                label=text(objective.model) or text(objective.id) or "Objective",
                brand=text(objective.manufacturer),
                model=text(objective.model),
                serial_number=text(objective.serial_number),
                magnification=objective.nominal_magnification
                or objective.calibrated_magnification,
                numerical_aperture=objective.lens_na,
                working_distance=um(
                    objective.working_distance, objective.working_distance_unit
                ),
                immersion_medium=IMMERSION_BY_OME.get(text(objective.immersion) or ""),
                iris=objective.iris,
                ports=[
                    _port("illum-in", "excitation in", PortRole.INPUT),
                    _port("illum-out", "excitation to sample", PortRole.OUTPUT),
                    _port("det-in", "fluorescence in", PortRole.INPUT),
                    _port("det-out", "fluorescence out", PortRole.OUTPUT),
                ],
            )
        )
        if tail is not None:
            hops.append((*tail, "objective", "illum-in", excitation_beam, "AIR"))
        tail = ("objective", "illum-out")

    sample_description = " — ".join(
        p
        for p in (
            text(image.description),
            f"stained for {text(channel.fluor)}" if channel.fluor else None,
            text(channel.name),
        )
        if p
    )
    elements.append(
        SampleElementInput(
            id="sample",
            label=text(image.name) or "Sample",
            description=sample_description or None,
            ports=[
                _port("illum-in", "illuminated volume", PortRole.INPUT),
                _port("em-out", "fluorescence", PortRole.OUTPUT, emission_band),
            ],
        )
    )
    if tail is not None:
        hops.append((*tail, "sample", "illum-in", excitation_beam, immersion))
    tail = ("sample", "em-out")

    if objective is not None:
        hops.append((*tail, "objective", "det-in", emission_beam, immersion))
        tail = ("objective", "det-out")

    if dichroic is not None:
        hops.append((*tail, "dichroic", "det-in", emission_beam, "AIR"))
        tail = ("dichroic", "det-out")

    if channel.pinhole_size is not None:
        elements.append(
            PinholeElementInput(
                id="pinhole",
                label="Confocal pinhole",
                diameter=um(channel.pinhole_size, channel.pinhole_size_unit),
                ports=_in_out(),
            )
        )
        hops.append((*tail, "pinhole", "in", emission_beam, "AIR"))
        tail = ("pinhole", "out")

    for index, filter_obj in enumerate(emission):
        element_id = f"em-filter-{index}"
        elements.append(
            FilterElementInput(
                id=element_id,
                label=text(filter_obj.model)
                or text(filter_obj.id)
                or f"Emission filter {index}",
                manufacturer=text(filter_obj.manufacturer),
                model=text(filter_obj.model),
                serial_number=text(filter_obj.serial_number),
                filter_kind=FILTER_KIND_BY_OME.get(text(filter_obj.type) or ""),
                transmittance=getattr(
                    filter_obj.transmittance_range, "transmittance", None
                ),
                description=text(getattr(filter_obj, "filter_wheel", None)),
                ports=_in_out(_band(filter_obj.transmittance_range)),
            )
        )
        hops.append((*tail, element_id, "in", emission_beam, "AIR"))
        tail = (element_id, "out")

    if detector is not None:
        # Built here, per channel, with this channel's gain — never looked up from a
        # shared map and mutated, which would let one channel's gain overwrite another's
        # on an object already handed to an earlier graph.
        detector_settings = channel.detector_settings
        gain = getattr(detector_settings, "gain", None)
        if gain is None:
            gain = detector.gain
        elements.append(
            DetectorElementInput(
                id="detector",
                label=text(detector.model) or text(detector.id) or "Detector",
                manufacturer=text(detector.manufacturer),
                model=text(detector.model),
                serial_number=text(detector.serial_number),
                gain=gain,
                amplifier_gain_db=detector.amplification_gain,
                ports=[_port("in", "sensor", PortRole.INPUT)],
            )
        )
        hops.append((*tail, "detector", "in", emission_beam, "AIR"))

    if len(elements) < 2 or not hops:
        # A graph of one sample and no path says nothing the anchor's label does not.
        return None

    edges = [
        LightEdgeInput(
            id=f"{src}:{src_port}->{dst}:{dst_port}",
            source_element_id=src,
            source_port_id=src_port,
            target_element_id=dst,
            target_port_id=dst_port,
            beam=beam,
            medium=medium,
        )
        for src, src_port, dst, dst_port, beam, medium in hops
    ]
    return LightpathGraphInput(elements=elements, edges=edges)
