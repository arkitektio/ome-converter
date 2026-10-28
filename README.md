# ome-converter

Converts any file Bio-Formats can read (czi, lif, nd2, ome-tiff, ...) into mikro datasets,
one pyramided `ArrayDataset` per series.

Everything the file says about a series is pinned to a coordinate. The OME document, the
microscope and the stage go on the series' origin. Each channel's histogram, name,
detector and light path go on that channel. Every calibrated series is registered into one
micrometer space. Each series gets a scene of its own over that space, with **one layer
per channel**. The only exception is a packed 8-bit RGB(A) photograph, which gets one RGB
layer. A file with no pixel calibration is not placed in that space. It is staged in its
own pixel grid instead. The module docstring in `app.py` has the whole contract.

Tile scans are handled differently. Bio-Formats splits a LIF tile scan into one series per
tile, all with the same name. Every tile is still uploaded and registered. If the file also
has the stitched image (the same name plus `merged`, e.g. `R 1_Merged`), only the stitched
image gets a scene. If it does not, the tiles share one mosaic scene.

The app started as a copy of `testing/converter_ome_dataset.py`.

## The action

**Convert OME file to Datasets** (`convert_ome_file_to_datasets`) takes a mikro `File`,
plus `levels`, `method`, `stage_scene` and `world_name`. It returns the datasets it made.
If one series fails, it is logged and the others still convert. The action fails only when
every series does.

## Java

Bio-Formats runs on the image's system JRE. The jars are resolved when the image is built
(`BIOFORMATS_VERSION`, pinned in the flavour's Dockerfile) and kept under `/opt/jgo`, so a
pod never needs the network to read a file. To move to a newer Bio-Formats, raise the
pin there. A local run (`arkitekt run dev`) downloads its own JRE and jars the first time
it reads a file.

## Developing

```sh
uv run pytest             # offline: no server, no JVM
uv run arkitekt check
uv run arkitekt plugin build -p linux/amd64
```
