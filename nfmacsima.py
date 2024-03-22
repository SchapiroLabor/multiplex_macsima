#!/usr/bin/python
import argparse
import copy
import os
import platform
from pathlib import Path
from uuid import uuid4

import numpy as np
import ome_types
import ome_types.model
import pandas as pd

import tifffile as tifff
from bs4 import BeautifulSoup
from ome_types import to_xml
from ome_types.model import OME, Image, Pixels, TiffData, Channel, Plane
from ome_types.model.simple_types import PixelsID, ImageID, Color, ChannelID
from ome_types.model.tiff_data import UUID
import re
import sys

# ---CLI-BLOCK---#
parser = argparse.ArgumentParser()

parser.add_argument(
    "-i",
    "--input",
    nargs="*",
    required=True,
    help="Ordered pairs of paths to the Antigen and corresponding Bleaching cycle, e.g [002_AntigenCycle,001_BleachCycle].",
)

parser.add_argument(
    "-o", "--output", required=True, help="Directory to save the stack."
)

parser.add_argument(
    "-c", "--cycle", required=True, type=int, help="Number of the antigen cycle."
)

parser.add_argument(
    "-rm",
    "--ref_marker",
    required=False,
    type=str,
    default="DAPI",
    help="Name of the reference marker.Default value is DAPI.",
)

parser.add_argument(
    "-he",
    "--hi_exposure_only",
    action="store_true",
    help="Activate this flag to extract only the set of images with the highest exposure time.",
)

args = parser.parse_args()
# ---END_CLI-BLOCK---#


# ------FORMAT INPUT BLOCK-------#
device_name = "MACSIMA"
antigen_dir = Path(args.input[0])
bleach_dir = Path(args.input[1])
stack_path = Path(args.output)
ref_marker = args.ref_marker
cycle_no = args.cycle

# ------ ENDINPUT BLOCK----#

os.makedirs(stack_path, exist_ok=True)


# ---- HELPER FUNCTIONS ----#


def extract_values(pattern, strings, number_cast=True):
    return [
        (int(m.group(1)) if number_cast else m.group(1))
        if (m := re.search(pattern, s))
        else None
        for s in strings
    ]


def cycle_info(data_folder: Path, ref_marker: str, antigen_cycle_no: int = cycle_no):
    if "AntigenCycle" in data_folder.name:
        cycle = "AntigenCycle"
    elif "BleachCycle" in data_folder.name:
        cycle = "BleachCycle"
    else:
        raise ValueError(
            "Folder name should contain either AntigenCycle or BleachCycle."
        )

    full_image_path = list(data_folder.glob("*.tif"))
    images = [x.name for x in full_image_path]

    # Define the patterns
    b_pattern = r"_B-(\d+)"  # What is this?
    rack_pattern = r"_R-(\d+)"
    well_pattern = r"_W-(\d+)"
    roi_pattern = r"_G-(\d+)"
    fov_pattern = r"_F-(\d+)"
    exposure_pattern = r"_E-(\d+)"
    filter_pattern = r".*_([^_]*)_\d+bit"

    marker_pattern = rf"{cycle}_(.*?)_"
    marker_values = extract_values(
        pattern=marker_pattern, strings=images, number_cast=False
    )
    filter_values = extract_values(
        pattern=filter_pattern, strings=images, number_cast=False
    )

    # Todo: Is this even needed?
    filter_values = [
        ref_marker if marker == ref_marker else value
        for marker, value in zip(marker_values, filter_values)
    ]

    info = pd.DataFrame(
        {
            "img_full_path": full_image_path,
            "image": images,
            "marker": extract_values(
                pattern=marker_pattern, strings=images, number_cast=False
            ),
            "filter": filter_values,
            "rack": extract_values(pattern=rack_pattern, strings=images),
            "well": extract_values(pattern=well_pattern, strings=images),
            "roi": extract_values(pattern=roi_pattern, strings=images),
            "fov": extract_values(pattern=fov_pattern, strings=images),
            "exposure": extract_values(pattern=exposure_pattern, strings=images),
        }
    )

    markers = info["marker"].unique()
    markers_subset = np.setdiff1d(markers, [ref_marker])

    # insert the exposure level colum at the end of the info dataframe
    info.insert(loc=info.shape[1], column="exposure_level", value=0)

    info["exposure_level"] = (
        info.groupby("marker")["exposure"].rank(method="dense").astype(int)
    )
    info.loc[info["marker"] == ref_marker, "exposure_level"] = "ref"

    if cycle == "BleachCycle":
        bleach_cycle = f"{antigen_cycle_no - 1:03d}"
        info.loc[info["marker"] == "None", "marker"] = (
            f"{bleach_cycle}_bleach_" + info["filter"]
        )

    return info


def create_ome(img_info, xy_tile_positions_units, img_path):
    img_name = img_info["name"]
    device_name = img_info["device"]
    no_of_channels = img_info["no_channels"]
    img_size = img_info["xy_img_size_pix"]
    markers = img_info["markers"]
    exposures = img_info["exposure_times"]
    bit_depth = img_info["bit_depth"]
    pixel_size = img_info["pix_size"]
    pixel_units = img_info["pix_units"]
    sig_bits = img_info["sig_bits"]
    if pixel_units == "mm":
        pixel_size = 1000 * pixel_size
        # pixel_units="um"
    no_of_tiles = len(xy_tile_positions_units)
    tifff.tiffcomment(img_path, "")
    # UUID=ome_types.model.tiff_data.UUID(file_name=img_name,value=uuid4().urn)
    unique_identifier = UUID(file_name=img_name, value=uuid4().urn)
    tiff_block = [
        TiffData(first_c=ch, ifd=ch, plane_count=1, uuid=unique_identifier)
        for ch in range(no_of_channels)
    ]
    plane_block = [
        Plane(
            the_c=ch,
            the_t=0,
            the_z=0,
            position_x=0,  # x=0 is just a place holder
            position_y=0,  # y=0 is just a place holder
            position_z=0,
            exposure_time=0,
            # position_x_unit=pixel_units,
            # position_y_unit=pixel_units
        )
        for ch in range(no_of_channels)
    ]
    chann_block = [
        Channel(
            id=ChannelID("Channel:{x}".format(x=ch)),
            color=Color((255, 255, 255)),
            emission_wavelength=1,  # place holder
            excitation_wavelength=1,  # place holder
        )
        for ch in range(no_of_channels)
    ]
    # --Generate pixels block--#
    pix_block = []
    ifd_counter = 0
    for t in range(no_of_tiles):
        template_plane_block = copy.deepcopy(plane_block)
        template_chann_block = copy.deepcopy(chann_block)
        template_tiffdata_block = copy.deepcopy(tiff_block)
        for ch, mark in enumerate(markers):
            template_plane_block[ch].position_x = xy_tile_positions_units[t][0]
            template_plane_block[ch].position_y = xy_tile_positions_units[t][1]
            template_plane_block[ch].exposure_time = exposures[ch]
            template_chann_block[ch].id = "Channel:{y}:{x}:{marker_name}".format(
                x=ch, y=100 + t, marker_name=mark
            )
            template_tiffdata_block[ch].ifd = ifd_counter
            ifd_counter += 1
        pix_block.append(
            Pixels(
                id=PixelsID("Pixels:{x}".format(x=t)),
                dimension_order=ome_types.model.pixels.DimensionOrder("XYCZT"),
                size_c=no_of_channels,
                size_t=1,
                size_x=img_size[0],
                size_y=img_size[1],
                size_z=1,
                type=bit_depth,
                big_endian=False,
                channels=template_chann_block,
                interleaved=False,
                physical_size_x=pixel_size,
                # physical_size_x_unit=pixel_units,
                physical_size_y=pixel_size,
                # physical_size_y_unit=pixel_units,
                physical_size_z=1.0,
                planes=template_plane_block,
                significant_bits=sig_bits,
                tiff_data_blocks=template_tiffdata_block,
            )
        )
    img_block = [
        Image(id=ImageID("Image:{x}".format(x=t)), pixels=pix_block[t])
        for t in range(no_of_tiles)
    ]
    # --Create the OME object with all prebiously defined blocks--#
    ome_custom = OME()
    ome_custom.creator = " ".join(
        [
            ome_types.__name__,
            ome_types.__version__,
            "/ python version-",
            platform.python_version(),
        ]
    )
    ome_custom.images = img_block
    ome_custom.uuid = uuid4().urn
    ome_xml = to_xml(ome_custom)
    tifff.tiffcomment(img_path, ome_xml)


def setup_coords(x, y, pix_units):
    if pix_units == "mm":
        x_norm = 1000 * (np.array(x) - np.min(x))  # /pixel_size
        y_norm = 1000 * (np.array(y) - np.min(y))  # /pixel_size
    x_norm = np.rint(x_norm).astype("int")
    y_norm = np.rint(y_norm).astype("int")
    # invert y
    y_norm = np.max(y_norm) - y_norm
    return list(zip(x_norm, y_norm))


def tile_position(metadata_string):
    ome = BeautifulSoup(metadata_string, "xml")
    x = float(ome.StageLabel["X"])
    y = float(ome.StageLabel["Y"])
    stage_units = ome.StageLabel["XUnit"]
    pixel_size = float(ome.Pixels["PhysicalSizeX"])
    pixel_units = ome.Pixels["PhysicalSizeXUnit"]
    bit_depth = ome.Pixels["Type"]
    significant_bits = int(ome.Pixels["SignificantBits"])
    return {
        "x": x,
        "y": y,
        "stage_units": stage_units,
        "pixel_size": pixel_size,
        "pixel_units": pixel_units,
        "bit_depth": bit_depth,
        "sig_bits": significant_bits,
    }


def create_stack(
    info,
    exp_level=1,
    antigen_cycle_no=cycle_no,
    cycle_folder="",
    isbleach=False,
    offset=0,
    device=device_name,
    ref_marker=ref_marker,
    results_path=stack_path,
):
    racks = info["rack"].unique()
    wells = info["well"].unique()
    antigen_cycle = f"{antigen_cycle_no:03d}"
    cycle_prefix = "Bleach" if isbleach else "Antigen"
    workdir = Path(cycle_folder)

    with tifff.TiffFile(info["img_full_path"][0]) as tif:
        img_ref = tif.pages[0].asarray()
    width = img_ref.shape[1]
    height = img_ref.shape[0]
    dtype_ref = img_ref.dtype
    ref_marker = list(info.loc[info["exposure_level"] == "ref", "marker"])[0]
    markers = info["marker"].unique()

    markers_subset = np.setdiff1d(markers, [ref_marker])
    sorted_markers = np.insert(markers_subset, 0, ref_marker)
    sorted_filters = [
        info.loc[info["marker"] == m, "filter"].tolist()[0] for m in sorted_markers
    ]
    if isbleach:
        target_name = "filters"
        target = "_".join(sorted_filters)
    else:
        target_name = "markers"
        target = "_".join(sorted_markers)

    for r in racks:
        output_levels = ["rack_{n}".format(n=f"{r:02d}")]
        for w in wells:
            well_no = "well_{n}".format(n=f"{w:02d}")
            output_levels.append(well_no)
            groupA = info.loc[(info["rack"] == r) & (info["well"] == w)]
            rois = groupA["roi"].unique()

            for roi in rois:
                roi_no = "roi_{n}".format(n=f"{roi:02d}")
                exp_level_no = "exp_level_{n}".format(n=f"{exp_level:02d}")
                output_levels.extend((roi_no, exp_level_no))
                counter = 0
                groupB = groupA.loc[groupA["roi"] == roi]
                A = groupB.loc[(groupB["exposure_level"] == "ref")]
                stack_size_z = A.shape[0] * len(markers)
                fov_id = groupB.loc[groupB["marker"] == ref_marker, "fov"].unique()
                fov_id = np.sort(fov_id)

                stack_name = "cycle-{C}-{prefix}-exp-{E}-rack-{R}-well-{W}-roi-{ROI}-{T}-{M}.{img_format}".format(
                    C=f"{(offset + antigen_cycle_no):03d}",
                    prefix=cycle_prefix,
                    E=exp_level,
                    R=r,
                    W=w,
                    ROI=roi,
                    T=target_name,  # markers or filters
                    M=target,
                    img_format="ome.tiff",
                )

                X = []
                Y = []
                stack = np.zeros((stack_size_z, height, width), dtype=dtype_ref)

                exp = groupB.loc[
                    (groupB["marker"] == ref_marker)
                    & (groupB["fov"] == 1)
                    & (groupB["exposure_level"] == "ref"),
                    "exposure",
                ].tolist()[0]
                exposure_per_marker = [exp]

                for s in markers_subset:
                    exp = groupB.loc[
                        (groupB["marker"] == s)
                        & (groupB["fov"] == 1)
                        & (groupB["exposure_level"] == exp_level),
                        "exposure",
                    ].tolist()[0]
                    exposure_per_marker.append(exp)

                for tile in fov_id:
                    if img_ref := groupB.loc[
                        (groupB["marker"] == ref_marker) & (groupB["fov"] == tile),
                        "img_full_path",
                    ].tolist():
                        with tifff.TiffFile(img_ref[0]) as tif:
                            img = tif.pages[0].asarray()
                            metadata = tif.ome_metadata

                        stack[counter, :, :] = img
                        tile_data = tile_position(metadata)
                        X.append(tile_data["x"])
                        Y.append(tile_data["y"])
                        counter += 1

                        for m in markers_subset:
                            img_marker = groupB.loc[
                                (groupB["marker"] == m)
                                & (groupB["fov"] == tile)
                                & (groupB["exposure_level"] == exp_level),
                                "img_full_path",
                            ].tolist()[0]
                            img = tifff.imread(img_marker)
                            stack[counter, :, :] = img
                            counter += 1

                output_folders_path = stack_path / "--".join(output_levels) / "raw"

                os.makedirs(output_folders_path, exist_ok=True)

                final_stack_path = os.path.join(output_folders_path, stack_name)
                tifff.imwrite(final_stack_path, stack, photometric="minisblack")
                img_info = {
                    "name": stack_name,
                    "device": device_name,
                    "no_channels": len(markers),
                    "markers": sorted_markers,
                    "filters": sorted_filters,
                    "exposure_times": exposure_per_marker,
                    "xy_img_size_pix": (width, height),
                    "pix_size": tile_data["pixel_size"],
                    "pix_units": tile_data["pixel_units"],
                    "bit_depth": tile_data["bit_depth"],
                    "sig_bits": tile_data["sig_bits"],
                }
                create_ome(
                    img_info,
                    setup_coords(X, Y, img_info["pix_units"]),
                    img_path=final_stack_path,
                )

    return img_info


def main():
    antigen_info = cycle_info(data_folder=antigen_dir, ref_marker=ref_marker)
    bleach_info = cycle_info(data_folder=bleach_dir, ref_marker=ref_marker)
    print(antigen_info)
    sys.exit()

    exp = antigen_info["exposure_level"].unique()
    exp = exp[exp != "ref"]
    exp.sort()

    if args.hi_exposure_only:
        exp = [max(exp)]

    for e in exp:
        antigen_stack_info = create_stack(
            antigen_info, antigen_cycle_no=cycle_no, exp_level=e
        )
        bleach_stack_info = create_stack(
            bleach_info, antigen_cycle_no=cycle_no, isbleach=True, exp_level=e
        )


if __name__ == "__main__":
    main()
