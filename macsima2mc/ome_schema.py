#!/usr/bin/python
import ome_types
import pandas as pd
import tifffile as tifff
from bs4 import BeautifulSoup
from uuid import uuid4
import copy
from ome_types import from_tiff,to_xml
from ome_types.model import OME,Image,Instrument,Pixels,TiffData,Channel,Plane
import ome_types.model
import platform

def TIFF_block(no_of_channels):

    TIFF=[
        TiffData(
            first_c=ch,
            ifd='',
            plane_count=1
            )
        for ch in range(0,no_of_channels)
        ]

    return TIFF

def PLANE_block(no_of_channels):

    PLANE=[
        Plane(
            the_c=ch,
            the_t=0,
            the_z=0,
            position_x='',#x=0 is just a place holder
            position_y='',#y=0 is just a place holder
            position_z='',#Z=0 is just a place holder
            exposure_time='',#0 is just a place holder
            position_x_unit='',#mm is just a place holder
            position_y_unit=''#mm is just a place holder
            )
        for ch in range(0,no_of_channels)
        ]

    return PLANE
    
def CHANN_block(no_of_channels):

    CHANN=[
        Channel(
            id=ome_types.model.simple_types.ChannelID('Channel:{x}'.format(x=ch)),
            color=ome_types.model.simple_types.Color((255,255,255)),
            emission_wavelength='',#place holder
            excitation_wavelength='',#place holder
            )
        for ch in range(0,no_of_channels)
        ]

    return CHANN

def PIXELS_block(no_of_tiles):

    PIXELS=[
        Pixels(
            id=ome_types.model.simple_types.PixelsID('Pixels:{x}'.format(x=t)),
            dimension_order=ome_types.model.pixels.DimensionOrder('XYCZT'),
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
            physical_size_x_unit=pixel_units,
            physical_size_y=pixel_size,
            physical_size_y_unit=pixel_units,
            physical_size_z=1.0,
            planes=template_plane_block,
            significant_bits=sig_bits,
            tiff_data_blocks=template_tiffdata_block
            )
            for t in range(0,no_of_tiles)
            ]

    return PIXELS

def IMAGE_block(no_of_tiles):
    
    IMAGE=[
        Image(
            id=ome_types.model.simple_types.ImageID('Image:{x}'.format(x=t)),
            pixels=pix_block[t]
            )
            for t in range(0,no_of_tiles)
            ]

    return IMAGE

def OME_block(image_block):

    ome=OME()
    ome.creator=" ".join([ome_types.__name__,
                        ome_types.__version__,
                        '/ python version-',
                        platform.python_version()
                        ]
                        )
    ome.images=image_block
    ome.uuid=uuid4().urn
    ome_xml=to_xml(ome_custom)

    return ome_xml




