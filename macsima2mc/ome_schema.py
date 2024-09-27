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
from helpers import merge_dicts



def INPUTS(frame,conformed_markers):
    features=frame.columns.values
    inputs={column:[] for column in features }
    metadata=[ frame.loc[ (frame['marker']==marker) & (frame['filter']==filt)] for marker,filt in conformed_markers ]
    for meta in metadata:
        for f in features:
            inputs[f].append(meta[f].values[0])
    return inputs






def TIFF_array(no_of_channels,
                inputs={'offset':0}
                ):

    TIFF=[
        TiffData(
            first_c=ch,
            ifd=n,
            plane_count=1
            )
        for n,ch in enumerate(range(0,no_of_channels), start=inputs['offset'])
        ]

    return TIFF

def PLANE_array(no_of_channels,inputs):

    PLANE=[
        Plane(
            the_c=ch,
            the_t=0,
            the_z=0,
            position_x=inputs['position_x'][ch],
            position_y=inputs['position_y'][ch],
            position_z=0,#Z=0 is just a place holder
            exposure_time=inputs['exposure_time'][ch],
            position_x_unit=inputs['position_x_unit'][ch],
            position_y_unit=inputs['position_y_unit'][ch]
            )
        for ch in range(0,no_of_channels)
        ]

    return PLANE
    
def CHANN_array(no_of_channels,inputs):

    CHANN=[
        Channel(
            id=ome_types.model.simple_types.ChannelID('Channel:{y}:{x}:{marker_name}'.format(x=ch,y=100+inputs['tile'][ch],marker_name=inputs['marker'][ch] )),
            color=ome_types.model.simple_types.Color((255,255,255)),
            emission_wavelength=inputs['emission_wavelenght'][ch],
            emission_wavelength_unit=inputs['emission_wavelenght_unit'][ch],
            excitation_wavelength=inputs['excitation_wavelenght'][ch],
            excitation_wavelength_unit=inputs['excitation_wavelenght_unit'][ch]

            )
        for ch in range(0,no_of_channels)
        ]

    return CHANN

def PIXELS_array(chann_block,plane_block,tiff_block,inputs):

    PIXELS=[
        Pixels(
            id=ome_types.model.simple_types.PixelsID('Pixels:{x}'.format(x=inputs['tile'][0])),
            dimension_order=ome_types.model.pixels.DimensionOrder('XYCZT'),
            size_c=len(chann_block),
            size_t=1,
            size_x=inputs['size_x'][0],
            size_y=inputs['size_y'][0],
            size_z=1,
            type=inputs['type'][0],#bit_depth
            big_endian=False,
            channels=chann_block,
            interleaved=False,
            physical_size_x=inputs['physical_size_x'][0],
            physical_size_x_unit=inputs['physical_size_x_unit'][0],
            physical_size_y=inputs['physical_size_y'][0],
            physical_size_y_unit=inputs['physical_size_y_unit'][0],
            physical_size_z=1.0,
            planes=plane_block,
            significant_bits=inputs['significant_bits'][0],
            tiff_data_blocks=tiff_block
            )
        ]

    return PIXELS

def IMAGE_array(pixels_block,imageID):
    
    IMAGE=[
        Image(
            id=ome_types.model.simple_types.ImageID('Image:{x}'.format(x=imageID)),
            pixels=pixels_block
            )
        ]

    return IMAGE

def OME_xml(image_block):

    ome=OME()
    ome.creator=" ".join([ome_types.__name__,
                        ome_types.__version__,
                        '/ python version-',
                        platform.python_version()
                        ]
                        )
    ome.images=image_block
    ome.uuid=uuid4().urn
    ome_xml=to_xml(ome)

    return ome_xml
    






