from pathlib import Path
from tools import extract_values
import re
import numpy as np
from ome_types import from_tiff
from pandas import DataFrame



def flatten_list(lol):
    """
    This function flattens a list of lists.
    Args:
        lol (list): list of lists.
    Returns:
        list: flattened list.
    """
    return [x for xs in lol for x in xs]


def markers_file():
    """
    This function creates a dictionary with the columns of the markers.csv file.
    Returns:
        dict: dictionary with the columns of the markers.csv file.
    """
    columns = {'channel_number': [],
               'cycle_number': [],
               'marker_name':[],
               'Filter':[],
               'background':[],
               'exposure':[],
               'remove':[]
               }

    return columns


def get_patterns():
    """
    Dictionary of tuples where first element indicates the pattern to search and
    second element is a boolean that indicates whether to transform the search output into a value or not
    the keys of this dictionary should be a subset of the keys in the markers_file function.
    Returns:
        dict: dictionary with regular expressions to extract metadata from macsima filenames.
    """

    patterns = {
        'cycle_number': (r"cycle-(.*?)-", True),
        'marker_name' : (r"markers-(.*?)-",False),
        'Filter'      : (r"-filters-(.*?).ome",False),
        'background'  : (r"-src-(.*?)-" ,False)
        }

    return patterns


def write_markers_file( data_path, ref_marker='DAPI'):
    """
    This function writes the markers.csv file.
    Args:
        data_path (Path): path to the folder containing the images.
        ref_marker (str): reference marker.
    Returns:
        dict: dictionary with the columns of the markers.csv file.
    """

    img_paths = list(sorted( data_path.glob('*.tif*')))
    mks_file = markers_file()
    patt = get_patterns()

    for img in img_paths:
        img_name = [img.stem]
        cycle_no = extract_values(patt['cycle_number'][0], img_name, number_cast=patt['cycle_number'][1])
        background = extract_values(patt['background'][0], img_name, number_cast=patt['background'][1])
        markers = extract_values(patt['marker_name'][0], img_name, number_cast=patt['marker_name'][1])[0].split('__')
        filters = extract_values(patt['Filter'][0], img_name, number_cast=patt['Filter'][1])[0].split('__')
        ome = from_tiff(img)

        if background[0]=='B':
            remove = len(markers)*['TRUE']
            markers = ['bg_{c}_{m}-{f}'.format( c= f'{cycle_no[0]:03d}' , m=x , f=y ) for x,y in zip(markers,filters) ]
            fmt_background = len(markers)*['']
        else:
            fmt_background = []
            remove = len(markers)*['']
            for x,y in zip(markers,filters):
                if x == ref_marker:
                    fmt_background.append('')
                else:
                    fmt_background.append(f'bg_{cycle_no[0]:03d}_{x}-{y}')

        mks_file['cycle_number'].extend(len(markers)*cycle_no)
        mks_file['marker_name'].extend(markers)
        mks_file['Filter'].extend(filters)
        mks_file['background'].extend(fmt_background)
        mks_file['exposure'].extend([ome.images[0].pixels.planes[ch].exposure_time for ch,_ in enumerate(markers)])
        mks_file['remove'].extend(remove)

    mks_file['channel_number'] = list(range(1, 1 + len(mks_file['marker_name'])))
    mks_file_df = DataFrame(mks_file)
    mks_file_df.to_csv( data_path.parent.absolute() / 'markers.csv' , index=False )

    return mks_file
