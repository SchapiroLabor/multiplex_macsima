import re


#img_info=
#{
#   'name':'',   #string
#    'device':'', #string
#    'no_channels':[], #list 
#    'markers':[],
#    'filters':[],
#    'exposure_times':exposure_per_marker,
#    'xy_img_size_pix':(width,height),
#    'pix_size':tile_data['pixel_size'],
#    'pix_units':tile_data['pixel_units'],
#    'bit_depth':tile_data['bit_depth'],
#    'sig_bits':tile_data['sig_bits'].
#    'tile_positions':
#
#}

def info_dic(target_pattern):
    '''
    creates a dictionary with keys mirroring the keys in the
    target_pattern dictionary.  The value of each key is an empty string.
    inputs:
        -target_pattern[dic]: dictionary taken from the macsima_pattern function.
    outputs:
        template[dic]: template with keys from macsima_pattern plus 2 extra keys,
        full path of image and image name.

    '''

    template={}
    template['full_path']=''
    template['img_name']=''

    for key in target_pattern:
        template[key]=''
    
    return template

def macsima_pattern(version=1):
    
    if version==1:

        pattern={
            "cycle" :  r"(.*?)_(.*?)Cycle",
            "source":  r"_(.*?)Cycle",
            "rack"  :  r"_R-(\d+)",
            "well"  :  r"_W-(\d+)",
            "roi"   :  r"_G-(\d+)",
            "tile"  :  r"_F-(\d+)",
            "exposure_time":r"_E-(\d+)",
            "marker":  r"Cycle_(.*?)_",
            "filter":  r".*_([^_]*)_\d+bit"
                }

    elif version==2:

        pattern={
            "cycle": r"CYC-(\d+)",
            "source": r"_ST-(.*?)_",
            "rack":  r"_R-(\d+)",
            "well":  r"_W-(.*?\d+)",
            "roi":   r"_ROI-(\d+)",
            "tile":  r"_F-(\d+)",
            "exposure_time": r"_EXP-(\d+(?:\.\d+)?)",
            "marker": r"_A-(.*?)_",
            "filter": r"_D-(.*?)_"   
            }
        
    else:
        raise ValueError(
            "version argument should be 1 or 2"
        )
    return pattern
