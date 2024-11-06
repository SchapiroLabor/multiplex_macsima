import re

def info_dic(target_pattern):
    """
    creates a dictionary with keys mirroring the keys in the target_pattern dictionary.
    The value of each key is an empty string.
    Args:
        target_pattern (dict): dictionary taken from the macsima_pattern function.
    Returns:
        dict: template with keys from macsima_pattern plus 2 extra keys, full path of image and image name.
    """

    # Initialize dictionary
    template = {
        'full_path': '',
        'img_name': ''
    }

    for key in target_pattern:
        template[key] = ''
    
    return template


def macsima_pattern(version=1):
    """
    Returns a dictionary with regular expressions to extract metadata from macsima filenames.
    Args:
        version (int): version of the macsima filenames. Default is 1.
    Returns:
        dict: dictionary with regular expressions to extract metadata from macsima filenames.
    """
    
    if version == 1:
        pattern = {
            "cycle"         : r"(.*?)_(.*?)Cycle",
            "source"        : r"_(.*?)Cycle",
            "rack"          : r"_R-(\d+)",
            "well"          : r"_W-(\d+)",
            "roi"           : r"_G-(\d+)",
            "tile"          : r"_F-(\d+)",
            "exposure_time" : r"_E-(\d+)",
            "marker"        : r"Cycle_(.*?)_",
            "filter"        : r".*_([^_]*)_\d+bit"
        }

    elif version==2:

        pattern = {
            "cycle"         : r"CYC-(\d+)",
            "source"        : r"_ST-(.*?)_",
            "rack"          : r"_R-(\d+)",
            "well"          : r"_W-(.*?\d+)",
            "roi"           : r"_ROI-(\d+)",
            "tile"          : r"_F-(\d+)",
            "exposure_time" : r"_EXP-(\d+(?:\.\d+)?)",
            "marker"        : r"_A-(.*?)_",
            "filter"        : r"_D-(.*?)_"
            }
        
    else:
        raise ValueError(
            "version argument should be 1 or 2"
        )
    return pattern
