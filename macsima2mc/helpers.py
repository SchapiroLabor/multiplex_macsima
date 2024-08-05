import templates
from templates import info_dic



def extract_values(pattern, strings, number_cast=True):
    return [
        (int(m.group(1)) if number_cast else m.group(1))
        if (m := re.search(pattern, s))
        else None
        for s in strings
    ]





def extract_img_info(full_img_path, platform_pattern,ref_marker= 'DAPI',dir_version=1):
    '''
    This function reads the images produced by the MACSima device and returns the acquistion information
    specified in the image name.
    inputs:
    -img_path[Path]= full path to the image
    -ref[str]=marker of reference used for registration
    -source[str]= valid values 'Antigen' or 'Bleach'
    -dir_version[int]=version of the macsima folder and file naming structure.  Valid values are 1 or 2.
        E.g. version_1 (001_AntigenCycle_DAPI_V0_DAPI_16bit_M-20x-S Fluor full sensor_B-1_R-2_W-2_G-1_F-30_E-16.0.tif) and 
        version_2 (CYC-001_SCN-001_ST-B_R-01_W-B01_ROI-001_F-001_A-Syk_C-_D-FITC_EXP-17.5781.tif)
    output:
    -info[dict]=dictionary with acquisition information, ROI, rack, exposure time etc.

    '''
    info=info_dic(platform_pattern)
    info['full_path']=full_img_path
    info['img_name']=Path(full_img_path).name
    for key,value in platform_pattern.items():
        info[key]=


    
    return info

