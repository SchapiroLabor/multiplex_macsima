import helpers 
from pathlib import Path
from templates import macsima_pattern,info_dic

#data_folder=Path('D:/macsima_data_samples/macsima_data_v1/001_AntigenCycle')
#images=list(data_folder.glob("*.tif"))

img_test=Path('000_BleachCycle_None_V50_PE_16bit_M-20x-S Fluor full sensor_B-1_R-2_W-2_G-1_F-30_E-144.0.tif')

def extract_img_info(img_test, macsima_pattern, info_dic,ref_marker= 'DAPI',dir_version=1)







