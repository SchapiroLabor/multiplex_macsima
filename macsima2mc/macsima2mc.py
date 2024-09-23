import helpers 
from pathlib import Path
from templates import macsima_pattern
import pandas as pd
import os


input_test_folder=Path("D:/macsima_data_samples/macsima_data_v2/6_Cycle1")
output_test_folder=Path('D:/test_folder')

info=helpers.cycle_info(input_test_folder, macsima_pattern(version=2),ref_marker= 'DAPI')
info_extra=helpers.append_extra_info(info)
info_extra.to_csv( output_test_folder / 'cycle_{c}_info.csv'.format(c=f'{6:03d}'), index=False )
helpers.create_stack(info_extra,output_test_folder,ref_marker='DAPI')





