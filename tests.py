#!/usr/bin/python
import ome_types
import numpy as np
import pandas as pd
from pathlib import Path
#from PIL import Image
#import PIL
#from PIL.TiffTags import TAGS
import tifffile as tifff
from bs4 import BeautifulSoup
import os
from uuid import uuid4
import copy
from ome_types import from_tiff,to_xml
from ome_types.model import OME,Image,Instrument,Pixels,TiffData,Channel,Plane
import ome_types.model
import platform
import argparse


def marker_filter_map(info,ref_marker='DAPI'):
    markers= info['marker'].unique()
    markers_subset=np.setdiff1d(markers,[ref_marker])
    sorted_markers=[ref_marker]
    sorted_filters=[ref_marker]
    for m in markers_subset:
        filters=info.loc[info['marker']==m,'filter'].unique()
        sorted_filters.extend(filters)
        for _ in range(0,len(filters)):
            sorted_markers.append(m)
    
    return sorted_markers,sorted_filters




def cycle_info(cycle_no,cycles_path,ref_marker='DAPI'):

    cycle_folder=list(filter( lambda x:  ('Cycle{n}'.format(n=cycle_no) in x.split('_')[-1] ) and ( len(x.split('_')[-1])==len('Cycle')+len(str(cycle_no)) )  , os.listdir(cycles_path) ) )
    cycle_folder=cycle_folder[0]
    workdir=cycles_path / cycle_folder
    images=list(filter(lambda x: x.endswith('.tif'),os.listdir(workdir)))
    antigen_images=list( filter(lambda x: '_ST-S_' in x ,images) )


    cycle_info={'cycle':[],
                'antigen_full_path':[],
                'bleach_full_path':[],
                'antigen_image':[],
                'bleach_image':[],
                'marker':[],
                'filter':[],
                'rack':[],
                'well':[],
                'roi':[],
                'fov':[],
                'exposure':[]
               }


    

    for im in antigen_images:

        file_tags=im.split('_')
        im_bg=im.replace('_ST-S_','_ST-B_')
        im_bg=im_bg.replace('_SCN-002_','_SCN-001_')

        cycle_info['cycle'].append(cycle_no)
        cycle_info['antigen_full_path'].append( workdir / im )
        cycle_info['bleach_full_path'].append( workdir / im_bg )
        cycle_info['antigen_image'].append( im )
        cycle_info['bleach_image'].append( im_bg )

        marker_name=file_tags[7].split('-')[-1]
        filter_name=file_tags[-2].split('-')[-1]
        rack=int( file_tags[3].split('-')[-1] )
        well=file_tags[4].split('-')[-1] 
        roi =int( file_tags[5].split('-')[-1] )
        tile=int( file_tags[6].split('-')[-1])
        exp=(file_tags[-1].split('-')[-1]).strip('.tif')


        cycle_info['marker'].append( marker_name )
        cycle_info['filter'].append( filter_name )

        cycle_info['rack'].append( rack )
        cycle_info['well'].append( well )
        cycle_info['roi'].append (  roi )
        cycle_info['fov'].append (  tile )
        cycle_info['exposure'].append (  exp )
     
        
    info=pd.DataFrame(cycle_info)
    

    #markers= info['marker'].unique()
    
    #markers_subset=np.setdiff1d(markers,[ref_marker])
    info.insert(len(cycle_info),'exposure_level',np.zeros(info.shape[0]))
   #info.loc[ info['marker']==ref_marker, 'exposure_level']='ref'

    #for m in markers_subset:
    sorted_markers,sorted_filters=marker_filter_map(info,ref_marker)
    for m,f in zip(sorted_markers,sorted_filters):
        #exposure=info.loc[info['marker']==m]['exposure'].unique()
        if m==ref_marker:
            info.loc[ info['marker']==ref_marker, 'exposure_level']='ref'
        else:
            exposure=info.loc[(info['marker']==m) & (info['filter']==f) ]['exposure'].unique()
            val_map=[ (float(e),e) for e in exposure]
            val_map.sort(key=lambda a: a[0])
            for level,values in enumerate(val_map,1):
                info.loc[ (info['marker']==m) & (info['exposure']==values[1]), 'exposure_level']=level
            

    info['rack']=pd.to_numeric(info['rack'],downcast='unsigned')
    info['roi']=pd.to_numeric(info['roi'],downcast='unsigned')
    info['fov']=pd.to_numeric(info['fov'],downcast='unsigned')
    info['exposure']=pd.to_numeric(info['exposure'],downcast='unsigned')
    
    return info

cycle_data=cycle_info(1,Path('D:/macsima_v2'))
cycle_data.to_csv('D:/cycle_info.csv',index=False)




print(dir(tifff))