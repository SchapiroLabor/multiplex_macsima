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


#---CLI-BLOCK---#
parser=argparse.ArgumentParser()

#Mandatory arguments

parser.add_argument('-i',
                    '--input',
                    required=True,
                    help='Directory containing the antigen & bleaching cycles. Use frontslash to specify path.'
                    )

parser.add_argument('-o',
                    '--output',
                    required=True,
                    help='Directory where the stacks will be saved. Use frontslash to specify path.\
                        If directory does not exist it will be created.'
                    )

parser.add_argument('-c',
                    '--cycles',
                    required=True,
                    type=int,
                    nargs='*',
                    help='By default this input accepts two integer numbers which mark the \
                    start and end of the cycles to be taken. Alternatively, if the flag -il is activated \
                    this input will accept a list of any number of specific cycles.'
                    )

#Optional arguments

parser.add_argument('-ofl',
                    '--output_folders_list',
                    action='store_true',
                    help='Activate this flag to save the output images in a list of folders rather than in a tree structure.  The list structure facilitates pointing to the files to run a batch job.'
                    )

parser.add_argument('-il',
                    '--input_mode_list',
                    action='store_true',
                    help='Activate this flag to provide the cycles argument a list of specific cycles of interest.'
                    )

parser.add_argument('-he',
                    '--hi_exposure_only',
                    action='store_true',
                    help='Activate this flag to extract only the set of images with the highest exposure time.'
                    )

parser.add_argument('-nb',
                    '--no_bleach_cycles',
                    action='store_false',
                    help='Activate this flag to deactivate the extraction of the bleaching cycles, i.e \
                        only the antigen images will be extracted.'
                    )

parser.add_argument('-rr',
                    '--remove_reference_marker',
                    action='store_true',
                    help='Set up in the markers.csv file the removal of the reference markers in all cycles except for the first one.'
                    )

args=parser.parse_args()
#---END_CLI-BLOCK---#



#------INPUT BLOCK-------#
device_name='MACSIMA'
cycles_dir=Path(args.input)
stack_path=Path(args.output)
condition=args.input_mode_list

if (condition==False and len(args.cycles)==2):
    start=min(args.cycles)
    end=max(args.cycles)
    cycle_numbers=list(range(start,1+end))

elif condition==True:
    cycle_numbers=args.cycles

else:
    print('Wrong input, try one of the following: \n',
           '1) Range mode: Give only two numbers to the cycles argument to define the start and end of cycles to be stacked.\n',
           '2) List mode: Activate the optional argument -il in the command line so the numbers are read as a list of specific cycles.')

#------ ENDINPUT BLOCK----#

if os.path.exists(stack_path):
    pass
else:
    os.mkdir(stack_path)
#----Extract names of all cycles
#def pull_all_cycles( cycles_path):
#   folders=list(filter(lambda x: 'Cycle' in x ,os.listdir( cycles_path )))
#    aux_list={ int(f.split('Cycle_')[-1]):f for f in folders } 
#   folders=dict( sorted(aux_list.items(),reverse=False) )  
#   #aux_list=sorted(aux_list,key=itemgetter(1),reverse=False)
#   return folders

#cycles_list=pull_all_cycles(cycles_dir)

#---- HELPER FUNCTIONS ----#


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
    

    
    info.insert(len(cycle_info),'exposure_level',np.zeros(info.shape[0]))

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
    

def create_dir(dir_path):
    
    Path(dir_path).mkdir(parents=False, exist_ok=True)



def create_ome(img_info,xy_tile_positions_units,img_path):
    img_name=img_info['name']
    device_name=img_info['device']
    no_of_channels=img_info['no_channels']
    img_size=img_info['xy_img_size_pix']
    markers=img_info['markers']
    exposures=img_info['exposure_times']
    bit_depth=img_info['bit_depth']
    pixel_size=img_info['pix_size']
    pixel_units=img_info['pix_units']
    sig_bits=img_info['sig_bits']
    if pixel_units=='mm':
        pixel_size=1000*pixel_size
        #pixel_units='um'
    no_of_tiles=len(xy_tile_positions_units)
    print(img_info)
    tifff.tiffcomment(img_path,'')
    #--Generate tiff_data_blocks--#
    tiff_block=[]
    #UUID=ome_types.model.tiff_data.UUID(value=uuid4().urn)
    for ch in range(0,no_of_channels):
        tiff_block.append(TiffData(first_c=ch,
                                    ifd=ch,
                                    plane_count=1#,
                                    #uuid=UUID
                                )
                         
                         )
    #--Generate planes block (contains the information of each tile)--#
    plane_block=[]
    #length_units=ome_types.model.simple_types.UnitsLength('µm')
    for ch in range(0,no_of_channels):
        plane_block.append(Plane(the_c=ch,
                                 the_t=0,
                                 the_z=0,
                                 position_x=0,#x=0 is just a place holder
                                 position_y=0,#y=0 is just a place holder
                                 position_z=0,
                                 exposure_time=0
                                 #position_x_unit=pixel_units,
                                 #position_y_unit=pixel_units
                                )
                          )
    #--Generate channels block--#
    chann_block=[]
    for ch in range(0,no_of_channels):
        chann_block.append(Channel(id=ome_types.model.simple_types.ChannelID('Channel:{x}'.format(x=ch)),
                                   color=ome_types.model.simple_types.Color((255,255,255)),
                                   emission_wavelength=1,#place holder
                                   excitation_wavelength=1,#place holder
                               
                                  )
                          )
    #--Generate pixels block--#
    pix_block=[]
    ifd_counter=0
    for t in range(0,no_of_tiles):
        template_plane_block=copy.deepcopy(plane_block)
        template_chann_block=copy.deepcopy(chann_block)
        template_tiffdata_block=copy.deepcopy(tiff_block)
        for ch,mark in enumerate(markers):
            template_plane_block[ch].position_x=xy_tile_positions_units[t][0]
            template_plane_block[ch].position_y=xy_tile_positions_units[t][1]
            template_plane_block[ch].exposure_time=exposures[ch]
            template_chann_block[ch].id='Channel:{y}:{x}:{marker_name}'.format(x=ch,y=100+t,marker_name=mark)
            template_tiffdata_block[ch].ifd=ifd_counter
            ifd_counter+=1
        pix_block.append(Pixels(id=ome_types.model.simple_types.PixelsID('Pixels:{x}'.format(x=t)),
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
                                #physical_size_x_unit=pixel_units,
                                physical_size_y=pixel_size,
                                #physical_size_y_unit=pixel_units,
                                physical_size_z=1.0,
                                planes=template_plane_block,
                                significant_bits=sig_bits,
                                tiff_data_blocks=template_tiffdata_block
                               )
                        )
    #--Generate image block--#
    img_block=[]
    for t in range(0,no_of_tiles):
        img_block.append(Image(id=ome_types.model.simple_types.ImageID('Image:{x}'.format(x=t)),
                               pixels=pix_block[t]
                              )
                        )
    #--Create the OME object with all prebiously defined blocks--#
    ome_custom=OME()
    ome_custom.creator=" ".join([ome_types.__name__,
                                 ome_types.__version__,
                                 '/ python version-',
                                 platform.python_version()
                                ]
                               )
    ome_custom.images=img_block
    ome_custom.uuid=uuid4().urn
    ome_xml=to_xml(ome_custom)
    tifff.tiffcomment(img_path,ome_xml)
    
def setup_coords(x,y,pix_units):
    if pix_units=='mm':
        x_norm=1000*(np.array(x)-np.min(x))#/pixel_size
        y_norm=1000*(np.array(y)-np.min(y))#/pixel_size
    x_norm=np.rint(x_norm).astype('int')
    y_norm=np.rint(y_norm).astype('int')
    #invert y
    y_norm=np.max(y_norm)-y_norm
    xy_tile_positions=[(i, j) for i,j in zip(x_norm,y_norm)]
    return xy_tile_positions

def tile_position(metadata_string):
    #meta_dict = {TAGS[key] : img.tag[key] for key in img.tag_v2}
    #ome=BeautifulSoup(meta_dict['ImageDescription'][0],'xml')
    ome=BeautifulSoup(metadata_string,'xml')
    x=float(ome.StageLabel["X"])
    y=float(ome.StageLabel["Y"])
    stage_units=ome.StageLabel["XUnit"]
    pixel_size=float(ome.Pixels['PhysicalSizeX'])
    pixel_units=ome.Pixels['PhysicalSizeXUnit']
    bit_depth=ome.Pixels['Type']
    significantBits=int(ome.Pixels['SignificantBits'])
    tile_info={'x':x,
               'y':y,
               'stage_units':stage_units,
               'pixel_size':pixel_size,
               'pixel_units':pixel_units,
               'bit_depth':bit_depth,
               'sig_bits':significantBits
              }
    return tile_info

def create_stack(info,
                cycle_no,
                exp_level,
                device,
                output_folder,
                ref_marker='DAPI',
                isbleach=False,
                offset=0):
    
    racks=info['rack'].unique()
    wells=info['well'].unique()

    sorted_markers,sorted_filters=marker_filter_map(info,ref_marker)
    markers_subset=sorted_markers[1::]
    filters_subset=sorted_filters[1::]




    if isbleach:

        stack_prefix='Background'
        tile_path='bleach_full_path'
        cycle_number_tag='bg_'+f'{(offset+cycle_no):03d}'
        target_name='filters'
        target='_'.join(sorted_filters)
        
        sorted_markers=[ 'bg_{n}_{mark}-{filt}'.format(n=f'{cycle_no:03d}',mark=ma,filt=fi) for ma,fi in zip(sorted_markers,sorted_filters) ]
        
    else:
        stack_prefix='Antigen'
        tile_path='antigen_full_path'
        cycle_number_tag=f'{(offset+cycle_no):03d}'
        target_name='markers'
        target='_'.join(sorted_markers)

    with tifff.TiffFile(info[tile_path][0]) as tif:
        img_ref=tif.pages[0].asarray()
    width=img_ref.shape[1]
    height=img_ref.shape[0]
    dtype_ref=img_ref.dtype


                
    #if isbleach:
    #   target_name='filters'
    #    target='_'.join(sorted_filters)
        
    #    sorted_markers=[ 'bg-{n}_{mark}-{filt}'.format(n=f'{cycle_no:03d}',mark=ma,filt=fi) for ma,fi in zip(sorted_markers,sorted_filters) ]
    #else:
    #    target_name='markers'
    #    target='_'.join(sorted_markers)

    
    for r in racks:
        output_levels=[]
        rack_no='rack_{n}'.format(n=f'{r:02d}')
        output_levels.append(rack_no)

        for w in wells:

            well_no='well_{well_name}'.format(well_name=w)
            output_levels.append(well_no)
            groupA=info.loc[(info['rack']==r) & (info['well']==w)]
            rois=groupA['roi'].unique()
            
            for roi in rois:

                roi_no='roi_{n}'.format(n=f'{roi:02d}')
                exp_level_no='exp_level_{n}'.format(n=f'{exp_level:02d}')
                output_levels.append(roi_no)
                output_levels.append(exp_level_no)

                counter=0
                groupB=groupA.loc[groupA['roi']==roi]
                A=groupB.loc[(groupB['exposure_level']=='ref')]
                stack_size_z=A.shape[0]*len(sorted_markers)
                fov_id=groupB.loc[groupB['marker']==ref_marker,'fov'].unique()
                fov_id=np.sort(fov_id)
                
                stack_name='cycle-{C}-{prefix}-exp-{E}-rack-{R}-well-{W}-roi-{ROI}-{T}-{M}.{img_format}'.format(C=cycle_number_tag,
                                                                                                       prefix=stack_prefix,
                                                                                                       E=exp_level,
                                                                                                       R=r,
                                                                                                       W=w,
                                                                                                       ROI=roi,
                                                                                                       T=target_name,#markers or filters
                                                                                                       M=target,
                                                                                                       img_format='ome.tiff'
                                                                                                      )


                X=[]
                Y=[]
                stack=np.zeros((stack_size_z,height,width),dtype=dtype_ref)
                
                exposure_per_marker=[]
                exp=groupB.loc[(groupB['marker']==ref_marker) & (groupB['fov']==1) & (groupB['exposure_level']=='ref'),'exposure'].tolist()[0]
                exposure_per_marker.append(exp)
                for ms,fs in zip(markers_subset,filters_subset):
                    exp=groupB.loc[(groupB['marker']==ms) & (groupB['filter']==fs) & (groupB['fov']==1) & (groupB['exposure_level']==exp_level),'exposure'].tolist()[0]
                    exposure_per_marker.append(exp)
                
                for tile in fov_id:
                    
                    img_ref=groupB.loc[(groupB['marker']==ref_marker) & (groupB['fov']==tile) ,tile_path].tolist()
                    
                
                    if len(img_ref)>0:

                        with tifff.TiffFile(img_ref[0]) as tif:
                            img=tif.pages[0].asarray()
                            metadata=tif.ome_metadata
                        #stack[counter,:,:]=tifff.imread(img_ref[0])
                        stack[counter,:,:]=img
                        tile_data=tile_position(metadata)
                        X.append(tile_data['x'])
                        Y.append(tile_data['y'])
                        counter+=1
                        
                        for ms,fs in zip(markers_subset,filters_subset):
                            img_marker=groupB.loc[(groupB['marker']==ms) & (groupB['filter']==fs) & (groupB['fov']==tile) & (groupB['exposure_level']==exp_level),tile_path].tolist()[0]
                            img=tifff.imread(img_marker)
                            stack[counter,:,:]=img
                            counter+=1
                
                if args.output_folders_list:
                    output_folders_path=output_folder / '--'.join(output_levels) / 'raw'
                else:
                    output_folders_path=output_folder / Path('/'.join(output_levels)) / 'raw'


                if os.path.exists(output_folders_path):
                    pass
                else:
                    os.makedirs(output_folders_path)

                final_stack_path=os.path.join(output_folders_path,stack_name)
                tifff.imwrite(final_stack_path, stack, photometric='minisblack')  
                img_info={'name':stack_name,
                          'device':device_name,
                          'no_channels':len(sorted_markers),
                          'markers':sorted_markers,
                          'filters':sorted_filters,
                          'exposure_times':exposure_per_marker,
                          'xy_img_size_pix':(width,height),
                          'pix_size':tile_data['pixel_size'],
                          'pix_units':tile_data['pixel_units'],
                          'bit_depth':tile_data['bit_depth'],
                          'sig_bits':tile_data['sig_bits']
                          }
                create_ome(img_info,setup_coords(X,Y,img_info['pix_units']),img_path=final_stack_path)
    
    return img_info





def main(): 
    out_ant={
             'cycle_number':[],
             'marker_name':[],
             'Filter':[],
             'background':[],
             'exposure':[],
             'remove':[],
             'exposure_level':[]
             }

    out_ble=copy.deepcopy(out_ant)
    
    #offset_value=1+max(cycle_numbers)

    for cycle_ in cycle_numbers:
        cycle_tiles_info=cycle_info(cycle_,cycles_dir,ref_marker='DAPI')
        exp=cycle_tiles_info['exposure_level'].unique()
        exp=exp[exp!='ref']
        exp.sort()

        if args.hi_exposure_only:
            exp=[max(exp)]
        else:
            pass

        print('extracting cycle:',cycle_)

        extract_bleach=args.no_bleach_cycles
            

        for e in exp:
            antigen_stack_info=create_stack(cycle_tiles_info,cycle_,e,device_name,stack_path)
            out_ant['cycle_number'].extend(antigen_stack_info['no_channels']*[cycle_])
            out_ant['marker_name'].extend(antigen_stack_info['markers'])
            out_ant['Filter'].extend(antigen_stack_info['filters'])
            out_ant['remove'].extend(antigen_stack_info['no_channels']*[''])
            out_ant['exposure'].extend(antigen_stack_info['exposure_times'])
            out_ant['exposure_level'].extend(antigen_stack_info['no_channels']*[e])

            if extract_bleach:
                print('extracting bleaching cycle:',cycle_)
                bleach_stack_info=create_stack(cycle_tiles_info,cycle_,e,device_name,stack_path,isbleach=True)
                background_channels=['']#the blank string corresponds to the reference marker, it is always the first in the sorted_markers list
                for fi in antigen_stack_info['filters'][1::]:
                    #ch_name=['bg-{n}_{mark_filt}'.format(mark_filt=x,n=f'{cycle_:03d}') for x in bleach_stack_info['markers'] if fi in x]
                    ch_name=[x for x in bleach_stack_info['markers'] if fi in x]
                    background_channels.extend(ch_name)
                
                out_ant['background'].extend(background_channels)

                out_ble['background'].extend(bleach_stack_info['no_channels']*[''])
                out_ble['cycle_number'].extend(bleach_stack_info['no_channels']*[cycle_])
                out_ble['marker_name'].extend(bleach_stack_info['markers'])
                out_ble['Filter'].extend(bleach_stack_info['filters'])
                out_ble['remove'].extend(bleach_stack_info['no_channels']*['TRUE'])
                out_ble['exposure'].extend(bleach_stack_info['exposure_times'])
                out_ble['exposure_level'].extend(bleach_stack_info['no_channels']*[e])
                
            else:
                out_ant['background'].extend(antigen_stack_info['no_channels']*[''])


    for e in exp:
        print(e)
        if extract_bleach:
            for key,value in out_ant.items():
                print(key,':',value)
            df1=pd.DataFrame(out_ant).groupby('exposure_level').get_group(e)
            df2=pd.DataFrame(out_ble).groupby('exposure_level').get_group(e)
            df=pd.concat([df1,df2],ignore_index=True)
        else:
            df=pd.DataFrame(out_ant).groupby('exposure_level').get_group(e)

        df.drop('exposure_level',axis=1,inplace=True)
        df.insert(0,'channel_number',list(range(1,1+df.shape[0])))

        if args.remove_reference_marker:
            dna_indices=df.loc[df['marker_name']=='DAPI'].index.values
            for n,i in enumerate(dna_indices):
                if n>0:
                    df.at[i,'remove']='TRUE'

        df.to_csv( stack_path / 'markers_exp_{level}.csv'.format(level=e),index=False)


if __name__=='__main__':
    main()
            
            
    
    