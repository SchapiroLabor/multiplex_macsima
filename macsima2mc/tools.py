from templates import info_dic
import re
import pandas as pd
import tifffile as tifff
from bs4 import BeautifulSoup
import numpy as np
from pathlib import Path
import ome_writer


def merge_dicts(list_of_dicts):
    merged_dict = {}
    for d in list_of_dicts:
        for key, value in d.items():
            if key in merged_dict:
                merged_dict[key].append(value)
            else:
                merged_dict[key] = [value]
    return merged_dict

def extract_values(target_pattern, strings,number_cast=True):
    return [
        (int(m.group(1)) if number_cast else m.group(1))
        if (m := re.search(target_pattern, s))
        else None
        for s in strings
    ]


def extract_metadata(tile_abs_path):

    with tifff.TiffFile(tile_abs_path) as tif:
            metadata = tif.ome_metadata

    ome = BeautifulSoup(metadata, "xml")
    return {
            "position_x": float(ome.StageLabel["X"]),
            "position_y": float(ome.StageLabel["Y"]),
            "position_x_unit": ome.StageLabel["XUnit"],
            "position_y_unit": ome.StageLabel["YUnit"],
            "physical_size_x": float(ome.Pixels["PhysicalSizeX"]),
            "physical_size_x_unit": ome.Pixels["PhysicalSizeXUnit"],
            "physical_size_y": float(ome.Pixels["PhysicalSizeY"]),
            "physical_size_y_unit": ome.Pixels["PhysicalSizeXUnit"],
            "size_x":ome.Pixels["SizeX"],
            "size_y":ome.Pixels["SizeY"],
            "type": ome.Pixels["Type"],#bit_depth
            "significant_bits": int(ome.Pixels["SignificantBits"]),
            "emission_wavelenght":ome.Channel["EmissionWavelength"],
            "excitation_wavelenght":ome.Channel["ExcitationWavelength"],
            "emission_wavelenght_unit":ome.Channel["EmissionWavelengthUnit"],
            "excitation_wavelenght_unit":ome.Channel["ExcitationWavelengthUnit"]
            }



def cycle_info(cycle_path, platform_pattern,ref_marker= 'DAPI'):
    '''
    This function reads the images produced by the MACSima device and returns the acquistion information
    specified in the image name.
    inputs:
    -cycle_path[Path]= full path to the cycle folder
    -ref[str]=marker of reference used for registration
    -source[str]= valid values 'Antigen' or 'Bleach'
    -dir_version[int]=version of the macsima folder and file naming structure.  Valid values are 1 or 2.
        E.g. version_1 (001_AntigenCycle_DAPI_V0_DAPI_16bit_M-20x-S Fluor full sensor_B-1_R-2_W-2_G-1_F-30_E-16.0.tif) and 
        version_2 (CYC-001_SCN-001_ST-B_R-01_W-B01_ROI-001_F-001_A-Syk_C-_D-FITC_EXP-17.5781.tif)
    output:
    -info[dict]=dictionary with acquisition information, ROI, rack, exposure time etc.

    '''

    full_image_paths = list(cycle_path.glob("*.tif"))
    file_names = [x.name for x in full_image_paths]

    info=info_dic(platform_pattern)

    info['full_path']=full_image_paths
    info['img_name']=file_names

    for feat,value in platform_pattern.items():

        info[feat]=extract_values(target_pattern=value, strings=file_names,number_cast=False)

    df=pd.DataFrame(info)
    df.loc[df['filter']==ref_marker,'marker']=ref_marker

    df.insert(loc=df.shape[1], column="exposure_level", value=0)
    df["exposure_time"] = df["exposure_time"].astype(float)
    df["exposure_level"] = ( df.groupby(["source","marker","filter"])["exposure_time"].rank(method="dense")).astype(int) 


    return df


def append_metadata(cycle_info_df):

    pos=list( map(extract_metadata, cycle_info_df['full_path'].values) )

    for key,val in merge_dicts(pos).items():
        cycle_info_df.insert(loc=cycle_info_df.shape[1], column=key, value=val)

    return cycle_info_df

def conform_markers(mf_tuple,ref_marker='DAPI'):

    markers=[tup for tup in mf_tuple if tup[0]!=ref_marker]
    markers.insert(0,(ref_marker,ref_marker))
    return markers

def any_ref(mf_tuple,ref_marker='DAPI'):
    exist_ref=False
    for m in mf_tuple:
        if m[0]==ref_marker:
            exist_ref=True
            break
    return exist_ref



def init_stack(ref_tile_index,groupby_obj,marker_filter_map):
    ref_tile=groupby_obj.get_group((ref_tile_index,))
    total_tiles=len(groupby_obj)
    width=ref_tile.size_x.values[0]
    height=ref_tile.size_y.values[0]
    depth=total_tiles*len(marker_filter_map)
    stack=np.zeros( (depth,int(height),int(width)) ,dtype=ref_tile.type.values[0] )

    return stack

def cast_stack_name(cycle_no,acq_group_index,marker_filter_map):
    #acq_group_index('source','rack','well','roi','exposure_level')
    markers='__'.join([element[0] for element in marker_filter_map ])
    filters='__'.join([element[1] for element in marker_filter_map ])
    cycle_no=int(cycle_no)

    name='cycle-{C}-src-{S}-rack-{R}-well-{W}-roi-{ROI}-exp-{E}-markers-{M}-filters-{F}.{img_format}'.format(
                                                                                                    C=f'{cycle_no:03d}',
                                                                                                    S=acq_group_index[0],
                                                                                                    E=acq_group_index[4],
                                                                                                    R=acq_group_index[1],
                                                                                                    W=acq_group_index[2],
                                                                                                    ROI=acq_group_index[3],
                                                                                                    M=markers,
                                                                                                    F=filters,
                                                                                                    img_format='ome.tiff'
                                                                                                    )

    return name

def cast_outdir_name(tup):
    #tuple('source','rack','well','roi','exposure_level'])
    name='rack-{R}-well-{W}-roi-{ROI}-exp-{E}'.format(
                                                R=tup[1],
                                                W=tup[2],
                                                ROI=tup[3],
                                                E=tup[4]
                                                    )

    return name

def outputs_dic():

    out={'index':[],
        'array':[],
        'full_path':[],
        'ome':[],

        }

    return out

def select_by_exposure(list_indices,exp_index=4,target='max'):
    selected_indices=[]
    df_aux=pd.DataFrame( np.row_stack(list_indices) )
    group_by_indices=np.setdiff1d( range(0, len(list_indices[0]) ), exp_index ).tolist()

    for key,frame in df_aux.groupby( group_by_indices ):
        if target=='max':
            selected_indices.append( key + ( int(frame[exp_index].max() ), ) )
        elif target=='min':
            selected_indices.append( key + ( int( frame[exp_index].min()), ) )

    return selected_indices

def create_stack(cycle_info_df,output_dir,ref_marker='DAPI',hi_exp=False,extended_outputs=False):

    if extended_outputs:
        out=outputs_dic()
    else:
        out={'output_paths':[]}

    acq_group=cycle_info_df.groupby(['source','rack','well','roi','exposure_level'])
    acq_index=list( acq_group.indices.keys() )

    if hi_exp:
        acq_index=select_by_exposure(acq_index)

    for index in acq_index:
        stack_output_dir=output_dir / cast_outdir_name(index) / 'staged'
        ( stack_output_dir ).mkdir(parents=True, exist_ok=True)
        group=acq_group.get_group(index)
        #use tile 1 as reference to determine the heigh and width of the tiles
        tile_no=group.tile.values
        ref_tile=group.groupby(['tile']).get_group((tile_no[0],))
        marker_filter_map=list(ref_tile.groupby(["marker","filter"]).indices.keys())
        exist_ref=any_ref(marker_filter_map,ref_marker)
        if not exist_ref:
            index_aux=list(index)
            index_aux[-1]=1
            index_aux=tuple(index_aux)
            aux_group=acq_group.get_group(index_aux)
            aux_group=aux_group.loc[aux_group['marker']==ref_marker]
            group=pd.concat( [group,aux_group] )

        #group.to_csv(stack_output_dir.parent.absolute() /'info.csv' )
        groups_of_tiles=group.groupby(['tile'])
        conformed_markers =conform_markers(marker_filter_map,ref_marker)
        stack=init_stack(tile_no[0],groups_of_tiles,conformed_markers)
        ome=ome_writer.create_ome(group,conformed_markers)
        counter=0
        for tile_no,frame in groups_of_tiles:
            for marker,filter in conformed_markers:
                target_path=frame.loc[ (frame['marker']==marker) & (frame['filter']==filter) ].full_path.values[0]
                stack[counter,:,:]=tifff.imread(Path(target_path))
                counter+=1
        stack_name =cast_stack_name(frame.cycle.iloc[0],index,conformed_markers)
        stack_file_path= stack_output_dir/ stack_name

        if extended_outputs:
            out['index'].append(index)
            out['array'].append(stack)
            out['full_path'].append(stack_full_path)
            out['ome'].append(ome)
        else:
            out['output_paths'].append( stack_output_dir )
            tifff.imwrite( stack_file_path , stack, photometric='minisblack' )
            ome,ome_xml=ome_writer.create_ome(group,conformed_markers)
            tifff.tiffcomment(stack_file_path, ome_xml)
        
    if extended_outputs:
        return out
    else:
        return np.unique( out['output_paths'] )
