import ome_schema as schema
import pandas as pd


df=pd.read_csv("D:/test_folder/cycle_006_info.csv")

acq_group=df.groupby(['source','rack','well','roi','exposure_level'])
acq_index=list(acq_group.indices.keys())
group=acq_group.get_group(('B', 1, 'B01', 1, 1))
group.to_csv("D:/test_folder/tile_info.csv")

def create_ome(tile_info,conformed_markers):
    grouped_tiles=tile_info.groupby(['tile'])
    no_of_channels=len(conformed_markers)
    tiles_counter=0
    image=[]
    for tileID,frame in grouped_tiles:
        metadata=schema.INPUTS(frame, conformed_markers)
        tiff=schema.TIFF_array( no_of_channels, inputs={'offset':no_of_channels*tiles_counter} )
        plane=schema.PLANE_array(no_of_channels, metadata)
        channel=schema.CHANN_array(no_of_channels,metadata)
        image.append( schema.IMAGE_array ( 
            schema.PIXELS_array(channel,plane,tiff,metadata),
             tiles_counter 
             ) 
             )
        tiles_counter=+1

    ome_xml=schema.OME_xml(image)

    return ome_xml

    

