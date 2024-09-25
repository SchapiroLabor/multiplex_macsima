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

    for tileID,frame in grouped_tiles:
        for marker,filter in conformed_markers:
                metadata=frame.loc[ (frame['marker']==marker) & (frame['filter']==filter) ]
                metadata.position_x.values[0]
                

        tiff=schema.TIFF_block(no_of_channels,
            inputs={'offset':n*no_of_channels}
            )
        plane=schema.PLANE_block(no_of_channels,
            inputs={'position_x':0,'position_x_unit':'','position_y':0,'position_y_unit':'','exposure_time':[]}
            )
        ch=schema.CHANN_block()
        

        a=[schema.IMAGE_block(

            schema.PIXELS_block( inputs={'tile_id':0,'size_x':0,'size_y':0,'type':'',
                        'chann_block':[],'pix_size_x':0,'pix_size_y':0,
                        'pix_size_x_units':'','pix_size_y_units':'',
                        'plane_block':[],'sig_bits':0,'tiff_block':[]
                        }

            }

            )


        )]


