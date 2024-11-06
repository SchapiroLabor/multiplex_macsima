import ome_schema as schema
import pandas as pd


def create_ome(tile_info,
               conformed_markers):
    """
    This function creates an OME-XML file from a pandas dataframe containing the metadata of the tiles.
    Args:
        tile_info (pd.DataFrame): dataframe containing the metadata of the tiles.
        conformed_markers (list): list of tuples with the name of the markers and their corresponding fluorophore.
    Returns:
        str: OME-XML file.
    """

    grouped_tiles = tile_info.groupby(['tile'])
    no_of_channels = len(conformed_markers)
    tiles_counter = 0
    image = []
    for tileID, frame in grouped_tiles:
        metadata = schema.INPUTS(frame, conformed_markers)
        tiff = schema.TIFF_array(no_of_channels, inputs={'offset': no_of_channels * tiles_counter})
        plane = schema.PLANE_array(no_of_channels, metadata)
        channel = schema.CHANN_array(no_of_channels, metadata)
        pixels = schema.PIXELS_array(channel, plane, tiff, metadata)
        image.append(schema.IMAGE_array (pixels, tiles_counter))
        tiles_counter += 1
    ome, ome_xml = schema.OME_metadata(image)

    return ome, ome_xml
