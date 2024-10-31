import argparse
import pathlib



#---CLI-BLOCK---#
def get_args():     
    parser=argparse.ArgumentParser()

    #Mandatory arguments

    parser.add_argument('-i',
                    '--input',
                    required=True,
                    type=pathlib.Path,
                    help='Path to the cycle folder'
                    )

    parser.add_argument('-o',
                    '--output',
                    required=True,
                    type=pathlib.Path,
                    help='Path where the stacks will be saved. If directory does not exist it will be created.'
                    )

    parser.add_argument('-rm',
                    '--reference_marker',
                    default='DAPI',
                    help='string specifying the name of the reference marker'
                    )

    parser.add_argument('-ic',
                    '--illumination_correction',
                    action='store_true',
                    help='Applies illumination correction to all tiles, the illumination profiles are created with basicpy'
                    )


    parser.add_argument('-he',
                    '--hi_exposure_only',
                    action='store_true',
                    help='Activate this flag to extract only the set of images with the highest exposure time.'
                    )

    parser.add_argument('-rr',
                    '--remove_reference_marker',
                    action='store_true',
                    help='It will mark the removal of the reference markers in the markers.csv except for the first cycle.'
                    )

    args=parser.parse_args()

    return args
#---END_CLI-BLOCK---#



