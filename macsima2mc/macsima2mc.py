import tools
from templates import macsima_pattern
import CLI
import mc_tools



#input_test_folder=Path("D:/macsima_data_samples/macsima_data_v2/8_Cycle2")
#output_test_folder=Path('D:/test_folder')

def main(args):
    input=args.input
    output=args.output
    ref=args.reference_marker
    basicpy_corr=args.illumination_correction 
    cycle_info=tools.cycle_info( input , macsima_pattern(version=2),ref_marker= ref)
    cycle_info=tools.append_metadata( cycle_info )
    #cycle_info.to_csv( args.output / 'cycle_{c}_info.csv'.format(c=f'{6:03d}'), index=False )
    output_dirs=tools.create_stack( cycle_info, output ,ref_marker=ref,hi_exp=args.hi_exposure_only,ill_corr=basicpy_corr)
    [mc_tools.write_markers_file(path)  for path in output_dirs]
        

if __name__ == '__main__':
    args = CLI.get_args()
    main(args)






