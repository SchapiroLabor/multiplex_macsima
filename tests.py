import pandas as pd
from pathlib import Path

frame=pd.read_csv(Path("D:/test_folder/frame.csv"))
frame.loc[ (frame['marker']=='Syk') & (frame['filter']=='FITC') ].full_path