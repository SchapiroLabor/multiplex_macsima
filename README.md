# About this repository
This repository contains the script macsima2mc.py, this script stacks the individual image files (.tif) produced by the MACSima system and saves the stack into an ome.tiff file.

The input of the script is the path to the directory that contains the images of the antigen and bleaching cycles, i.e the path to 001_AntigenCycle,001_BleachCycle,002_AntigenCycle...etc.  
The output of the script is a stack for each cycle.  These stacks are saved into a hierarchical file structure based on the following acquisition parameters: rack_number->well_number->roi_number->exposure_level.  
Along with the stacks a .csv file is produced for each exposure level used in the acquisition, the structure of this file is the required format by the markers.csv file used in MCMICRO.
The stacks can be transfered to the raw folder of MCMICRO and the markers.csv used to run the analysis pipeline.

# The script macsima2mc.py
## CLI 
### Required arguments:
* --input, -i  : path to the folder containing the antigen and bleaching cycles.
* --output, -o : path where the output, i.e. stacks and csv files, will be saved.  If the path does not exist it will be created.
* --cycles, -c : two integer numbers marking a range, i.e. the beginning and end of the cycles to be taken.
### Optional flags: 
* --input_mode_list, -il : if activated the argument cycles will be interpreted as a list of numbers.  Use this flag to provide -c with a list of numbers of specific cycles to be taken.
* --high_exposure_only, -he : activate this flag to extract only the set of images acquired with the highest exposure time.
* --no_bleach_cycles, -nb : if activated the bleach cycles will not be extracted.
### CLI help
To print the description of the arguments in the command prompt run:

```
python macsima2mc.py -h
```
# Examples
The following command will read the images from the directory "/data/folder/with/cycles", bleaching and antigen cycles from 1 to 20 will be taken and their stacks written in "/output/folder/with/stacks". 
```
python macsima2mc.py -i /data/folder/with/cycles -o /output/folder/with/stacks -c 1 20
```
Similar example as the one above but this time only cycles 1,2,3 and 5 will be taken (-il).  Bleaching cycles will not be considered (-nb). 

```
python macsima2mc.py -i /data/folder/with/cycles -o /output/folder/with/stacks -il -c 1 2 3 5 -nb
```
# Using the docker image
1.- pull the image:
```
docker pull ghcr.io/schapirolabor/multiplex_macsima:main
```
2.- run container interactively in command prompt:

```
docker run -it -v /local/folder/with/cycles:/folder/inside/docker image_id bash 
```
The path /folder/inside/docker can be any directory inside the container, /media is an option for it.

3.- run script insde the container:

```
python macsima2mc.py -i /media -o /media/output -c 1 20
```





