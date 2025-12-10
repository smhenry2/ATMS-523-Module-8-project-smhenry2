#!/bin/bash -l
#PBS -N 12_2021
#PBS -A UIUC0059
#PBS -l walltime=06:00:00
#PBS -q casper
#PBS -j oe 
#PBS -k eod 
#PBS -m abe 
#PBS -M smhenry@ucar.edu
#PBS -l select=1:ncpus=1:mem=16GB

source ~/.bashrc #configures your shell to use conda activate
conda activate analysis

echo ""
echo "#Job Begin"
echo ""

###Run the executable
python 2_process_ERA5.py  #change as needed

echo "DONE"

