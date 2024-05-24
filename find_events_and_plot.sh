#!/bin/bash

# Set the version
version=1

# Run Python scripts with the version argument
python translocation_events.py 
python dip_signals_analysis.py 

# Zip the directory, incorporating the version in the zip file's name
# zip -r plots/signal_feature_analysis_23_00s_300s_soft_v${version}.zip plots/signal_feature_analysis_23_00s_300s_soft_v${version}
