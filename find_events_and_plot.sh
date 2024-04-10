#!/bin/bash

# Set the version
version=27

# Run Python scripts with the version argument
python translocation_events.py --version $version
python dip_signals_analysis.py --version $version

# Zip the directory, incorporating the version in the zip file's name
zip -r plots/signal_feature_analysis_07_00s_300s_soft_v${version}.zip plots/signal_feature_analysis_07_00s_300s_soft_v${version}
