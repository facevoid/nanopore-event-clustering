
# Translocating Event Analysis of Nanopore

## Description

This repository enables the analysis of raw data in the .abf format, primarily focused on the study of translocating events through nanopores. It offers a range of features for comprehensive signal analysis, including:

1. **Identification of Translocating Events:** Automatically detect events within the raw data.
2. **Deep Analysis of Signals:** Apply various smoothing techniques for in-depth signal analysis.
3. **Clustering:** Utilize k-means or hierarchical clustering to group events, find the optimal number of clusters, and more. Additional clustering methods will be introduced in future updates.
4. **Similar Signal Retrieval:** Find signals that resemble each other within the dataset.
5. **ML-Based Classification:** (Upcoming feature) Classify signals using machine learning techniques.

## Installation

### Requirements

Before starting, ensure you have Python installed on your system. This project has been tested on Python 3.8+. You can install all necessary dependencies by running:

```sh
pip install -r requirements.txt
```

### Setting Up

Clone this repository to your local machine using:

```sh
git clone https://github.com/yourusername/translocating-event-analysis.git
cd translocating-event-analysis
```

## Usage

To utilize the features of this repository, follow the steps outlined below for each functionality.

## Features

### Identifying Events and Analysis

For identifying translocating events, deep signal analysis, and generating plots, run the provided bash script:

```sh
bash find_and_analyze.sh
```
The analysis behavior can be customized by modifying the `configs/event-analysis.yaml` configuration file. This file contains important parameters that control various aspects of the analysis, including paths to data files, smoothing techniques parameters, and versioning for output directories. Update the configuration file according to your specific requirements before running the analysis scripts.

Example configuration parameters include:

- `version`: Specifies the version of the analysis, affecting output directory naming.
- `data_file_path`: The path to the raw .abf data file for analysis.
- `sampling_rate`, `base_sigma`, `gaussian_sigma`, etc.: Parameters that control the data processing and analysis techniques.

**Notes:**

-  `plots/signal_feature_analysis_07_00s_300s_soft/dip_100_start_72.254200s_end_72.261580s.png` and `plots/signal_feature_analysis_07_00s_300s_soft/dip_112_start_80.829836s_end_80.830548s.png` with the actual paths to your sample images. Upload these images to your repository or a suitable hosting service to ensure they're accessible from the README.

### Finding Optimal Cluster Size

To find the optimal cluster size in your data:

```sh
python some_script.py
```

### Clustering with K-Means or Hierarchical Methods

For applying hierarchical and k-means clustering:

```sh
python some_script.py
```

### Retrieving Similar Events

To find similar translocating events within the dataset:

```sh
python some_script.py
```

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to fork this repository, make your changes, and submit a pull request.

## License

[MIT License](LICENSE.txt)
