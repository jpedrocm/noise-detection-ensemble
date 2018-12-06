# Project

![python](https://img.shields.io/badge/python-3.7-blue.svg)
![status](https://img.shields.io/badge/status-in%20progress-yellow.svg)
![license](https://img.shields.io/badge/license-MIT-green.svg)

This is the final project for the Multiple Classifiers System's class.

## Description

The goal of this project is to implement the proposed method of the paper ["A two-stage ensemble method for the detection of class-label noise"](https://www.sciencedirect.com/science/article/pii/S0925231217317265) and reproduce the experiments in some of the paper datasets, which can be found in the [UCI Machine Learning repository](http://archive.ics.uci.edu/ml/datasets.html). The experimented datasets were: [Blood](http://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center), [Breast](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29), [Chess](http://archive.ics.uci.edu/ml/datasets/Chess+%28King-Rook+vs.+King-Pawn%29), [Heart](http://archive.ics.uci.edu/ml/datasets/Statlog+%28Heart%29), [Ionosfere](http://archive.ics.uci.edu/ml/datasets/Ionosphere), [Liver](http://archive.ics.uci.edu/ml/datasets/Liver+Disorders), [Parkinsons](http://archive.ics.uci.edu/ml/datasets/Parkinsons), [Sonar](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+%28Sonar,+Mines+vs.+Rocks%29) and [Spambase](http://archive.ics.uci.edu/ml/datasets/Spambase).

## Getting Started

### Requirements

* [Python](https://www.python.org/) >= 3.7.1
* [NumPy](http://www.numpy.org/) >= 1.15.4
* [SciPy](https://www.scipy.org/) >= 1.1.0
* [pandas](https://pandas.pydata.org/) >= 0.23.4
* [scikit-learn](http://scikit-learn.org/stable/) >= 0.20.1
* [matplotlib](https://matplotlib.org/) >= 3.0.1


### Installing

* Clone this repository into your machine
* Download and install all the requirements listed above in the given order
* Download the listed datasets in .txt format
* Place all these files inside the data/ folder
* Change their file types to .csv
* Change their filenames according to the names in the ConfigHelper function get_datasets

### Reproducing

* Enter into the project main folder in your local repository
* Run this first command to generate all metrics
```
python main.py
```
* Run this second command to aggregate these metrics and generate the final error table and graphics
```
python aggregate.py
```

## Project Structure

    .            
    ├── data                                  # Datasets files
    ├── results                               # Results files
    ├── src                                   # Source code files
    |   ├── aggregate.py
    |   ├── main.py
    |   ├── majority_filtering.py
    |   ├── noise_detection_ensemble.py
    |   ├── config_helper.py
    |   ├── data_helper.py
    |   ├── io_helper.py
    |   └── metrics_helper.py 
    ├── LICENSE.md
    └── README.md

## Author

* [jpedrocm](https://github.com/jpedrocm)

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.