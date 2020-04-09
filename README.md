# Iterative Feature Normalisation for Emotional Speech Recognition

This repository contains a Python implementation of the paper:

>[Iterative Feature Normalisation for Emotional Speech Recognition *Carlos Busso*, *Angeliki Metallinou* and *Shrikanth S. Narayanan*  *ICASSP 2011*](https://ieeexplore.ieee.org/document/5947652)

This repository implements IFN and its variants as mentioned in the paper: 
- Global Normalisation
- Optimal Initialisation
- Without Normalisation
- Ablation Study

For further details on the results and analysis, please refer to the report - ```report.pdf``` 

## Installing Dependencies

Make sure you have a Python3+ version. Run the following command - 

```
pip install -r requirements.txt
```

## Data Preprocessing

Download the __RAVDESS__ dataset from [here](https://zenodo.org/record/1188976#.XnrzbYAzbkw). Put it inside a folder titles ```data/``` in the root directory. 

The preprocessing can be done with the help of ```data_organisation.py```, using -

```
usage: data_organisation.py [-h] --base_path BASE_PATH --processed_path
                            PROCESSED_PATH

Preprocess RAVDESS data for IFN

optional arguments:
  -h, --help            show this help message and exit
  --base_path BASE_PATH
                        Path of the original RAVDESS data
  --processed_path PROCESSED_PATH
                        Path to put the processed files in
 
 ```
 
## Training the Models

For further details on the description of hyperparameters, features used and other analysis, refer to ```report.pdf```.

The training can be done with the help of ```train.py```, using - 
```
usage: train.py [-h] --experiment {norm,without,global,optimal,ablation}
                [--train_path TRAIN_PATH] [--ref_path REF_PATH]
                [--write_path WRITE_PATH] [--dataframe_path DATAFRAME_PATH]
                [--threshold THRESHOLD] [--upper_cap UPPER_CAP]
                [--lower_cap LOWER_CAP]

Train Iterative Feature Normalisation and its variants

optional arguments:
  -h, --help            show this help message and exit
  --experiment {norm,without,global,optimal,ablation}
                        Type of variant to run. A
  --train_path TRAIN_PATH
                        Path to the train set
  --ref_path REF_PATH   Path to the reference corpus
  --write_path WRITE_PATH
                        Path to save the models
  --dataframe_path DATAFRAME_PATH
                        Path to the dataframes if available
  --threshold THRESHOLD
                        Classification Threshold
  --upper_cap UPPER_CAP
                        Upper cap on the scaling factor values
  --lower_cap LOWER_CAP
                        Lower cap on the scaling factor values

```
## Analysis

The training can be done with the help of ```analysis.py```, using - 
```
usage: analysis.py [-h] --read_path READ_PATH --experiment_type
                   EXPERIMENT_TYPE

Run the analysis script

optional arguments:
  -h, --help            show this help message and exit
  --read_path READ_PATH
                        Path to the models
  --experiment_type EXPERIMENT_TYPE
                        Path to the experiment data

```

## License 

Copyright (c) 2020 Brihi Joshi

For license information, see [LICENSE](LICENSE) or http://mit-license.org


- - -

This code was written as a part of a course assignment in **Affective Computing** with [Dr. Jainendra Shukla](https://www.iiitd.ac.in/jainendra) at IIIT Delhi during Winter 2020 Semester. 

For bugs in the code, please write to: brihi16142 [at] iiitd [dot] ac [dot] in

