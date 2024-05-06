<!-- markdownlint-disable MD033 -->

# GraphEGFR
GraphEGFR: Multi-task and Transfer Learning Based on Molecular Graph Attention Mechanism and Fingerprints Improving Inhibitor Bioactivity Prediction for EGFR Family Proteins on Data Scarcity

![GraphEGFR architecture](https://github.com/manbaritone/GraphEGFR/blob/main/graphegfr_architect.png)

## Instructions

The file structure of the project is shown as the following diagram

```text
GraphEGFR
 ├─configs
 ├─examples
 ├─graphegfr
 ├─misc
 ├─resources
 │  └─LigEGFR
 ├─state_dict
 ├─run-colab.ipynb
 ├─run.ipynb
 ├─run.py
 └─README.md
```

To run the experiment with specific configuration, enter the following script

```python
python3 run.py --config configs/sample.json
```

There are several options to set up in the configuration file:


- `target` - selected proteins used in the study
- `hyperparam` - configuration for model building process (in json format)
- `result_folder` *[optional]* - the directory where the results will be stored
- `database` *[optional]* - identify which database to obtain data (only option available currently is `LigEGFR`; can be omitted)
- `metrics` *[optional]* - a list of string representing the metrics to report in the experiment. The available options are
    `RMSE`,
    `MAE`,
    `MSE`,
    `PCC`,
    `R2`,
    `SRCC`

## Datasets and Pretrained Models Availability
The datasets and pretrained models can be retrieved from [![https://doi.org/10.5281/zenodo.11122146](https://zenodo.org/badge/DOI/10.5281/zenodo.11122146.svg)](https://doi.org/10.5281/zenodo.11122146).

## Dependencies

packages             | version
-------------------- | ----------
arrow                | 1.2.2
deepchem             | 2.5.0
imbalanced-learn     | 0.10.1
numpy                | 1.21.5
pandas               | 1.3.5
scikit-learn         | 1.2.2
scipy                | 1.7.3
torch                | 2.0.0
torch-geometric      | 2.0.4
torchmetrics         | 0.11.4
xgboost              | 1.6.1
dgl                  | 1.1.3
dgllife              | Any
