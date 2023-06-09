<!-- markdownlint-disable MD033 -->

# GraphEGFR

## Instructions

The file structure of the project is shown as the following diagram

```text
GraphEGFR
 ├─graphegfr
 ├─resources
 ├─configs
 │  ├─setting1.json
 │  └─setting2.json
 ├─run.py
 └─README.md
```

To run the experiment with specific configuration, enter the following script

```python
python3 run.py --config configs/settings1.json
```

There are several options to set up in the configuration file:

- `result_folder` - the directory where the results will be stored
- `database` - identify which database to obtain data
- `target` - selected proteins used in the study.
  The table below shows datasets availability of corresponding proteins in  the database used in our project.

  <!-- <html>
    <style>
      td.right-bound {
        border-right: 1px solid;
      }
    </style>
    <table>
      <tr>
        <th>Target \ Database</th>
        <th>LigEGFR</th>
        <th>KIP</th>
        <th>BindingDB</th>
      </tr>
      <tr>
        <td class="right-bound">HER1</td>
        <td> &#x2713; </td>
        <td> &#x2713; </td>
        <td> &#x2713; </td>
      </tr>
      <tr>
        <td class="right-bound">HER2</td>
        <td> &#x2713; </td>
        <td> &#x2713; </td>
        <td> &#x2713; </td>
      </tr>
      <tr>
        <td class="right-bound">HER3</td>
        <td> &#x2713; </td>
        <td> &#x2713; </td>
        <td> &#x2713; </td>
      </tr>
    </table>
  </html> -->
  Target | LigEGFR | KIP | BindingDB
  ------ | ------- | --- | ---------
  HER1   | &#x2713; | &#x2713; | &#x2713;
  HER2   | &#x2713; | &#x2713; | &#x2713;
  HER3   | &#x2713; | &#x2713; | &#x2713;

- `metrics` - a list of string representing the metrics to report in the experiment. The available options are
  - Classification
    - `Accuracy`
    - `AUCPR`
    - `AUROC`
    - `Balanced_Accuracy`
    - `BCE`
    - `F1`
    - `GMeans`
    - `Kappa`
    - `MCC`
    - `Precision`
    - `Recall`
    - `Specificity`
  - Regression
    - `RMSE`
    - `MAE`
    - `MSE`
    - `PCC`
    - `R2`
    - `SRCC`

## Dependencies

packages             | version
-------------------- | ----------
arrow                | 1.2.2
deepchem             | 2.5.0
imbalanced-learn     | 0.10.1
numpy                | 1.21.5
pandas               | 1.3.5
scikit-learn         | 1.0.2
scipy                | 1.7.3
torch                | 1.10.1
torch-geometric      | 2.0.4
torchmetrics         | 0.11.4
xgboost              | 1.6.1
