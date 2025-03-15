# ESMM-pytorch

This is unoffical pytorch implementation for [Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate](https://arxiv.org/abs/1804.07931)

## Dataset

- AliExpressDataset: This is a dataset gathered from real-world traffic logs of the search system in AliExpress. This dataset is collected from 5 countries: Russia, Spain, French, Netherlands, and America, which can utilized as 5 multi-task datasets. 

Reference: https://github.com/easezyc/Multitask-Recommendation-Library?tab=readme-ov-file#datasets


## Usage
1. Donwload the AliExpressDataset from the above link.
2. (Optimal) Downsampling dataset.
```bash
poetry run python dataset/preprocess.py
```
3. Run a model.
```bash
poetry run python src/main.py
```

## Results
```bash
epoch : 0 / 5
Train CTR loss : 1690.1747879981995
Train CTCVR loss : 147.66193425946403
Valid CTR AUC : 0.7329004795401588
Valid CTCVR AUC : 0.8444262268562281

epoch : 1 / 5
Train CTR loss : 1650.4933294057846
Train CTCVR loss : 134.32671758241486
Valid CTR AUC : 0.7367752624054503
Valid CTCVR AUC : 0.8498708280114962

epoch : 2 / 5
Train CTR loss : 1643.4910127818584
Train CTCVR loss : 133.33148448949214
Valid CTR AUC : 0.7388665546438932
Valid CTCVR AUC : 0.850372522863628

epoch : 3 / 5
Train CTR loss : 1638.1462603211403
Train CTCVR loss : 132.50975981692318
Valid CTR AUC : 0.741082286774471
Valid CTCVR AUC : 0.8542888035810546

epoch : 4 / 5
Train CTR loss : 1632.8683380186558
Train CTCVR loss : 131.6214914105367
Valid CTR AUC : 0.7429551879248175
Valid CTCVR AUC : 0.8571372397512074

```