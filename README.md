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