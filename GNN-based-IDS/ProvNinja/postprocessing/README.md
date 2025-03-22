# Postprocessing Script for DGL

This directory contains scripts that process JSON graphs and convert them to CSV format for use with DGL framework. This DGL [Loading data from csv files]( https://www.dgl.ai/dgl_docs/en/0.8.x/guide/data-loadcsv.html) tutorial was followed.

## 1. Convert Graph Json to CSV

```shell
$ python dgl_dataprocessing.py -i ..\..\Data
```

### output

```shell
$ python dgl_dataprocessing.py -i ..\..\Data\
100%|██████████████████████████████████████████████████████████████████████████████████| 1042/1042 [05:23<00:00,  3.22it/s]
```

## 2. Categorize Graphs into Train/Validation/Test

```shell
$ python folder_categorization.py 0.7 0.1 0.2 -d ..\..\Data\FiveDirections\ -bf benign-500 -af anomaly
$ python folder_categorization.py 0.7 0.1 0.2 -d ..\..\Data\FiveDirections-DP\ -bf benign-500-dp_N=500_epsilon=1.0_delta=0.5_alpha=0.7_beta=0.1_gamma=0.1_eta=0.1_k=0.1 -af anomaly
```

### output

```shell
python .\folder_categorization.py 0.7 0.1 0.2 -d ..\..\Data\FiveDirections-DP\ -bf benign-500 -af anomaly
# of benign folder found: 500
# of anomaly folder found: 21
Done.

python .\folder_categorization.py 0.7 0.1 0.2 -d ..\..\Data\FiveDirections-DP\ -bf benign-500-dp_N=500_epsilon=1.0_delta=0.5_alpha=0.7_beta=0.1_gamma=0.1_eta=0.1_k=0.1 -af anomaly
# of benign folder found: 500
# of anomaly folder found: 21
Done.
```