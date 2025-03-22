# ProvNinja [[1]](#references)

This folder contains the ProvNinja GNN-based IDS scripts and helper utilities.

```shell
$ python provgat.py -if 768 -hf 64 -lr 0.001 -e 50 -n 7 -bs 16 -dl 'C:\Users\prov-dp\GNN-based-IDS\Data\FiveDirections\\' --device cpu -at 0.01
```

```shell
$ python provgat.py -if 768 -hf 64 -lr 0.001 -e 50 -n 7 -bs 16 -dl 'C:\Users\prov-dp\GNN-based-IDS\Data\FiveDirections-DP\\' --device cpu -at 0.01
```

### Inference Output

#### ProvNinja using [Original FiveDirections](../Data/FiveDirections/) Dataset

```shell
$ python provgat.py -if 768 -hf 64 -lr 0.001 -e 50 -n 7 -bs 16 -dl 'C:\Users\prov-dp\GNN-based-IDS\Data\FiveDirections\\' --device cpu -at 0.01  
2025-03-20 17:07:39,601 | INFO  | using C:\Users\prov-dp\GNN-based-IDS\Data\FiveDirections\ as input data directory
2025-03-20 17:07:39,602 | INFO  | 7 Layer GAT. Input Feature Size: 768. Hidden Layer Size(s): 64. Loss Rate: 0.001. Batch Size: 16
2025-03-20 17:07:39,602 | INFO  | Input Device: cpu
2025-03-20 17:07:39,602 | INFO  | Variable Prediction Threshold for Anomalous graphs have been enabled & set to 0.01
2025-03-20 17:07:39,602 | INFO  | Training on 50 epochs...
Done loading data from cached files.
Done loading data from cached files.
Done loading data from cached files.
2025-03-20 17:07:40,218 | INFO  | Length of dataset: 521
2025-03-20 17:07:40,248 | INFO  | Evaluating on Device: cpu
2025-03-20 17:07:40,248 | INFO  | # Parameters in model: 1501505
2025-03-20 17:07:40,249 | INFO  | # Trainable parameters in model: 1501505
2025-03-20 17:07:40,250 | INFO  | Stratified sampler enabled
2025-03-20 17:08:09,715 | INFO  | === test stats ===
2025-03-20 17:08:09,715 | INFO  | Number Correct: 97
2025-03-20 17:08:09,716 | INFO  | Number Graphs in test Data: 105
2025-03-20 17:08:09,716 | INFO  | test accuracy: 0.92381
2025-03-20 17:08:09,717 | INFO  | [[93  7]
 [ 1  4]]
2025-03-20 17:08:09,719 | INFO  |               precision    recall  f1-score   support

      Benign       0.99      0.93      0.96       100
     Anomaly       0.36      0.80      0.50         5

    accuracy                           0.92       105
   macro avg       0.68      0.86      0.73       105
weighted avg       0.96      0.92      0.94       105
```

#### ProvNinja using [DP FiveDirections](../Data/FiveDirections-DP/) Dataset

```shell
$ python provgat.py -if 768 -hf 64 -lr 0.001 -e 50 -n 7 -bs 16 -dl 'C:\Users\prov-dp\GNN-based-IDS\Data\FiveDirections-DP\\' --device cpu -at 0.01
2025-03-20 17:08:20,201 | INFO  | using C:\Users\prov-dp\GNN-based-IDS\Data\FiveDirections-DP\ as input data directory
2025-03-20 17:08:20,201 | INFO  | 7 Layer GAT. Input Feature Size: 768. Hidden Layer Size(s): 64. Loss Rate: 0.001. Batch Size: 16
2025-03-20 17:08:20,201 | INFO  | Input Device: cpu
2025-03-20 17:08:20,201 | INFO  | Variable Prediction Threshold for Anomalous graphs have been enabled & set to 0.01
2025-03-20 17:08:20,201 | INFO  | Training on 50 epochs...
Done loading data from cached files.
Done loading data from cached files.
Done loading data from cached files.
2025-03-20 17:08:20,829 | INFO  | Length of dataset: 521
2025-03-20 17:08:20,849 | INFO  | Evaluating on Device: cpu
2025-03-20 17:08:20,849 | INFO  | # Parameters in model: 125129
2025-03-20 17:08:20,849 | INFO  | # Trainable parameters in model: 125129
2025-03-20 17:08:20,849 | INFO  | Stratified sampler enabled
2025-03-20 17:08:34,310 | INFO  | === test stats ===
2025-03-20 17:08:34,310 | INFO  | Number Correct: 95
2025-03-20 17:08:34,311 | INFO  | Number Graphs in test Data: 105
2025-03-20 17:08:34,311 | INFO  | test accuracy: 0.90476
2025-03-20 17:08:34,312 | INFO  | [[90 10]
 [ 0  5]]
2025-03-20 17:08:34,314 | INFO  |               precision    recall  f1-score   support

      Benign       1.00      0.90      0.95       100
     Anomaly       0.33      1.00      0.50         5

    accuracy                           0.90       105
   macro avg       0.67      0.95      0.72       105
weighted avg       0.97      0.90      0.93       105
```

### Training Output

#### ProvNinja using [Original FiveDirections](../Data/FiveDirections/) Dataset

```shell
$ python provgat.py -if 768 -hf 64 -lr 0.001 -e 50 -n 7 -bs 16 -dl 'C:\Users\prov-dp\GNN-based-IDS\Data\FiveDirections\\' --device cpu -at 0.01
2025-03-20 21:03:10,984 | INFO	| using C:\Users\prov-dp\GNN-based-IDS\Data\FiveDirections\ as input data directory
2025-03-20 21:03:10,984 | INFO	| 7 Layer GAT. Input Feature Size: 768. Hidden Layer Size(s): 64. Loss Rate: 0.001. Batch Size: 16
2025-03-20 21:03:10,984 | INFO	| Input Device: cuda
2025-03-20 21:03:10,984 | INFO	| Variable Prediction Threshold for Anomalous graphs have been enabled & set to 0.01
2025-03-20 21:03:10,984 | INFO	| Training on 50 epochs...
2025-03-20 21:03:11,469 | INFO	| Length of dataset: 521
2025-03-20 21:03:11,654 | INFO	| Training on Device: cuda
2025-03-20 21:03:11,655 | INFO	| Number benign in training dataset: 350
2025-03-20 21:03:11,656 | INFO	| Number anomaly in training dataset: 14
2025-03-20 21:03:11,656 | INFO	| Number benign in validation dataset: 50
2025-03-20 21:03:11,656 | INFO	| Number anomaly in validation dataset: 2
2025-03-20 21:03:11,656 | INFO	| Number benign in test dataset: 100
2025-03-20 21:03:11,656 | INFO	| Number anomaly in test dataset: 5
2025-03-20 21:03:11,657 | INFO	| # Parameters in model: 1501505
2025-03-20 21:03:11,657 | INFO	| # Trainable parameters in model: 1501505
2025-03-20 21:03:11,658 | INFO	| Stratified sampler enabled
2025-03-20 21:03:11,694 | INFO	| Computed weights for loss function: tensor([ 0.5210, 12.4048], device='cuda:0')
2025-03-20 21:03:24,074 | INFO	| Epoch 0: Training Accuracy: 0.30220, Average Training Loss: 0.47176, Validation Accuracy: 0.78846, Average Validation Loss: 0.12767
2025-03-20 21:03:36,099 | INFO	| Epoch 1: Training Accuracy: 0.48626, Average Training Loss: 0.22395, Validation Accuracy: 0.38462, Average Validation Loss: 0.07866
2025-03-20 21:03:48,579 | INFO	| Epoch 2: Training Accuracy: 0.42033, Average Training Loss: 0.27084, Validation Accuracy: 0.67308, Average Validation Loss: 0.10503
2025-03-20 21:04:00,745 | INFO	| Epoch 3: Training Accuracy: 0.52198, Average Training Loss: 0.14923, Validation Accuracy: 0.50000, Average Validation Loss: 0.07706
...
2025-03-20 21:12:18,871 | INFO	| Epoch 29: Training Accuracy: 0.96429, Average Training Loss: 0.00822, Validation Accuracy: 0.90385, Average Validation Loss: 0.60675
2025-03-20 21:13:28,261 | INFO	| === test stats ===
2025-03-20 21:13:28,261 | INFO	| Number Correct: 97
2025-03-20 21:13:28,262 | INFO	| Number Graphs in test Data: 105
2025-03-20 21:13:28,262 | INFO	| test accuracy: 0.92381
2025-03-20 21:13:28,262 | INFO	| [[93  7]
 [ 1  4]]
2025-03-20 21:13:28,266 | INFO	|               precision    recall  f1-score   support

      Benign       0.99      0.93      0.96       100
     Anamoly       0.36      0.80      0.50         5

    accuracy                           0.92       105
   macro avg       0.68      0.86      0.73       105
weighted avg       0.96      0.92      0.94       105
```

#### ProvNinja using [DP FiveDirections](../Data/FiveDirections-DP/) Dataset

```shell
$ python provgat.py -if 768 -hf 64 -lr 0.001 -e 50 -n 7 -bs 16 -dl 'C:\Users\prov-dp\GNN-based-IDS\Data\FiveDirections-DP\\' --device cpu -at 0.01
2025-03-20 16:51:02,915 | INFO	| using C:\Users\prov-dp\GNN-based-IDS\Data\FiveDirections-DP\ as input data directory
2025-03-20 16:51:02,915 | INFO	| 7 Layer GAT. Input Feature Size: 768. Hidden Layer Size(s): 64. Loss Rate: 0.001. Batch Size: 16
2025-03-20 16:51:02,915 | INFO	| Input Device: cuda
2025-03-20 16:51:02,915 | INFO	| Training on 50 epochs...
2025-03-20 16:51:46,931 | INFO	| Length of dataset: 521
2025-03-20 16:51:47,087 | INFO	| Training on Device: cuda
2025-03-20 16:51:47,088 | INFO	| Number benign in training dataset: 350
2025-03-20 16:51:47,088 | INFO	| Number anomaly in training dataset: 14
2025-03-20 16:51:47,088 | INFO	| Number benign in validation dataset: 50
2025-03-20 16:51:47,089 | INFO	| Number anomaly in validation dataset: 2
2025-03-20 16:51:47,089 | INFO	| Number benign in test dataset: 100
2025-03-20 16:51:47,089 | INFO	| Number anomaly in test dataset: 5
2025-03-20 16:51:47,089 | INFO	| # Parameters in model: 125129
2025-03-20 16:51:47,089 | INFO	| # Trainable parameters in model: 125129
2025-03-20 16:51:47,090 | INFO	| Stratified sampler enabled
2025-03-20 16:51:47,222 | INFO	| Computed weights for loss function: tensor([ 0.5210, 12.4048], device='cuda:0')
2025-03-20 16:51:54,621 | INFO	| Epoch 0: Training Accuracy: 0.92857, Average Training Loss: 0.89057, Validation Accuracy: 0.94231, Average Validation Loss: 0.31273
2025-03-20 16:52:01,567 | INFO	| Epoch 1: Training Accuracy: 0.96703, Average Training Loss: 0.15293, Validation Accuracy: 0.96154, Average Validation Loss: 0.08399
2025-03-20 16:52:06,667 | INFO	| Epoch 2: Training Accuracy: 0.98077, Average Training Loss: 0.15683, Validation Accuracy: 0.96154, Average Validation Loss: 0.07897
2025-03-20 16:52:11,703 | INFO	| Epoch 3: Training Accuracy: 0.98352, Average Training Loss: 0.06046, Validation Accuracy: 0.96154, Average Validation Loss: 0.07670
...
2025-03-20 16:54:35,757 | INFO	| Epoch 29: Training Accuracy: 0.99725, Average Training Loss: 0.00992, Validation Accuracy: 1.00000, Average Validation Loss: 0.02093
2025-03-20 16:54:37,732 | INFO	| === test stats ===
2025-03-20 16:54:37,732 | INFO	| Number Correct: 103
2025-03-20 16:54:37,732 | INFO	| Number Graphs in test Data: 105
2025-03-20 16:54:37,733 | INFO	| test accuracy: 0.98095
2025-03-20 16:54:37,733 | INFO	| [[99  1]
 [ 1  4]]
2025-03-20 16:54:37,736 | INFO	|               precision    recall  f1-score   support

      Benign       0.99      0.99      0.99       100
     Anamoly       0.80      0.80      0.80         5

    accuracy                           0.98       105
   macro avg       0.90      0.90      0.90       105
weighted avg       0.98      0.98      0.98       105
```

## File Structure

* [dataloaders/](dataloaders/)
  * [BaseDataloader.py](dataloaders/BaseDataloader.py)
      * This is a [DGL Dataset](https://docs.dgl.ai/en/0.6.x/guide/data.html) loader. It will take in the `.csv` files
        generated by [jsonToCsv.py] and parse them into a format consumable by DGL (ie a DGL heterograph). This class 
        serves as the base class for all other dataloaders.
      * More specifically, it will go through all the different relations (edges) in the dataset and put these relations
        into a DGL heterograph. From there, we import user defined node/edge attributes from the .csv files into the DGL
        heterograph.
      * The constructor function has documentation on how to instantiate a ProvDataset object.
      * You probably do not need to worry about this file.
  * [AnomalyBenignDataset.py](dataloaders/AnomalyBenignDataset.py)
    * Loads data for binary classification (anomaly/benign)
* [nn_types/](nn_types/)

  * [provgat.py](nn_types/provgat.py)
    * Houses the code for relational graph attention network  [Graph Attention Layer](https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/9_gat.html)
    * We use `HeteroRGATLayer` as a building block for RGAT. It is a single layer in the RGAT.
    * We use `KLayerHeteroRGAT` as a K number of layers layer RGAT
    * See provgat.py to run GAT network

* [provgat.py](provgat.py)
    * Houses the code for running ProvNinja.
    * In order to run the GNN:
        * For binary classification, your data must be set up in a way such that benign data is is split into train, test, and validation folders
          * Each train, test, and validation folder must have an anomaly/ and benign/ subfolder containing the respective graphs
        * Usage: `python3 provgat.py (binary|multi) (gcn|gat|mlp) -dl <dataset_dir_path> -if <input feature size> -hf <hidden feature size> -lr <loss rate> -e <# epochs> -n <# layers> -bs <batch_size> --device <device> [-bdst <percentage to downsample>] [-at <anomaly threshold>] `          
          * You can specify the device you want to run the models on with `--device <device>`. By default, the model will use GPU if it's available. (`<device>` parameter can be cpu, cuda, cuda:1, etc..)
          * You can add percentage for benign downsampling for training data `-bdst <percentage to downsample [0.0-1.0]>` flag
          * You can add a prediction threshold for classification of anomaly graphs `-at <anomaly threshold [0.0-1.0]>` flag for binary classification

* [gnnUtils.py](gnnUtils.py)
  * Houses utility functions for provgat.py.

* [samplers.py](samplers.py)
  * Contains Samplers & Batch Samplers for the DGL's GraphDataLoader class.

## Outputs

* [run](runs/) - Stats for each run is stored in the `runs` folder and can be viewed using tensorboard by doing `tensorboard --logdir=runs`
* [models](models/) - Model for each run is stored in the `models` folder
* [logs](logs/) - Logs for each run is stored in the `logs` folder

## References 

[1] K. Mukherjee, et al., “_Evading Provenance-Based ML Detectors with Adversarial System Actions_,” in
USENIX Security Symposium (SEC), 2023. <br>