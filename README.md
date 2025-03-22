# ProvDP: Differential Privacy for System Provenance Dataset

Reproducibility artifacts for the paper ProvDP: Differential Privacy for System Provenance Dataset.

## Environment Setup
1. Create a new virtual environment: `pip -m venv venv` or `conda create --name provdp python=3.11`
2. Activate virtual environment: `source venv/bin/activate` or `conda activate provdp`
3. Install dependencies via `pip install -r requirements.txt`

Note: Scripts are ran using `python -m` to avoid having to manipulate the `PYTHONPATH` environment variable.

## Directory Structure

| Directory | Description|
| -------|-----------|
| `DP`        | Directory containing the code and data to execute the DP algorithm. |
| `GNN-based-IDS` | Directory containing the code and data files for GNN-based IDS execution. |

## ProvDP

- To run the ProvDP pipeline, run the following command. More information on the arguments can be found in the
[`parse_args()`](DP/ProvDP/src/cli/perturb.py#70) function in [`perturb.py`](DP/ProvDP/src/cli/perturb.py).

```shell
$ python -m src.cli.perturb -i ../FiveDirections/benign-500 -o ../FiveDirections/benign-500-dp --epsilon 1 --alpha 0.7 --beta 0.1 --gamma 0.1 --eta 0.1
```

- **Input**: 500 benign provenance graphs from FiveDirections dataset: [FiveDirections/benign-500](DP/FiveDirections/benign-500/) directory.
- **Output**: 500 benign DP provenance graphs from FiveDirections dataset: [FiveDirections/benign-500-dp_N=500_epsilon=1.0_delta=0.5_alpha=0.7_beta=0.1_gamma=0.1_eta=0.1_k=0.1](DP/FiveDirections/benign-500-dp_N=500_epsilon=1.0_delta=0.5_alpha=0.7_beta=0.1_gamma=0.1_eta=0.1_k=0.1/) directory.


### ProvNinja [[1]](#references)

* Driver script for [Prov-GAT](GNN-based-IDS/ProvNinja/provgat.py), which is a GAT based IDS that detects anomalous graphs.
* Separated the benign and malicious graphs from the FiveDirections DARPA Transparent Computing Dataset and stored them in  [FiveDirections](GNN-based-IDS/Data/FiveDirections/) directory.
* ProvDP processed FiveDirections benign and anomalous graphs are available in [FiveDirections-DP](GNN-based-IDS/Data/FiveDirections-DP/) directory.

```shell
$ python provgat.py -if 768 -hf 64 -lr 0.001 -e 50 -n 7 -bs 16 -dl 'C:\Users\prov-dp\GNN-based-IDS\Data\FiveDirections\\' --device cpu -at 0.01
```

```shell
$ python provgat.py -if 768 -hf 64 -lr 0.001 -e 50 -n 7 -bs 16 -dl 'C:\Users\prov-dp\GNN-based-IDS\Data\FiveDirections-DP\\' --device cpu -at 0.01
```

## Citing Us

```
@inproceedings{mukherjee2025provDP,
	title        = {ProvDP: Differential Privacy for System Provenance Dataset},
	author       = {Kunal Mukherjee and Jonathan Yu and Partha De and Dinil Mon Divakaran},
	year         = 2025,
	booktitle    = {23rd Conference on Applied Cryptography and Network Security (ACNS)},
	series       = {ACNS '25}
}
```

## References 

[1] K. Mukherjee, et al., “_Evading Provenance-Based ML Detectors with Adversarial System Actions_,” in
USENIX Security Symposium (SEC), 2023. <br>