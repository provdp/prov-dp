# ProvDP: Differential Privacy for System Provenance Dataset

Reproducibility artifacts for the paper ProvDP: Differential Privacy for System Provenance Dataset. This directory contains the library and source code related to applying differentially private perturbation to 
provenance graphs.

## Project Structure

### src.algorithm
This package contains the core logic of the project.
- [`graph_processor.py`](src/algorithm/graph_processor.py) is responsible for loading files, pruning trees, and then reattaching the pruned trees according to differential privacy. The core of the differential privacy pipeline is called in [`perturb_graphs()`](DP/ProvDP/src/cli/perturb.py) function.

### src.algorithm.wrappers
- [`edge.py`](src/algorithm/wrappers/edge.py), [`node.py`](src/algorithm/wrappers/node.py) contains simple wrappers for [`raw_edge.py`](src/graphson/raw_edge.py) and [`raw_node.py`](src/graphson/raw_node.py), respectively.
- [`tree.py`](src/algorithm/wrappers/tree.py) contains the graph-to-tree conversion logic, as well as functions to help prune and re-attach subtrees.

### src.cli
This package contains CLI wrappers to interact with the `algorithm` package.
- [`perturb.py`](src/cli/perturb.py) - Run the graph processing pipeline
- [`tree_to_graph.py`](src/cli/tree_to_graph.py) - Convert provenance trees into graphs

### src.graphson
This package contains simple scripts used to serialize graphs to and from json.

## ProvDP

## Running ProvDP
To run the ProvDP pipeline, run the following. More information on the arguments can be found in the
`parse_args()` function in [`perturb.py`](src/cli/perturb.py).

```shell
python -m src.cli.perturb -i ../FiveDirections/benign-500 -o ../FiveDirections/benign-500-dp --epsilon 1 --alpha 0.7 --beta 0.1 --gamma 0.1 --eta 0.1
```

### Output
```shell
ProvDP: started run with INPUT dir: ..\FiveDirections\benign-500 and OUTPUT dir: benign-500-dp_N=500_epsilon=1.0_delta=0.5_alpha=0.7_beta=0.1_gamma=0.1_eta=0.1_k=0.1
ARGUMENTS: epsilon=1.0, delta=0.5, alpha=0.7, beta=0.1, gamma=0.1 eta=0.1
ARGUMENTS: epsilon=1.0, delta=0.5 (e_1=0.5, e_2=0.5)
Preprocessing graphs: 100%|████████████████████████████████████████████████████████████████████| 500/500 [00:10<00:00, 47.38it/s]
(1) Pruning graphs: 100%|██████████████████████████████████████████████████████████████████████| 500/500 [00:20<00:00, 24.45it/s]
  Wrote 500 graphs to ..\FiveDirections\benign-500-dp_N=500_epsilon=1.0_delta=0.5_alpha=0.7_beta=0.1_gamma=0.1_eta=0.1_k=0.1\pruned_graphs.pkl
Bucket sizes:
  size 1: 101280 subtrees
  size 2: 6419 subtrees
  size 3: 1929 subtrees
  size 4: 459 subtrees
  size 5: 204 subtrees
  size 6: 77 subtrees
  size 7: 25 subtrees
  size 8: 9 subtrees
  size 9: 5 subtrees
  size 10: 10 subtrees
  size 11: 2 subtrees
(2) Re-attaching subgraphs: 100%|███████████████████████████████████████████████████████████████| 500/500 [26:17<00:00,  3.15s/it]
Saving graphs: 100%|██████████████████████████████████████████████████████| 500/500 [00:34<00:00, 14.30it/s]
```