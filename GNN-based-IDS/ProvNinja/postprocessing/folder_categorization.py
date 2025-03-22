import math
import os
import pickle
import shutil
from argparse import ArgumentParser

import numpy as np
import pathlib

def categorized_binary_dataset(parsed_args):
    root_graph_dir = pathlib.Path(parsed_args.dir).resolve()
    benign_folder_name = parsed_args.benign_folder_name
    anomaly_folder_name = parsed_args.anomaly_folder_name

    train_split = parsed_args.train_split
    val_split = parsed_args.validation_split
    test_split = parsed_args.test_split

    assert train_split + val_split + test_split == 1.0, 'Train, val, and test split must equal 1'

    train_dir = root_graph_dir / 'train'
    val_dir = root_graph_dir / 'validation'
    test_dir = root_graph_dir / 'test'

    for dir in [train_dir, val_dir, test_dir]:
        dir.mkdir(exist_ok=True)

    train_benign_dir = train_dir / 'benign'
    val_benign_dir = val_dir / 'benign'
    test_benign_dir = test_dir / 'benign'

    train_anomaly_dir = train_dir / 'anomaly'
    val_anomaly_dir = val_dir / 'anomaly'
    test_anomaly_dir = test_dir / 'anomaly'

    for dir in [
            train_benign_dir, val_benign_dir, test_benign_dir,
            train_anomaly_dir, val_anomaly_dir, test_anomaly_dir
    ]:
        dir.mkdir(exist_ok=True)

    benign_folders = list((root_graph_dir / benign_folder_name).glob("nd-*")) + \
        list((root_graph_dir / benign_folder_name).glob("nd_*"))
    anomaly_folders = [dir for dir in (root_graph_dir / anomaly_folder_name).iterdir() if dir.is_dir()]

    benign_folder_count = len(benign_folders)
    anomaly_folder_count = len(anomaly_folders)
    
    print(f"# of benign folder found: {benign_folder_count}")
    print(f"# of anomaly folder found: {anomaly_folder_count}")

    train_max_counts = {
        'benign': math.floor(train_split * benign_folder_count),
        'anomaly': math.floor(train_split * anomaly_folder_count),
    }

    val_max_counts = {
        'benign': math.floor(val_split * benign_folder_count),
        'anomaly': math.floor(val_split * anomaly_folder_count),
    }

    test_max_counts = {
        'benign': math.floor(test_split * benign_folder_count),
        'anomaly': math.floor(test_split * anomaly_folder_count),
    }

    train_counts = {t: 0 for t in ['benign', 'anomaly']}
    val_counts = {t: 0 for t in ['benign', 'anomaly']}
    test_counts = {t: 0 for t in ['benign', 'anomaly']}

    # move all benign folders
    benign_random_permut = np.random.RandomState(seed=0).permutation(benign_folders)
    for graph_dir in benign_random_permut:
        if train_counts['benign'] < train_max_counts['benign']:
            shutil.move(
                os.path.join(root_graph_dir, benign_folder_name, graph_dir),
                train_benign_dir)
            train_counts['benign'] += 1
        elif val_counts['benign'] < val_max_counts['benign']:
            shutil.move(
                os.path.join(root_graph_dir, benign_folder_name, graph_dir),
                val_benign_dir)
            val_counts['benign'] += 1
        elif test_counts['benign'] < test_max_counts['benign']:
            shutil.move(
                os.path.join(root_graph_dir, benign_folder_name, graph_dir),
                test_benign_dir)
            test_counts['benign'] += 1
        else:
            print(f'Skipped benign graph {graph_dir}')

    # move all anomaly folders
    anomaly_random_permut = np.random.RandomState(seed=0).permutation(anomaly_folders)  # shuffle first
    for graph_dir in anomaly_random_permut:
        if train_counts['anomaly'] < train_max_counts['anomaly']:
            shutil.move(
                os.path.join(root_graph_dir, anomaly_folder_name, graph_dir),
                train_anomaly_dir)
            train_counts['anomaly'] += 1
        elif val_counts['anomaly'] < val_max_counts['anomaly']:
            shutil.move(
                os.path.join(root_graph_dir, anomaly_folder_name, graph_dir),
                val_anomaly_dir)
            val_counts['anomaly'] += 1
        elif test_counts['anomaly'] < test_max_counts['anomaly']:
            shutil.move(
                os.path.join(root_graph_dir, anomaly_folder_name, graph_dir),
                test_anomaly_dir)
            test_counts['anomaly'] += 1
        else:
            print(f'Skipped anomaly graph {graph_dir}')

    print('Done.')

if __name__ == '__main__':
    binary_parser = ArgumentParser(prog='Folder Categorization Script')

    binary_parser.add_argument('train_split',
                        type=float,
                        help='Train percent split')
    binary_parser.add_argument('validation_split',
                        type=float,
                        help='Validation percent split')
    binary_parser.add_argument('test_split',
                        type=float,
                        help='Test percent split')
    binary_parser.add_argument('-d',
                        '--dir',
                        type=str,
                        required=True,
                        help='Input Directory Containing Graphs')
    binary_parser.add_argument('-bf',
                               '--benign_folder_name',
                               type=str,
                               required=True,
                               help='Name of folder containing '
                               'benign graphs')
    binary_parser.add_argument('-af',
                               '--anomaly_folder_name',
                               type=str,
                               required=True,
                               help='Name of folder containing '
                               'anomaly graphs')
    binary_parser.set_defaults(func=categorized_binary_dataset)

    parsed_args = binary_parser.parse_args()
    parsed_args.func(parsed_args)
