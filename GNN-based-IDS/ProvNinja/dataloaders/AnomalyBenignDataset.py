from .BaseDataloader import ProvDataset
import os
import torch as th


class AnomalyBenignDataset(ProvDataset):
    _ProvDataset__cache_file_name = '_anomaly_benign_prov_dataset.bin'

    def __init__(self,
                 input_dir,
                 benign_folder_name,
                 anomaly_folder_name,
                 node_attributes_map,
                 relation_attributes_map,
                 bidirection=False,
                 force_reload=False,
                 verbose=False):
        self.benign_folder = os.path.join(input_dir, benign_folder_name)
        self.anomaly_folder = os.path.join(input_dir, anomaly_folder_name)

        super(AnomalyBenignDataset,
              self).__init__(name='Anomaly Benign Provenance Graph',
                             input_dir=input_dir,
                             node_attributes_map=node_attributes_map,
                             relation_attributes_map=relation_attributes_map,
                             bidirection=bidirection,
                             force_reload=force_reload,
                             verbose=verbose)

    def process(self):
        benign_subfolders = [
            f.path for f in os.scandir(self.benign_folder) if f.is_dir()
        ]
        anomaly_subfolders = [
            f.path for f in os.scandir(self.anomaly_folder) if f.is_dir()
        ]

        for benign_file in benign_subfolders:
            self._ProvDataset__processGraph(benign_file, 0)

        for anomaly_file in anomaly_subfolders:
            self._ProvDataset__processGraph(anomaly_file, 1)

        self.labels = th.tensor(
            self.labels,
            dtype=th.float)

        num_benign_processed = sum(label == 0 for label in self.labels)
        num_anomaly_processed = sum(label == 1 for label in self.labels)

        print(
            f'Processed {num_benign_processed}/{len(benign_subfolders)} benign graphs ({float(num_benign_processed) / len(benign_subfolders) * 100:.2f}%)'
        )
        print(
            f'Processed {num_anomaly_processed}/{len(anomaly_subfolders)} anomaly graphs ({float(num_anomaly_processed) / len(anomaly_subfolders) * 100:.2f}%)'
        )