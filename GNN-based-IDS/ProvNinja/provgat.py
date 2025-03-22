import torch
import logging
import argparse
import warnings

from __init__ import device
from datetime import datetime

from nn_types.gat import KLayerHeteroRGAT
from torch.utils.tensorboard import SummaryWriter
from gnnUtils import *


log = logging.getLogger(__name__)
date_time = datetime.now().strftime('%b%d_%H-%M-%S')

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

GNN_TYPES = ['gat']

class BinaryHeteroClassifier(nn.Module):

    def __init__(
        self,
        gnn_type,
        num_layers,
        input_feature_size,
        hidden_dimension_size,
        graph_relation_list,
        graph_node_types
    ):
        super(BinaryHeteroClassifier, self).__init__()
        assert gnn_type in GNN_TYPES, f'Supported GNN types are {GNN_TYPES}'

        self.gnn = KLayerHeteroRGAT(
            num_layers,
            input_feature_size,
            hidden_dimension_size,
            hidden_dimension_size,
            graph_relation_list,
            graph_node_types,
            structural=False,
        )

        self.classifier = nn.Linear(hidden_dimension_size, 1)

    def forward(self, graph, feature_set=None):
        
        x = self.gnn(graph, feature_set)
        
        # classify graph as benign/malicious
        with graph.local_scope():
            graph.ndata['x'] = x
            hg = 0
            for ntype in graph.ntypes:
                if graph.num_nodes(ntype):
                    hg = hg + dgl.mean_nodes(graph, 'x', ntype=ntype)

            return th.sigmoid(self.classifier(hg))


class BinaryHeteroClassifierAdapter(nn.Module):

    def __init__(self, model, threshold=0.5):
        super(BinaryHeteroClassifierAdapter, self).__init__()
        self.model = model
        self.threshold = threshold

    def forward(self, graph, features=None):
        
        prob = self.model(graph, features)
        prob = torch.log(prob / (1 - prob + 1e-20))
        
        return torch.tensor([[1 - prob, prob]])


def add_gnn_args(parser):

    parser.add_argument(
        '-dl',
        '--location',
        type=str,
        help='Path to the dataset.',
        required=True
    )
    parser.add_argument(
        '-if',
        '--input_feature',
        type=int,
        help='Input feature size.',
        required=True
    )
    parser.add_argument(
        '-hf',
        '--hidden_feature',
        type=int,
        help='Hidden feature size.',
        required=True
    )
    parser.add_argument(
        '-lr',
        '--loss_rate',
        type=float,
        default=0.01,
        help='Loss rate.',
        required=True
    )
    parser.add_argument(
        '-e',
        '--epochs',
        type=int,
        help='Number of epochs.',
        required=True
    )
    parser.add_argument(
        '-n',
        '--layers',
        type=int,
        default=2,
        help='Number of layers in the GNN.',
        required=True,
    )
    parser.add_argument(
        '-bs',
        '--batch_size',
        type=int,
        help='How many graphs we want each minibatch to have',
        required=True,
    )
    parser.add_argument(
        '--force_reload',
        help='Reload the dataset without using cached data',
        action='store_true',
        default=False,
        required=False,
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help=
        'Device to run. Options are cpu, cuda, cuda:N. N is the index of the cuda device which can be fetched using nvidia-smi command.',
        required=True,
    )
    parser.add_argument(
        '-bdst',
        '--benign_downsampling_training',
        type=float,
        default=0.0,
        help='A percentage for benign downsampling for training [0.0-1.0]',
        required=False,
    )
    parser.add_argument(
        '-at',
        '--anomaly_threshold',
        type=float,
        default=0.0,
        help='Threshold for classification of anomalous graphs [0.0-1.0]',
        required=False,
    )


def get_log_path(args):
    neural_network_type = GNN_TYPES[0]
    dataset_dir_path = args.location
    input_feature_size = args.input_feature
    hidden_feature_size = args.hidden_feature
    loss_rate = args.loss_rate
    epochs = args.epochs
    num_layers = args.layers
    batch_size = args.batch_size
    
    dataset_name = os.path.basename(os.path.normpath(dataset_dir_path))

    log_name = (
        f'{neural_network_type}_{input_feature_size}_{hidden_feature_size}_{loss_rate}'
        f'_{epochs}_{num_layers}_{batch_size}_{dataset_name}')

    return log_name


def main():
    parser = argparse.ArgumentParser(
        description='Runner script for Heterogeneous GNNs.')

    add_gnn_args(parser)

    parsed_arguments = parser.parse_args()

    neural_network_type = GNN_TYPES[0]
    dataset_dir_path = parsed_arguments.location
    input_feature_size = parsed_arguments.input_feature
    hidden_feature_size = parsed_arguments.hidden_feature
    loss_rate = parsed_arguments.loss_rate
    epochs = parsed_arguments.epochs
    num_layers = parsed_arguments.layers
    batch_size = parsed_arguments.batch_size
    force_reload = parsed_arguments.force_reload
    benign_downsampling_training = parsed_arguments.benign_downsampling_training
    anomaly_threshold = parsed_arguments.anomaly_threshold

    log_name = get_log_path(parsed_arguments)

    global device

    if parsed_arguments.device is not None:
        device = th.device(parsed_arguments.device)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s\t| %(message)s',
        handlers=[
            logging.FileHandler(
                os.path.join(os.getcwd(), 'logs', log_name + '.log'),
                mode='w+',
                encoding='utf-8',
            ),
            logging.StreamHandler(),
        ],
    )

    outputInputArguments(
        log,
        dataset_dir_path,
        epochs,
        num_layers,
        input_feature_size,
        hidden_feature_size,
        loss_rate,
        batch_size,
        device,
        benign_downsampling_training,
        anomaly_threshold,
    )

    ATTR_DICT = {
    'ProcessNode': 'EXE_NAME',
    'FileNode': 'FILENAME_SET',
    'SocketChannelNode': 'REMOTE_INET_ADDR',
    'VirtualNode': []
    }

    node_attributes = {
        'ProcessNode': [ATTR_DICT['ProcessNode']],
        'FileNode': [ATTR_DICT['FileNode']],
        'SocketChannelNode': [ATTR_DICT['SocketChannelNode']],
        "VirtualNode": ["TYPE"],
    }

    relation_attributes = {
        ('ProcessNode', 'PROC_END', 'ProcessNode'): [],
        ('ProcessNode', 'FILE_EXEC', 'ProcessNode'):[],
        ('ProcessNode', 'PROC_CREATE', 'ProcessNode'): [],
        ('ProcessNode', 'READ', 'FileNode'): [],
        ('ProcessNode', 'WRITE', 'FileNode'): [],
        ('ProcessNode', 'FILE_EXEC', 'FileNode'): [],
        ('ProcessNode', 'WRITE', 'SocketChannelNode'): [],
        ('ProcessNode', 'READ', 'SocketChannelNode'): [],
        ('ProcessNode', 'FILE_EXEC', 'SocketChannelNode'): [],
        ('ProcessNode', 'IP_CONNECTION_EDGE', 'ProcessNode'): [],
        ('ProcessNode', 'IP_CONNECTION_EDGE', 'FileNode'): [],
        ("ProcessNode", "FILE_EXEC", "VirtualNode"): [],
    }

    def feature_aggregation_function(graph):
        return {
            'ProcessNode':
                torch.cat(
                    [graph.nodes['ProcessNode'].data[ATTR_DICT['ProcessNode']].to(device)],
                    dim=1,
                ) if graph.num_nodes('ProcessNode') else torch.empty((0, 768), device=device),
            'FileNode': 
                torch.cat(
                    [graph.nodes['FileNode'].data[ATTR_DICT['FileNode']].to(device)],
                    dim=1,
                ) if graph.num_nodes('FileNode') else torch.empty((0, 768), device=device),
            'SocketChannelNode':
                torch.cat(
                    [graph.nodes['SocketChannelNode'].data[ATTR_DICT['SocketChannelNode']].to(device)],
                    dim=1,
                ) if graph.num_nodes('SocketChannelNode') else torch.empty((0, 768), device=device),
            'VirtualNode':
                torch.zeros(graph.num_nodes('VirtualNode'), 768),
            }

    agg_func = feature_aggregation_function

    for relation_attribute in list(relation_attributes.keys()):
        flipped_relation = relation_attribute[::-1]
        if flipped_relation not in relation_attributes:
            relation_attributes[flipped_relation] = relation_attributes[
                relation_attribute]

    train_dataset, val_dataset, test_dataset = get_binary_train_val_test_datasets(
        dataset_dir_path,
        'benign',
        'anomaly',
        node_attributes,
        relation_attributes,
        force_reload=force_reload,
        verbose=True,
    )

    dataset_length = len(train_dataset) + len(val_dataset) + len(test_dataset)
    
    log.info(f'Length of dataset: {dataset_length}')

    writer = SummaryWriter(os.path.join(os.getcwd(), 'runs', log_name))

    model = BinaryHeteroClassifier(
        neural_network_type,
        num_layers,
        input_feature_size,
        hidden_feature_size,
        list(relation_attributes.keys()),
        list(node_attributes.keys())
    )

    # training loop
    # train_binary_graph_classification(
    #     model,
    #     writer,
    #     train_dataset,
    #     val_dataset,
    #     test_dataset,
    #     loss_rate,
    #     epochs,
    #     logger=log,
    #     feature_aggregation_func=agg_func,
    #     batch_size=batch_size,
    #     device=device,
    #     benign_downsampling_training=benign_downsampling_training,
    #     anomaly_threshold=anomaly_threshold,
    # )

    # th.save(model, os.path.join(os.getcwd(), 'models', log_name + '.pt'))
    
    # evaluation loop    
    model = th.load(os.path.join(os.getcwd(), "models", log_name + ".pt"), map_location=device)
    model.eval()

    evaluate_binary_graph_classification(
        model,
        writer,
        test_dataset,
        log,
        batch_size,
        feature_aggregation_func=agg_func,
        device=device,
        benign_downsampling_training=benign_downsampling_training,
        anomaly_threshold=anomaly_threshold,
    )
    
    writer.close()


if __name__ == '__main__':
    main()
