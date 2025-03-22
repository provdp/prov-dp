import logging
import os
import dgl
import warnings

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plot

from datetime import datetime
from collections.abc import Mapping, Sequence

from dgl.dataloading import GraphDataLoader

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from samplers import StratifiedBatchSampler

DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_WORKERS = 1

# suppress PyTorch cuda warnings
warnings.filterwarnings(
    'ignore',
    message=
    'User provided device_type of \'cuda\', but CUDA is not available. Disabling'
)

from dataloaders.AnomalyBenignDataset import AnomalyBenignDataset


def outputInputArguments(logger,
                         dataset_dir_path,
                         number_epochs,
                         number_of_layers,
                         input_feature_size,
                         hidden_feature_size,
                         loss_rate,
                         batch_size,                         
                         device,
                         benign_downsampling_training,
                         anomaly_threshold):
    logger.info(
        f'using {dataset_dir_path} as input data directory')
    logger.info(
        f'{number_of_layers} Layer GAT. Input Feature Size: {input_feature_size}. Hidden Layer Size(s):'
        f' {hidden_feature_size}. Loss Rate: {loss_rate}. Batch Size: {batch_size}')
    logger.info(f'Input Device: {device}')

    if benign_downsampling_training:
        logger.info(f'Benign Down Sampling: {benign_downsampling_training}')
    if anomaly_threshold:
        logger.info(
            f'Variable Prediction Threshold for Anomalous graphs have been enabled & set to {anomaly_threshold}'
        )

    logger.info(f'Training on {number_epochs} epochs...')


def outputBinaryCounts(train_dataset, validation_dataset, test_dataset,
                       logger):
    logger.info(
        f'Number benign in training dataset: {sum(label.item() == 0 for _, label in train_dataset)}'
    )
    logger.info(
        f'Number anomaly in training dataset: {sum(label.item() == 1 for _, label in train_dataset)}'
    )

    logger.info(
        f'Number benign in validation dataset: {sum(label.item() == 0 for _, label in validation_dataset)}'
    )
    logger.info(
        f'Number anomaly in validation dataset: {sum(label.item() == 1 for _, label in validation_dataset)}'
    )

    logger.info(
        f'Number benign in test dataset: {sum(label.item() == 0 for _, label in test_dataset)}'
    )
    logger.info(
        f'Number anomaly in test dataset: {sum(label.item() == 1 for _, label in test_dataset)}'
    )


def outputEpochStats(epoch, train_acc, train_loss, val_acc, val_loss, logger):
    logger.info(f'Epoch {epoch}: '
                f'Training Accuracy: {train_acc:.5f}, '
                f'Average Training Loss: {train_loss:.5f}, '
                f'Validation Accuracy: {val_acc:.5f}, '
                f'Average Validation Loss: {val_loss:.5f}')


def outputStats(data_type,
                num_correct,
                len_data,
                pred_y_vals,
                true_y_vals,
                dataset_labels,
                logger,
                incorrect_graphs=None):
    logger.info(f'=== {data_type} stats ===')
    logger.info(f'Number Correct: {num_correct}')

    if incorrect_graphs:
        logger.info(f'Incorrect graph names: {incorrect_graphs}')

    logger.info(f'Number Graphs in {data_type} Data: {len_data}')
    logger.info(f'{data_type} accuracy: {(float(num_correct) / len_data):.5f}')

    logger.info(confusion_matrix(true_y_vals, pred_y_vals))

    if data_type == 'test':
        logger.info(
            classification_report(true_y_vals,
                                  pred_y_vals,
                                  target_names=dataset_labels))


def outputPerEpochStatsToSummaryWriter(summaryWriter, epoch, train_acc,
                                       train_loss, val_acc, val_loss):
    summaryWriter.add_scalar('Training Accuracy', train_acc, epoch)
    summaryWriter.add_scalar('Average Training Loss', train_loss, epoch)
    summaryWriter.add_scalar('Validation Accuracy', val_acc, epoch)
    summaryWriter.add_scalar('Average Validation Loss', val_loss, epoch)


def outputTestStatsToSummaryWriter(summaryWriter, test_num_correct, len_test_data, len_dataset):
    summaryWriter.add_scalar('Test Accuracy', float(test_num_correct) / len_test_data)
    summaryWriter.add_scalar('Test Number Correct', test_num_correct)
    summaryWriter.add_scalar('Length of Test Dataset', len_test_data)
    summaryWriter.add_scalar('Length of Entire Dataset', len_dataset)


def get_binary_train_val_test_datasets(dataset_dir_path,
                                       benign_folder_name,
                                       anamoly_folder_name,
                                       node_attributes_map,
                                       relation_attributes_map,
                                       force_reload=False,
                                       verbose=True):
    train_dataset = AnomalyBenignDataset(os.path.join(dataset_dir_path,
                                                      'train'),
                                         benign_folder_name,
                                         anamoly_folder_name,
                                         node_attributes_map,
                                         relation_attributes_map,
                                         bidirection=True,
                                         force_reload=force_reload,
                                         verbose=verbose)
    val_dataset = AnomalyBenignDataset(os.path.join(dataset_dir_path,
                                                    'validation'),
                                       benign_folder_name,
                                       anamoly_folder_name,
                                       node_attributes_map,
                                       relation_attributes_map,
                                       bidirection=True,
                                       force_reload=force_reload,
                                       verbose=verbose)
    test_dataset = AnomalyBenignDataset(os.path.join(dataset_dir_path, 'test'),
                                        benign_folder_name,
                                        anamoly_folder_name,
                                        node_attributes_map,
                                        relation_attributes_map,
                                        bidirection=True,
                                        force_reload=force_reload,
                                        verbose=verbose)

    return train_dataset, val_dataset, test_dataset


def cal_acc_and_loss(num_correct_predictions, len_of_data, loss_history):
    acc = float(num_correct_predictions) / len_of_data
    loss = float(sum(loss_history)) / len(loss_history)
    return acc, loss


def getTotalParams(model):
    return sum(p.numel() for p in model.parameters())


def getTotalTrainableParams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def saveProbabilityDistibution(data):
    bins = np.arange(0, 1, 0.01)  # fixed bin size

    plot.xlim([min(data) - 0.01, max(data) + 0.01])

    plot.hist(data, bins=bins, alpha=0.5)
    plot.title('Probability distribution of anomaly (fixed bin size)')
    plot.xlabel('Probability of anomaly (bin size = 0.01)')
    plot.ylabel('count')


def train_binary_graph_classification(model,
                                      summaryWriter,
                                      train_dataset,
                                      validation_dataset,
                                      test_dataset,
                                      loss_rate,
                                      epochs,
                                      logger,
                                      feature_aggregation_func=None,                                      
                                      batch_size=DEFAULT_BATCH_SIZE,
                                      device=None,
                                      benign_downsampling_training=None,
                                      anomaly_threshold=None):
    assert isinstance(
        train_dataset, AnomalyBenignDataset
    ), 'Train Dataset is not suited for Binary Classification'
    assert isinstance(validation_dataset, AnomalyBenignDataset
                      ), 'Val Dataset is not suited for Binary Classification'
    assert isinstance(test_dataset, AnomalyBenignDataset
                      ), 'Test Dataset is not suited for Binary Classification'

    if device is None:
        device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    logger.info(f'Training on Device: {device}')
    outputBinaryCounts(train_dataset, validation_dataset, test_dataset, logger)

    logger.info(f'# Parameters in model: {getTotalParams(model)}')
    logger.info(
        f'# Trainable parameters in model: {getTotalTrainableParams(model)}')

    model.to(device)

    
    logger.info(f'Stratified sampler enabled')

    train_batch_sampler = StratifiedBatchSampler(train_dataset.labels,
                                                    batch_size=batch_size,
                                                    shuffle=True)
    val_batch_sampler = StratifiedBatchSampler(validation_dataset.labels,
                                                batch_size=batch_size,
                                                shuffle=False)
    test_batch_sampler = StratifiedBatchSampler(test_dataset.labels,
                                                batch_size=batch_size,
                                                shuffle=False)

    train_dataloader = GraphDataLoader(train_dataset,
                                        collate_fn=collate_func,
                                        num_workers=DEFAULT_NUM_WORKERS,
                                        batch_sampler=train_batch_sampler)
    val_dataloader = GraphDataLoader(validation_dataset,
                                        collate_fn=collate_func,
                                        num_workers=DEFAULT_NUM_WORKERS,
                                        batch_sampler=val_batch_sampler)
    test_dataloader = GraphDataLoader(test_dataset,
                                        collate_fn=collate_func,
                                        num_workers=DEFAULT_NUM_WORKERS,
                                        batch_sampler=test_batch_sampler)

    # compute class weights for sample weighting (
    # https://medium.com/gumgum-tech/handling-class-imbalance-by-introducing-sample-weighting-in-the-loss-function-3bdebd8203b4)
    dataset_labels_numpy = th.cat(
        (train_dataset.labels, validation_dataset.labels, test_dataset.labels),
        0).numpy()

    assert 0 in dataset_labels_numpy, 'Dataset must contain at least one 0 label'
    assert 1 in dataset_labels_numpy, 'Dataset must contain at least one 1 label'

    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=[0, 1],
                                         y=dataset_labels_numpy)

    class_weights = th.tensor(class_weights, dtype=th.float, device=device)
    logger.info(f'Computed weights for loss function: {class_weights}')

    opt = th.optim.Adam(model.parameters(), lr=loss_rate)

    training_accuracy_hist = []
    validation_accuracy_hist = []

    avg_training_loss_hist = []
    avg_validation_loss_hist = []

    for epoch in range(epochs):
        train_loss_history = []
        training_num_correct = 0
        train_y_pred = []
        train_y_true = []
        train_incorrect_graphs = [] if benign_downsampling_training else None # if remove_stratified_sampler is turned on
        # lets print incorrect graph names since
        # batch size would = 1 anyways
        for batched_graphs, batched_labels in train_dataloader:
            
            if benign_downsampling_training:
                
                unbatched_graphs = dgl.unbatch(batched_graphs)
                benign_unbatched_graphs = np.array([
                    graph
                    for graph, label in zip(unbatched_graphs, batched_labels)
                    if label == 0
                ])
                anomaly_unbatched_graphs = np.array([
                    graph
                    for graph, label in zip(unbatched_graphs, batched_labels)
                    if label == 1
                ])

                chosen_benign_unbatch_graphs_indx = th.randperm(
                    len(benign_unbatched_graphs))[:int(
                        len(benign_unbatched_graphs) *
                        benign_downsampling_training)]
                chosen_benign_unbatch_graphs = benign_unbatched_graphs[
                    chosen_benign_unbatch_graphs_indx.numpy().astype(int)]

                batched_graphs = dgl.batch(
                    list(
                        np.concatenate((chosen_benign_unbatch_graphs,
                                        anomaly_unbatched_graphs), axis=None)))
                batched_labels = th.tensor([
                    0.0 for _ in range(
                        int(
                            len(benign_unbatched_graphs) *
                            benign_downsampling_training))
                ] + [1.0 for _ in range(len(anomaly_unbatched_graphs))])

                # logger.info(
                #     f'Benign Batched Graphs: {len([label for label in batched_labels if label == 0])} Anomaly Batched Graphs: {len([label for label in batched_labels if label == 1])}'
                # )

            batched_graphs = batched_graphs.to(device)
            batched_labels = th.reshape(batched_labels, (batched_labels.shape[0], 1)).to(device)

            # prediction
            pred = model(batched_graphs, feature_aggregation_func(batched_graphs))

            # calculate loss
            # we need to calculate the weight for each batch element before we pass it into F.binary_cross_entropy
            # https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy.html#torch.nn.functional.binary_cross_entropy
            with batched_graphs.local_scope():
                loss_weights = th.tensor([
                    class_weights[1]
                    if rounded_prediction else class_weights[0]
                    for rounded_prediction in th.round(pred)
                ],
                                         dtype=th.float,
                                         device=device).reshape((-1, 1))

            loss = F.binary_cross_entropy(pred, batched_labels, weight=loss_weights)
            train_loss_history.append(loss.item())

            # get prediction
            if anomaly_threshold:
                pred = (pred > anomaly_threshold).float()
            else:
                pred = th.round(pred)

            num_correct = (pred == batched_labels).sum().item()
            training_num_correct += num_correct

            if num_correct == 0:  # batch size = 1, so if training_num_correct
                # is 0, then it misclassified
                train_incorrect_graphs.append((
                    "graph",
                    f"misclassified as {'benign' if batched_labels[0].item() else 'anomaly'}"
                ))

            train_y_pred = train_y_pred + th.reshape(pred, (pred.shape[0],)).tolist()
            train_y_true = train_y_true + batched_labels.tolist()

            # optimize
            opt.zero_grad()
            loss.backward()
            opt.step()

        validation_loss_history = []
        validation_num_correct = 0
        validation_y_pred = []
        validation_y_true = []
        validation_incorrect_graphs = [] if benign_downsampling_training else None
        
        for batched_graphs, batched_labels in val_dataloader:
            batched_graphs = batched_graphs.to(device)
            batched_labels = th.reshape(
                batched_labels, (batched_labels.shape[0], 1)).to(device)

            with batched_graphs.local_scope():
                pred = model(batched_graphs,
                             feature_aggregation_func(batched_graphs))

                # we need to calculate the weight for each batch element before we pass it into F.binary_cross_entropy
                # https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy.html#torch.nn.functional.binary_cross_entropy
                loss_weights = th.tensor([
                    class_weights[1]
                    if rounded_prediction else class_weights[0]
                    for rounded_prediction in th.round(pred)
                ],
                                         dtype=th.float,
                                         device=device).reshape(-1, 1)

                loss = F.binary_cross_entropy(pred,
                                              batched_labels,
                                              weight=loss_weights)
                validation_loss_history.append(loss.item())

                # for binary classification, use rounding
                if anomaly_threshold:
                    pred = (pred > anomaly_threshold).float()
                else:
                    pred = th.round(pred)

                num_correct = (pred == batched_labels).sum().item()
                validation_num_correct += num_correct

                if num_correct == 0:
                    validation_incorrect_graphs.append((
                        batched_graphs.folder_name,
                        f"misclassified as {'benign' if batched_labels[0].item() else 'anomaly'}"
                    ))

                validation_y_pred = validation_y_pred + th.reshape(
                    pred, (pred.shape[0], )).tolist()
                validation_y_true = validation_y_true + batched_labels.tolist()

        training_accuracy, avg_training_loss = cal_acc_and_loss(
            training_num_correct, len(train_dataset), train_loss_history)
        
        validation_accuracy, avg_validation_loss = cal_acc_and_loss(
            validation_num_correct, len(validation_dataset), validation_loss_history)

        outputEpochStats(epoch, training_accuracy, avg_training_loss,
                         validation_accuracy, avg_validation_loss, logger)

        training_accuracy_hist.append(training_accuracy)
        avg_training_loss_hist.append(avg_training_loss)
        validation_accuracy_hist.append(validation_accuracy)
        avg_validation_loss_hist.append(avg_validation_loss)

        outputPerEpochStatsToSummaryWriter(summaryWriter, epoch,
                                           training_accuracy,
                                           avg_training_loss,
                                           validation_accuracy,
                                           avg_validation_loss)

    test_num_correct = 0
    test_y_pred = []
    test_y_true = []
    anomaly_prob = []
    test_incorrect_graphs = [] if benign_downsampling_training else None
    for batched_graphs, batched_labels in test_dataloader:
        batched_graphs = batched_graphs.to(device)
        batched_labels = batched_labels.to(device)

        pred = model(batched_graphs, feature_aggregation_func(batched_graphs))
        if anomaly_threshold:
            pred = (pred > anomaly_threshold).float()
        else:
            anomaly_prob.append(pred[batched_labels.long() == 1].view(-1).tolist())

            pred = th.round(pred)

        pred = th.reshape(pred, (pred.shape[0],))

        num_correct = (pred == batched_labels).sum().item()
        test_num_correct += num_correct

        if num_correct == 0:
            test_incorrect_graphs.append((
                batched_graphs.folder_name,
                f"misclassified as {'benign' if batched_labels[0].item() else 'anomaly'}"
            ))

        test_y_pred = test_y_pred + pred.tolist()
        test_y_true = test_y_true + batched_labels.tolist()

    if not anomaly_threshold:
        saveProbabilityDistibution([round(val, 2) for vals in anomaly_prob for val in vals])

    outputStats('test',
                test_num_correct,
                len(test_dataset),
                test_y_pred,
                test_y_true, ['Benign', 'Anamoly'],
                logger,
                incorrect_graphs=test_incorrect_graphs)
    
    outputTestStatsToSummaryWriter(
        summaryWriter, test_num_correct, len(test_dataset),
        len(train_dataset) + len(validation_dataset) + len(test_dataset))


def evaluate_binary_graph_classification(
    model,
    summaryWriter,
    test_dataset,
    logger,
    batch_size=DEFAULT_BATCH_SIZE,
    feature_aggregation_func=None,
    device=None,
    benign_downsampling_training=None,
    anomaly_threshold=None,
):


    logger.info(f"Evaluating on Device: {device}")
    logger.info(f"# Parameters in model: {getTotalParams(model)}")
    logger.info(f"# Trainable parameters in model: {getTotalTrainableParams(model)}")

    model.to(device)

    logger.info(f"Stratified sampler enabled")

    test_batch_sampler = StratifiedBatchSampler(
        test_dataset.labels, batch_size=batch_size, shuffle=False
    )

    test_dataloader = GraphDataLoader(
        test_dataset,
        collate_fn=collate_func,
        num_workers=DEFAULT_NUM_WORKERS,
        batch_sampler=test_batch_sampler,
    )

    test_num_correct = 0
    test_y_pred = []
    test_y_true = []
    anomaly_prob = []
    test_incorrect_graphs = [] if benign_downsampling_training else None
    
    for batched_graphs, batched_labels in test_dataloader:
        batched_graphs = batched_graphs.to(device)
        batched_labels = batched_labels.to(device)

        pred = model(batched_graphs, feature_aggregation_func(batched_graphs))
        
        if anomaly_threshold:
            pred = (pred > anomaly_threshold).float()
        else:
            anomaly_prob.append(pred[batched_labels.long() == 1].view(-1).tolist())
            pred = th.round(pred)

        pred = th.reshape(pred, (pred.shape[0],))

        num_correct = (pred == batched_labels).sum().item()
        test_num_correct += num_correct

        if num_correct == 0:
            test_incorrect_graphs.append(
                (
                    batched_graphs.folder_name,
                    f"misclassified as {'benign' if batched_labels[0].item() else 'anomaly'}",
                )
            )

        test_y_pred = test_y_pred + pred.tolist()
        test_y_true = test_y_true + batched_labels.tolist()

    if not anomaly_threshold:
        saveProbabilityDistibution(
            [round(val, 2) for vals in anomaly_prob for val in vals]
        )

    outputStats(
        "test",
        test_num_correct,
        len(test_dataset),
        test_y_pred,
        test_y_true,
        ["Benign", "Anomaly"],
        logger,
        incorrect_graphs=test_incorrect_graphs,
    )
    outputTestStatsToSummaryWriter(
        summaryWriter, test_num_correct, len(test_dataset), len(test_dataset)
    )
    
# Based upon dgl.dataloading.GraphCollator.collate
# used for dataloader collation & to inject folder name into a batched graph
def collate_func(items):
    elem = items[0]
    elem_type = type(elem)
    if isinstance(elem, dgl.DGLHeteroGraph):
        batched_graphs = dgl.batch(items)
        batched_graphs.folder_name = elem.folder_name  # assign folder name for the batch of graphs to be the name
        # of the first graph
        batched_graphs.additional_node_data = elem.additional_node_data
        return batched_graphs
    elif th.is_tensor(elem):
        return th.stack(items, 0)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            return collate_func([F.tensor(b) for b in items])
        elif elem.shape == ():  # scalars
            return th.tensor(items)
    elif isinstance(elem, float):
        return th.tensor(items, dtype=th.float64)
    elif isinstance(elem, int):
        return th.tensor(items)
    elif isinstance(elem, (str, bytes)):
        return items
    elif isinstance(elem, Mapping):
        return {key: collate_func([d[key] for d in items]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate_func(samples) for samples in zip(*items)))
    elif isinstance(elem, Sequence):
        # check to make sure that the elements in batch have consistent size
        item_iter = iter(items)
        elem_size = len(next(item_iter))
        if not all(len(elem) == elem_size for elem in item_iter):
            raise RuntimeError(
                'each element in list of batch should be of equal size')
        transposed = zip(*items)
        return [collate_func(samples) for samples in transposed]

    raise TypeError('collate_func encountered an unexpected input type')
