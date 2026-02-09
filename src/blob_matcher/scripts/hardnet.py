# Copyright 2019 EPFL, Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This is HardNet local patch descriptor. The training code is based on PyTorch TFeat implementation
https://github.com/edgarriba/examples/tree/master/triplet
by Edgar Riba.

If you use this code, please cite ::

    @article{HardNet2017,
    author = {Anastasiya Mishchuk, Dmytro Mishkin, Filip Radenovic, Jiri Matas},
        title = "{Working hard to know your neighbor's margins:Local descriptor learning loss}",
        year = 2017}
    (c) 2017 by Anastasiia Mishchuk, Dmytro Mishkin
"""


from __future__ import division, print_function


import argparse
import random
import os


import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim


from blob_matcher.hardnet.data import Augmentor, \
    BlobinatorTrainingData, \
    BlobinatorValidationData, \
    BlobinatorValidationPreBatchedData
from blob_matcher.hardnet.loggers import FileLogger
from blob_matcher.hardnet.losses import distance_matrix_vector, loss_HardNet_weighted
from blob_matcher.hardnet.models import HardNet
from blob_matcher.hardnet.utils import show_images
from blob_matcher.hardnet.eval_metrics import ErrorRateAt95Recall


def create_train_loader(cfg):

    kwargs = {
        'num_workers': cfg.TRAINING.NUM_WORKERS,
        'pin_memory': cfg.TRAINING.PIN_MEMORY
    } if not cfg.TRAINING.NO_CUDA else {}

    transformer_dataset = BlobinatorTrainingData(cfg, os.path.join(cfg.BLOBINATOR.DATASET_PATH, "training"))
    # transformer_dataset.preprocess()
    train_loader = torch.utils.data.DataLoader(
        transformer_dataset,
        batch_size=cfg.TRAINING.BATCH_SIZE,
        shuffle=True,
        **kwargs)

    return train_loader


def create_test_loaders(cfg):

    kwargs = {
        'num_workers': cfg.TRAINING.NUM_WORKERS,
        'pin_memory': cfg.TRAINING.PIN_MEMORY
    } if not cfg.TRAINING.NO_CUDA else {}

    val_loaders = [{
        'name':
        'duplicated_blobs_validation',
        'dataloader':
        torch.utils.data.DataLoader(
            BlobinatorValidationPreBatchedData(cfg, os.path.join(cfg.BLOBINATOR.DATASET_PATH,  "validation")),
            batch_size=None,
            shuffle=True,
            **kwargs
        )
    } if os.path.basename(cfg.BLOBINATOR.DATASET_PATH) == "real" else {
        'name':
        'no_duplicated_blobs_validation',
        'dataloader':
        torch.utils.data.DataLoader(
            BlobinatorValidationData(cfg, os.path.join(cfg.BLOBINATOR.DATASET_PATH,  "validation")),
            batch_size=cfg.TEST.TEST_BATCH_SIZE,
            shuffle=True,
            **kwargs
        )
    }]

    test_loaders = None
    return val_loaders, test_loaders


# function to train the network
def train(cfg,
          train_loader,
          val_loaders,
          test_loaders,
          model,
          augm,
          optimizer,
          epoch,
          device,
          logger,
          file_logger,
          best_fpr_val=100):
    # switch to train mode
    model.train()
    pbar = tqdm(enumerate(train_loader))

    # for batch_idx, data in pbar: # iterate over batches of train_loader
    for batch_idx, data in enumerate(
            train_loader):  # iterate over batches of train_loader
        # print(batch_idx)

        img_a, img_p, img_g, garbage_available = data
        img_g = img_g[garbage_available]
        img_a = img_a.to(device)
        img_p = img_p.to(device)
        img_g = img_g.to(device)

        # forward-propagate input through network and get [batchSize x 1 x
        # resolution x resolution] output array
        out_a, p_a = model(img_a)

        # forward-propagate input through network and get [batchSize x 1 x
        # resolution x resolution] output array
        out_p, p_p = model(img_p)

        if img_g.size(0) > 0:
            out_g, p_g = model(img_g)
        else:
            out_g, p_g = None, None

        loss, min_neg_idx = loss_HardNet_weighted(
            out_a,
            out_p,
            out_g,
            anchor_swap=cfg.TRAINING.ANCHOR_SWAP,
            margin=cfg.TRAINING.MARGIN,
            batch_reduce=cfg.TRAINING.BATCH_REDUCE,
            loss_type=cfg.TRAINING.LOSS)

        # plot triplets (anchor, positive, hardest negative)
        if batch_idx % 40 == 0:
            tripletIDX = 2
            show_images([
                p_a[tripletIDX, :, :, :].squeeze().data.cpu().numpy() * 255,
                p_p[tripletIDX, :, :, :].squeeze().data.cpu().numpy() * 255,
                (torch.cat([p_p, p_g]) if p_g is not None else p_p)[min_neg_idx[tripletIDX], :, :, :].squeeze().data.cpu(
                ).numpy() * 255
            ], cfg.LOGGING.IMGS_DIR + '/img_' + '.png')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        adjust_learning_rate(cfg, optimizer)

        # logging
        if batch_idx % cfg.LOGGING.LOG_INTERVAL == 0:
            #logger.log_value('loss', loss.item()).step()

            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * img_a.size(0),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        # evaluate the network on validation data
        if batch_idx % cfg.TEST.EVAL_INTERVAL == 0 and batch_idx > 0:

            for data_loader in val_loaders:
                # get FPR at TPR 95% on validation data
                fpr_val_ = test(cfg, data_loader['dataloader'], model, device,
                                epoch, logger, file_logger, data_loader['name'])
                if fpr_val_ < best_fpr_val:
                    best_fpr_val = fpr_val_
                    print('saving best model with val fpr: {}'.format(
                        best_fpr_val))
                    os.makedirs(cfg.LOGGING.MODEL_DIR, exist_ok=True)
                    torch.save(
                        {
                            'best_fpr_val:': best_fpr_val,
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict()
                        }, '{}/_best_model_checkpoint.pth'.format(
                            cfg.LOGGING.MODEL_DIR))

                    torch.save(
                        {"optimizer": optimizer.state_dict()},
                        '{}/_best_optimizer_checkpoint.pth'.format(
                            cfg.LOGGING.MODEL_DIR))

            # clean cache and switch back to train mode
            torch.cuda.empty_cache()
            model.train()

        # evaluate the network on test data
        # if batch_idx % cfg.TEST.EVAL_INTERVAL == 0 and batch_idx > 0:
        #     for data_loader in test_loaders:
        #         test(cfg, data_loader['dataloader'], model, augm, epoch,
        #              logger, data_loader['name'])

            # clean cache and switch back to train mode
            torch.cuda.empty_cache()
            model.train()

    # checkpoint the model
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict()
    }, '{}/model_checkpoint_{}.pth'.format(cfg.LOGGING.MODEL_DIR, epoch))

    # checkpoint the optimizer
    torch.save({'optimizer': optimizer.state_dict()},
               '{}/optimizer_checkpoint_{}.pth'.format(cfg.LOGGING.MODEL_DIR,
                                                       epoch))

    return best_fpr_val


def filter_orientation(out_a, out_p, label, theta_a, theta_p, orientCorrect,
                       filter_value):

    # get the orientations of the anchor and the paired keypoint
    orientations_a = np.degrees(theta_a[2].data.cpu().numpy())
    orientations_p = np.degrees(theta_p[2].data.cpu().numpy())

    # get orientation correction terms (in radians), which denote the angle from the anchor keypoint's orientation
    # to its reprojection's orientation, ie the orientation correction terms
    # need to be subtracted from the paired keypoint
    orient_correct = np.degrees(orientCorrect.data.cpu().numpy())
    orient_p_corr = orientations_p - orient_correct

    #differences   = np.abs(orientations_a-orientations_p)

    # compute the difference in angles (taking care of wrap-around, such that e.g. diff(355°,5°) = 10°
    # and 180° being the maximum possible difference between two keypoint
    # orientations)
    differences = 180 - abs(abs(orientations_a - orient_p_corr) % 360 - 180)
    labels_numpy = label.data.cpu().numpy()

    indices_filtered = np.where(differences <= filter_value)[0]
    indices_positives = np.where(labels_numpy == 1)[0]

    indices_filtered_positives = np.intersect1d(indices_filtered,
                                                indices_positives)

    indices_negatives = np.where(labels_numpy == 0)[0]
    total_indices = np.concatenate(
        [indices_filtered_positives, indices_negatives])

    out_a = out_a[total_indices]
    out_p = out_p[total_indices]
    label = label[total_indices]

    return out_a, out_p, label


# function to test the network


def test(cfg, test_loader, model, device, epoch, logger, file_logger, logger_test_name):
    # switch to evaluate mode
    model.eval()
    
    pbar = tqdm(enumerate(test_loader))
        # data for our data set: img,img,meta,meta,ID,ID,match
        # data for Brown data  : img,img,match

        # patchwise = len(data) < 7
        # imgIDs_a = None if patchwise else data[4]
        # img_a = data[0].to(device) if not imgIDs_a else dict(
        #     zip(imgIDs_a, data[0]["img"]))  # create dictionary of images
        # theta_a = None if patchwise else [
        #     theta.float().to(device) for theta in data[2]#[0:-1]
        # ]

        # imgIDs_p = None if patchwise else data[5]
        # img_p = data[1].to(device) if not imgIDs_p else dict(
        #     zip(imgIDs_p, data[1]["img"]))  # create dictionary of images
        # theta_p = None if patchwise else [
        #     theta.float().to(device) for theta in data[3]#[0:-1]
        # ]

        # # get scale correction factor, orientation correction constant and check whether the second keypoint
        # # falls on the anchor's image (in which case not to apply correction)
        # # or on the paired image (requiring correction)
        # # for test data, negative keypoints may be on the anchor's image or on
        # # the paired image
        # diffImg = None if patchwise else data[8]
        # scaleCorrect = None if patchwise else 1 - diffImg + diffImg * \
        #     data[9]  # set to correction factor if on different images, 0 else
        # # set to correction factor if on different images, 0 else
        # orientCorrect = None if patchwise else diffImg * data[10]
    if isinstance(test_loader.dataset, BlobinatorValidationData):
        num_tests = 0
        labels, distances = [], []
        for batch_idx, data in pbar:
            img_a, img_p, label = data
            img_a = img_a.to(device)
            img_p = img_p.to(device)

            # forward-propagate input through network and get [batchSize x 1 x
            # resolution x resolution] output array
            out_a, _ = model(img_a)

            # forward-propagate input through network and get [batchSize x 1 x
            # resolution x resolution] output array
            out_p, _ = model(img_p)

            label = label.to(device)

            # if cfg.TEST.ENABLE_ORIENTATION_FILTERING and theta_a:
            #     # Filter for all the orienations less than specified value
            #     out_a, out_p, label = filter_orientation(
            #         out_a, out_p, label, theta_a, theta_p, orientCorrect,
            #         cfg.TRAINING.ORIENTATION_FILTER_VALUE)

            num_tests += len(out_a)

            dists = torch.sqrt(torch.sum((out_a - out_p)**2,
                                        1))  # euclidean distance
            distances.extend(dists.data.cpu().numpy().reshape(-1, 1))
            ll = label.data.cpu().numpy().reshape(-1, 1)
            labels.extend(ll)

            if batch_idx % cfg.LOGGING.LOG_INTERVAL == 0:
                pbar.set_description(logger_test_name +
                                    ' Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                                        epoch, batch_idx * len(label),
                                        len(test_loader.dataset), 100. *
                                        batch_idx / len(test_loader)))
        # compute statistics
        print('Number of test samples: {}'.format(num_tests))
        labels = np.vstack(labels).reshape(num_tests)
        distances = np.vstack(distances).reshape(num_tests)
        if cfg.TEST.ENABLE_ORIENTATION_FILTERING:

            positive_indices = np.where(labels == 1)[0]
            negative_indices = np.where(labels == 0)[0]

            # to get random same negatives, not to omit any particular sequence
            random.Random(42).shuffle(negative_indices)

            negative_indices = negative_indices[0:len(positive_indices)]

            labels = np.concatenate(
                [labels[positive_indices], labels[negative_indices]])
            distances = np.concatenate(
                [distances[positive_indices], distances[negative_indices]])

        fpr95 = ErrorRateAt95Recall(labels, 1.0 / (distances + 1e-8))
        print('\33[91m{} Test set: Accuracy(FPR95): {:.8f}\n\33[0m'.format(
            logger_test_name, fpr95))
    else:
        fpr95_num = 0
        fpr95_sum = 0
        for batch_idx, data in pbar:
            img_a, img_p, img_g = data

            out_a, _ = torch.cat(img_a.to(device))
            out_p, _ = torch.cat(img_p.to(device))
            out_g, _ = torch.cat(img_g.to(device))

            distances = distance_matrix_vector(out_a, torch.concat((out_p, out_g))).detach().cpu().numpy().flatten()
            label = torch.eye(out_a.size(0), out_p.size(0) + out_g.size(0)).cpu().numpy().flatten()
            fpr95_num += distances.size
            fpr95_sum += distances.size * ErrorRateAt95Recall(label, 1.0 / (distances + 1e-8))
            pbar.set_description(logger_test_name +
                                    ' Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                                        epoch, batch_idx * len(label),
                                        len(test_loader.dataset), 100. *
                                        batch_idx / len(test_loader)))
        fpr95 = fpr95_sum / fpr95_num
    if (cfg.LOGGING.ENABLE_LOGGING):
        #logger.log_value(logger_test_name + ' fpr95', fpr95)
        file_logger.log_string(
            'stats.txt',
            'Epoch: {},  Dataset:{}, FPR: {} '.format(epoch, logger_test_name,
                                                      fpr95))
    return fpr95


def adjust_learning_rate(cfg, optimizer):
    """
    Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for idx, group in enumerate(optimizer.param_groups):
        init_lr = cfg.TRAINING.LR
        if 'step' not in group:
            group['step'] = 0.
        else:
            group['step'] += 1.

        group['lr'] = init_lr * (
            1.0 - float(group['step']) * float(cfg.TRAINING.BATCH_SIZE) /
            (cfg.TRAINING.N_TRIPLETS * float(cfg.TRAINING.EPOCHS)))
    return


def create_optimizer(cfg, model):
    # setup optimizer
    if cfg.TRAINING.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=cfg.TRAINING.LR,
                              momentum=0.9,
                              dampening=0.9,
                              weight_decay=cfg.TRAINING.W_DECAY)
    elif cfg.TRAINING.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.TRAINING.LR,
                               weight_decay=cfg.TRAINING.W_DECAY)
    else:
        raise Exception('Not supported optimizer: {0}'.format(
            cfg.TRAINING.OPTIMIZER))
    return optimizer


def main():
    from blob_matcher.configs.defaults import _C as cfg

    parser = argparse.ArgumentParser(description="HardNet Training")
    config_path = os.path.join(os.getcwd(), "configs", "init.yml")

    parser.add_argument("--config_file",
                        #default=config_path,
                        default=None,
                        help="path to config file",
                        type=str)
    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(
        os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)

    cfg.merge_from_list(args.opts)

    run_training(cfg)


def run_training(cfg):
    cfg = create_logging_directories(cfg)

    if not cfg.TRAINING.NO_CUDA:
        # cudnn.benchmark = True
        torch.cuda.manual_seed_all(cfg.TRAINING.SEED)
        torch.backends.cudnn.deterministic = True

    # set random seeds
    random.seed(cfg.TRAINING.SEED)
    torch.manual_seed(cfg.TRAINING.SEED)
    np.random.seed(cfg.TRAINING.SEED)

    model = HardNet(patch_size=cfg.TRAINING.IMAGE_SIZE)
    logger, file_logger = None, None

    if cfg.LOGGING.ENABLE_LOGGING:
        logger = None
        file_logger = FileLogger(cfg.LOGGING.LOG_DIR)
        file_logger.log_string('cfg.txt', str(cfg))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.TRAINING.GPU_ID)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    # print the experiment configuration
    print('\nparsed options:\n{}\n'.format(cfg))

    model.to(device)  # place model on device
    augm = Augmentor(cfg, device)

    # set up optimizer for the model
    optimizer1 = create_optimizer(cfg, model)

    if cfg.TRAINING.RESUME:
        if os.path.isfile(cfg.TRAINING.RESUME):
            print('=> loading checkpoint {}'.format(cfg.TRAINING.RESUME))
            checkpoint = torch.load(cfg.TRAINING.RESUME)
            cfg.TRAINING.START_EPOCH = checkpoint['epoch']
            start = cfg.TRAINING.START_EPOCH
            end = cfg.TRAINING.EPOCHS

            checkpoint = torch.load(cfg.TRAINING.RESUME)
            model.load_state_dict(checkpoint['state_dict'])

            path_to_optimizer = cfg.TRAINING.RESUME.replace(
                'model_checkpoint', 'optimizer_checkpoint')
            print('Path to the checkpoint of optimizer:', path_to_optimizer)
            optimizer1.load_state_dict(
                torch.load(path_to_optimizer)['optimizer'])

        else:
            print('=> no checkpoint found at {}'.format(cfg.TRAINING.RESUME))
    else:
        start = cfg.TRAINING.START_EPOCH
        end = start + cfg.TRAINING.EPOCHS

    # create training data set
    train_loader = create_train_loader(cfg)
    # create testing data sets
    val_loaders, test_loaders = create_test_loaders(cfg)

    # val_loaders[0]["dataloader"].dataset.validation_keypoints = train_loader.dataset.validation_keypoints
    # val_loaders[0]["dataloader"].dataset.validation_background_filenames = train_loader.dataset.validation_background_filenames

    best_fpr_val = 100  # initial value of best FPR

    # train the network for a given epoch interval
    for epoch in range(start, end):

        # if epoch > start:

            # get new training pairs at the start of each epoch
            # train_loader.dataset.get_pairs(train_sequences)

        torch.cuda.empty_cache()
        # train the network on the current epoch's data
        best_fpr_val = train(cfg, train_loader, val_loaders, test_loaders,
                             model, augm, optimizer1, epoch, device, logger,
                             file_logger, best_fpr_val)
        torch.cuda.empty_cache()


def create_logging_directories(cfg):

    # add experiment name to path
    cfg.LOGGING.LOG_DIR = os.path.join(cfg.LOGGING.LOG_DIR,
                                       cfg.TRAINING.EXPERIMENT_NAME)
    cfg.LOGGING.MODEL_DIR = os.path.join(cfg.LOGGING.MODEL_DIR,
                                         cfg.TRAINING.EXPERIMENT_NAME)
    cfg.LOGGING.IMGS_DIR = os.path.join(cfg.LOGGING.IMGS_DIR,
                                        cfg.TRAINING.EXPERIMENT_NAME)

    log_directories = [
        cfg.LOGGING.LOG_DIR, cfg.LOGGING.MODEL_DIR, cfg.LOGGING.IMGS_DIR
    ]

    for log_dir in log_directories:

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    return cfg


if __name__ == '__main__':
    main()
