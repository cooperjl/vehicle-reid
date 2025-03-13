import os
from argparse import BooleanOptionalAction, Namespace
from contextlib import nullcontext

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from tqdm import tqdm

from vehicle_reid import args, gms, losses, utils
from vehicle_reid.datasets import load_data
from vehicle_reid.eval import eval_model
from vehicle_reid.model import cresnet50
from vehicle_reid.optimizer import init_optimizer

torch.multiprocessing.set_sharing_strategy('file_system')

triplet_loss_fn = losses.TripletLoss()
ce_loss_fn = losses.CrossEntropyLoss()

CFG = Namespace()
rng = np.random.default_rng()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

scaler = torch.GradScaler()

def parse_arguments():
    parser = args.add_subparser(name="train", help="train the model")
    parser.add_argument('dataset', metavar='dataset',
                        choices=args.DATASETS,
                        help='the name of the dataset to train on')
    parser.add_argument('--data-path', type=args.dir_path,
                        default=args.DEFAULTS.data_path,
                        help='path where the datasets are stored (default: %(default)s)')
    parser.add_argument('--gms-path', type=args.dir_path,
                        default=args.DEFAULTS.gms_path,
                        help='path to load gms matches from (default: %(default)s)')
    parser.add_argument('--epochs', type=int,
                        required=True,
                        help='number of epochs for training')
    parser.add_argument('--batch_size', type=int,
                        required=True,
                        help='batch size')
    parser.add_argument('--width', type=int,
                        default=args.DEFAULTS.width,
                        help='width to resize images to (default: %(default)s)')
    parser.add_argument('--height', type=int,
                        default=args.DEFAULTS.height,
                        help='width to resize images to (default: %(default)s)')
    parser.add_argument('--amp', action=BooleanOptionalAction,
                        help='enable amp support for use on gpus which benefit from it')
    parser.set_defaults(func=train)


def train_one_epoch(model, optimizer, dataset, dataloader, gms_dict: dict, desc=""):
    """
    Singular epoch training function.
    """

    model.train()

    for images, labels, indices, _ in tqdm(dataloader, desc=desc):
        triplets, tri_labels = mine_triplets(dataset, gms_dict, images, labels, indices)

        triplets = triplets.to(device)
        tri_labels = tri_labels.to(device)

        optimizer.zero_grad()
        # global module TODO rename and reorder
        with torch.autocast(device_type='cuda') if CFG.amp else nullcontext():
            outputs, features = model(triplets)

            # need to decide whether to calculate cross entropy loss for whole triplet or just anchor
            ce_loss = ce_loss_fn(outputs[0:CFG.batch_size], tri_labels[0:CFG.batch_size])
            
            #magic_gcn_work_time_todo(features, tribak)

            triplet_loss = triplet_loss_fn(features, tri_labels)

            loss = (1.0 * triplet_loss) + (1.0 * ce_loss)
        
        if CFG.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()


def magic_gcn_work_time_todo(featuremaps: torch.Tensor, triplets: torch.Tensor):
    # TODO This should not be here! this is really a test okayyyy
    size = featuremaps.size()
    pmaps = []
    for map in featuremaps:
        map = map.squeeze(0)
        pmaps.append((torch.sum(map, 0)/map.shape[0]).to('cpu').detach().numpy())
    #(top_left, top_right, bottom_left, bottom_right, centre) = transforms.FiveCrop(size=(150, 150))(featuremaps)
    fig = plt.figure(figsize=(2, 3))
    fig.add_subplot(2, 3, 1)
    plt.imshow(pmaps[0])
    fig.add_subplot(2, 3, 2)
    plt.imshow(pmaps[CFG.batch_size])
    fig.add_subplot(2, 3, 3)
    plt.imshow(pmaps[CFG.batch_size*2])
    fig.add_subplot(2, 3, 4)
    plt.imshow(triplets[0].to(torch.uint8).permute(1, 2, 0))
    fig.add_subplot(2, 3, 5)
    plt.imshow(triplets[CFG.batch_size].to(torch.uint8).permute(1, 2, 0))
    fig.add_subplot(2, 3, 6)
    plt.imshow(triplets[CFG.batch_size*2].to(torch.uint8).permute(1, 2, 0))


    plt.show()



def mine_triplets(dataset, gms_dict: dict, images: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor):
    triplets = torch.zeros((CFG.batch_size * 3, 3, CFG.height, CFG.width), dtype=torch.float32)
    tri_labels = torch.zeros((CFG.batch_size * 3), dtype=torch.int64)

    for i, label in enumerate(labels):
        anchor = images[i]

        label_str = utils.pad_label(label.item(), dataset.name)
        index = int(indices[i].item())

        matches = gms_dict[label_str][index]

        # mean threshold
        threshold = np.mean(matches)
        # max threshold
        # threshold = np.max(matches)

        # mask out values which are below the threshold, and select the smallest
        # maybe if below say 50, we switch to max instead of mean? NOTE: important to remember this
        pos_idx = np.argmin(ma.masked_where(matches < threshold, matches))
        positive = dataset.get_by_index(label_str, pos_idx)

        # select a random negative anchor
        neg_label = label_str
        while neg_label == label_str or neg_label not in dataset.label_index:
            neg_label = rng.choice(np.fromiter(dataset.label_index.keys(), dtype='<U4'))

        neg_idx = rng.integers(0, len(dataset.label_index[neg_label]))
        negative = dataset.get_by_index(neg_label, neg_idx)

        triplets[i] = anchor
        triplets[i + CFG.batch_size] = positive
        triplets[i + (CFG.batch_size * 2)] = negative

        tri_labels[i] = label
        tri_labels[i + CFG.batch_size] = label
        tri_labels[i + (CFG.batch_size * 2)] = int(neg_label)

    return triplets, tri_labels


def train(args: Namespace):
    print("Train called")

    global CFG 
    CFG = args # TODO have a global args

    gms_path = os.path.join(args.gms_path, args.dataset)
    gms_dict = gms.load_data(gms_path)

    dataset, dataloader = load_data(args, "train")

    model = cresnet50(dataset.num_classes)
    model = model.to(device)
    optimizer = init_optimizer("sgd", model.parameters())

    for epoch in range(args.epochs):
        train_one_epoch(model, optimizer, dataset, dataloader, gms_dict, desc=f"epoch {epoch+1}")
        eval_model(model, args)

