import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader

from vehicle_reid.config import cfg
from vehicle_reid.datasets import match_dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# TODO have the args be global its stupid passing all this data around
def load_data(split: str):
    batch_size = cfg.SOLVER.BATCH_SIZE if split == "train" else cfg.TEST.BATCH_SIZE
    
    dataset = match_dataset(split)

    #mask = list(range(0, len(dataset), 20))
    #dataset_subsample = torch.utils.data.Subset(dataset, mask)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    return dataset, dataloader
