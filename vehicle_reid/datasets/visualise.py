import torch
import matplotlib.pyplot as plt

from vehicle_reid.datasets import VehicleReIdDataset, match_dataset


def visualise(dataset: VehicleReIdDataset):
    """Basic visualisation for the dataset

    Note: do not use the raw VehicleReIdDataset type.
    """

    for i in range(16):
        ax = plt.subplot(4, 4, i+1)
        plt.tight_layout()
        ax.axis('off')

        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label, _, _ = dataset[sample_idx]
        img = img.permute(1, 2, 0)
        plt.title(label)
        plt.imshow(img)

    plt.show()

def visualise_dataset():
    # use images from the train set
    dataset = match_dataset("train")
    visualise(dataset)

