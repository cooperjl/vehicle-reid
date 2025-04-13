from tqdm import tqdm

from vehicle_reid.datasets import load_data


def calculate_normal_values():
    """Calculate and print the mean and std of the dataset set in the configuration."""

    dataset, dataloader = load_data("normal")

    running_mean = 0
    running_std = 0

    for images, *_ in tqdm(dataloader):
        # reshape the tensor from (batch_size, 3, 128, 128) to (batch_size, 3, 128*128)
        images = images.view(images.size(0), images.size(1), -1)

        # calculate the mean and std over the 128*128 dimension which includes all the values of the images
        # sum those values and add them to running value
        running_mean += images.mean(dim=2).sum(dim=0)
        running_std += images.std(dim=2).sum(dim=0)

    # print values to be copied into respective config files
    print(f"mean: {running_mean/len(dataset)}")
    print(f"std: {running_std/len(dataset)}")


if __name__ == "__main__":
    calculate_normal_values()

