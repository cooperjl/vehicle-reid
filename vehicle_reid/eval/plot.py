import os

import matplotlib.pyplot as plt
import numpy as np

from vehicle_reid.config import cfg


def visualise_logfile(filename: str, label: str, axs):
    """
    Parse a logfile, to visualise the loss and mean average precision.

    Parameters
    ----------
    filename : str
        Path to the logfile
    label : str
        Label for the legend of the plot
    axs : list[Axes, Axes]
        Collection of axes for the plot to be drawn.
    """
    losses = []
    accuracies = []
    start_epoch = None
    end_epoch = None

    with open(filename, "r") as f:
        for line in f:
            if "Epoch" in line:
                epoch = int(line.split("Epoch")[1].split(":")[0][1:])
                if start_epoch is None:
                    start_epoch = epoch
                end_epoch = (
                    epoch  # Updated each time, so will be the final one in the end.
                )
                continue

            # Since we know the exact format of the log file, we can index exactly in the line.
            if "mAP: " in line:
                accuracy = line.split("mAP: ")[1][:-2]
                accuracies.append(float(accuracy))
            if "Loss: " in line:
                loss = line.split("Loss: ")[1].split("(")[1].split(")")[0]
                losses.append(float(loss))

    # Assert start and end has been found, prevent the linter from thinking they can be None
    assert start_epoch is not None and end_epoch is not None

    epochs_a = np.linspace(start_epoch, end_epoch, num=len(accuracies))
    epochs_l = np.linspace(start_epoch, end_epoch + 1, num=len(losses))

    axs[0].plot(epochs_l, losses, "-", label=label)
    axs[1].plot(epochs_a, accuracies, "-", label=label)

    axs[0].set_ylabel("average loss")
    axs[1].set_ylabel("mean average precision")

    for ax in axs:
        ax.set_xticks(np.arange(start_epoch, end_epoch + 1, 1.0))
        ax.set_xlabel("epochs")
        ax.legend()


def plot():
    """Main plot function, to plot the loss and mAP for visualisation."""

    logfiles = {
        "l1.txt": "1.0:1.0",
        "l2.txt": "0.5:0.5",
        "l3.txt": "0.5:1.0",
        "l4.txt": "1.0:2.0",
        "l5.txt": "1.0:0.5",
        "l6.txt": "2.0:1.0",
    }

    _, axs = plt.subplots(1, 2)

    for log, label in logfiles.items():
        visualise_logfile(os.path.join(cfg.MISC.LOG_DIR, log), label, axs)

    plt.tight_layout()
    plt.show()
