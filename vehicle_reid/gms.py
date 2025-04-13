import json
import logging
import os

import cv2 as cv
import numpy as np
from tqdm import tqdm

from vehicle_reid.config import cfg
from vehicle_reid.datasets import match_dataset
from vehicle_reid.utils import NumpyEncoder, pad_label

logger = logging.getLogger(__name__)

def main():
    """
    Main function, which computes the index files and gms files for dataset specified using configuration file, and
    writes them to .json files.
    """
    # train hardcoded as gms matches should only ever use the train set.
    dataset = match_dataset("train")

    output = os.path.join(cfg.MISC.GMS_PATH, cfg.DATASET.NAME)
    if(not os.path.exists(output)):
        os.mkdir(output)
        logger.info(f"Output directory created at: {output}")

    orb = cv.ORB_create(nfeatures=10000, fastThreshold=0)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

    grouped = dataset.get_grouped()

    # create the index for the gms values
    image_index = {}
    label_index = {} # guarantee the order matches the rest of script for future loading
    label_idx = 0
    for label, images in grouped:
        padded_label = pad_label(label, cfg.DATASET.NAME)
        # "label": [<image_name>.jpg, ..., <image_name>.jpg]
        #label_index[padded_label] = [os.path.basename(image) for image in images]
        label_index[padded_label] = []
        # idx is the index of the image in the class adjacency matrix
        for image_idx, image in enumerate(images):
            # "<image_name>.jpg": [label, image_idx]
            image_index[os.path.basename(image)] = (padded_label, image_idx)
            # "label": [idx]
            label_index[padded_label].append(label_idx)
            label_idx += 1

    with open(os.path.join(output, "image_index.json"), 'w') as f:
        json.dump(image_index, f, cls=NumpyEncoder)
    
    with open(os.path.join(output, "label_index.json"), 'w') as f:
        json.dump(label_index, f, cls=NumpyEncoder)

    total_iters = len(dataset)

    with tqdm(total=total_iters) as pbar:
        for label, images in grouped:
            padded_label = pad_label(label, cfg.DATASET.NAME)
            filename = os.path.join(output, f"{padded_label}.json")

            # only compute if the file does not exist, allowing to continue from previous stopping point
            if not os.path.isfile(filename):
                pbar.set_description(f"Processing {cfg.DATASET.NAME} gms matches for class {label:4} with {len(images):2} images")
                adj_matrix = process_class(images.to_numpy(), cfg.INPUT.WIDTH, cfg.INPUT.HEIGHT, orb, bf, pbar)

                with open(filename, 'w') as f:
                    json.dump(adj_matrix, f, cls=NumpyEncoder)
            else:
                pbar.update(len(images))

    logger.info(f"Processing complete. Outputs written to {output}")


def process_class(image_paths: np.ndarray, width: int, height: int, orb: cv.ORB, bf: cv.BFMatcher, pbar: tqdm) -> np.ndarray:
    """
    Calculate the adjacency matrix of a given class of images.

    Parameters
    ----------
    image_paths : np.ndarray
        array of paths to every image in the class
    width : int
        width of the images
    height : int
        height of the images
    orb : cv.ORB
        opencv ORB object for feature detection and description
    bf : cv.BFMatcher
        opencv brute force matcher for use in gms
    pbar : tqdm
        tqdm progress bar for manual updates managed in this function

    Returns
    -------
    adj_matrix : np.ndarray
        adjacency matrix of gms matches for the given class.
    """
    n = len(image_paths)
    adj_matrix = np.zeros((n, n), dtype=np.int32)

    # optimisation, compute keypoints and descriptors outside gms_matches function,
    # as otherwise kp1 and des1 are recomputed j times for each i
    for i in range(n):
        img1 = cv.imread(image_paths[i], cv.IMREAD_GRAYSCALE)
        img1 = cv.resize(img1, (width, height))

        kp1, des1 = orb.detectAndCompute(img1, None)

        for j in range(i+1, n):
            img2 = cv.imread(image_paths[j], cv.IMREAD_GRAYSCALE)
            img2 = cv.resize(img2, (width, height))

            kp2, des2 = orb.detectAndCompute(img2, None)
                
            n_matches = gms_matches(bf, img1.shape[:2], img2.shape[:2], kp1, des1, kp2, des2)

            # symmeterical, so update both at once
            adj_matrix[i, j] = n_matches
            adj_matrix[j, i] = n_matches

        pbar.update(1)
    
    return adj_matrix


def gms_matches(
    bf: cv.BFMatcher,
    size1: np.ndarray, size2: np.ndarray,
    kp1: np.ndarray, des1: np.ndarray,
    kp2: np.ndarray, des2: np.ndarray,
) -> int:
    """
    Computes the number of GMS matches between two images.

    Takes the keypoints and descriptors instead of the images to reduce repetition of expensive operations.

    Parameters
    ----------
    bf : cv.BFMatcher
        OpenCV brute force matcher.
    size1 : np.ndarray
        2x2 array containing the width and height of image 1.
    size2 : np.ndarray
        2x2 array containing the width and height of image 2.
    kp1: np.ndarray
        ORB keypoints from image 1.
    des1: np.ndarray
        ORB descriptor for image 1.
    kp2: np.ndarray
        ORB keypoints from image 2.
    des2: np.ndarray
        ORB descriptor for image 2.

    Returns
    -------
    gms_matches : int
        number of gms matches calculated from the images
    """
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        logger.warning("Empty gms descriptors, skipping...")
        return 0

    if des1.shape[1] != des2.shape[1]:
        error = "The GMS descriptors are of different sizes"
        logger.error(error)
        raise ValueError(error)

    if(des1.dtype != [np.uint8, np.float32]) or (des1.dtype != [np.uint8, np.float32]):
        des1 = des1.astype(np.uint8)
            
    if(des2.dtype != [np.uint8, np.float32]) or (des2.dtype != [np.uint8, np.float32]):
        des2 = des2.astype(np.uint8)

    matches_bf = bf.match(des1, des2)
    matches_gms = cv.xfeatures2d.matchGMS(size1=size1, size2=size2,
                                          keypoints1=kp1, keypoints2=kp2,
                                          matches1to2=matches_bf,
                                          withScale=False, withRotation=True)

    return len(matches_gms)


def load_data(gms_path: str) -> dict[str, list[list[str]]]:
    """
    Load the gms data for use in training.

    Parameters
    ----------
    gms_path : str
        path of the directory storing the adjacency matrices of the dataset.

    Returns
    -------
    gms : dict[str, list[list[str]]]
        dict of classes and their adjacency matrices.
    """
    gms = {}
    entries = sorted(os.listdir(gms_path))

    for name in entries:
        if name != "index.json":
            with open(os.path.join(gms_path, name), 'r') as f:
                s = os.path.splitext(name)[0]
                gms[s] = np.array(json.load(f))

    return gms

