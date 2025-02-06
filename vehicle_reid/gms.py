import argparse
import json
import os

import cv2 as cv
import numpy as np
from tqdm import tqdm

import vehicle_reid.args as args
from vehicle_reid.datasets import VRIC
from vehicle_reid.utils import NumpyEncoder


def parse_arguments():
    parser = args.add_subparser(name="gms", help="compute GMS matches")
    parser.add_argument('dataset', metavar='dataset',
                        choices=["vric", "vehicleid", "veri"],
                        help='the name of the dataset to compute')
    parser.add_argument('--width', type=int,
                        default=args.DEFAULTS.width,
                        help='width to resize images to (default: %(default)s)')
    parser.add_argument('--height', type=int,
                        default=args.DEFAULTS.height,
                        help='width to resize images to (default: %(default)s)')
    parser.add_argument('--data-path', type=args.dir_path,
                        default=args.DEFAULTS.data_path,
                        help='path where the datasets are stored (default: %(default)s)')
    parser.add_argument('--output-path', type=args.dir_path,
                        default='gms',
                        help='path to output gms matches (default: %(default)s)')
    parser.set_defaults(func=main)


def gms_matches(bf: cv.BFMatcher, size1: np.ndarray, size2: np.ndarray, kp1: tuple, des1: tuple, kp2: tuple, des2: tuple):
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        # print("WARN: The GMS descriptors are empty.") 
        return 0

    if des1.shape[1] != des2.shape[1]:
        raise ValueError("ERROR: The GMS descriptors are of different sizes.")

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


def process_class(image_paths: np.ndarray, width: int, height: int, orb: cv.ORB, bf: cv.BFMatcher, pbar: tqdm) -> np.ndarray:
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


def main(args: argparse.Namespace):
    match args.dataset:
        case "vric":
            root = os.path.join(args.data_path, "vric")
            dataset = VRIC(root=root, split="train") # train hardcoded as gms matches should only ever use the train set.
        case _:
            raise ValueError("TODO: other datasets")

    output = os.path.join(args.output_path, args.dataset)
    if(not os.path.exists(output)):
        os.mkdir(output)
        print(f"INFO: output directory created at: {output}")

    orb = cv.ORB_create(nfeatures=10000, fastThreshold=0)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    
    total_iters = len(dataset)

    with tqdm(total=total_iters) as pbar:
        for label, images in dataset.get_grouped():
            filename = os.path.join(output, f"{label}.json")

            # only compute if the file does not exist, allowing to continue from previous stopping point
            if not os.path.isfile(filename):
                pbar.set_description(f"Processing {args.dataset} gms matches for class {label:4} with {len(images):2} images")
                adj_matrix = process_class(images.to_numpy(), args.width, args.height, orb, bf, pbar)

                with open(filename, 'w') as f:
                    json.dump(adj_matrix, f, cls=NumpyEncoder)
            else:
                pbar.update(len(images))

    print(f"Processing complete. Outputs written to {output}")

