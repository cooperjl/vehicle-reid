import argparse
import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import vehicle_reid.args as args
from vehicle_reid.datasets import VRIC

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


def gms_matches(orb: cv.ORB, bf: cv.BFMatcher, img1: NDArray[np.uint8], img2: NDArray[np.uint8]):
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        print("e: empty descriptors")
        raise ValueError

    if des1.shape[1] != des2.shape[1]:
        print("e: different sizes")
        raise ValueError

    if(des1.dtype != [np.uint8, np.float32]) or (des1.dtype != [np.uint8, np.float32]):
            des1 = des1.astype(np.uint8)
            
    if(des2.dtype != [np.uint8, np.float32]) or (des2.dtype != [np.uint8, np.float32]):
        des2 = des2.astype(np.uint8)

    matches_bf = bf.match(des1, des2)
    matches_gms = cv.xfeatures2d.matchGMS(size1=img1.shape[:2], size2=img2.shape[:2],
                                          keypoints1=kp1, keypoints2=kp2,
                                          matches1to2=matches_bf,
                                          withScale=False, withRotation=True)

    match_img = cv.drawMatches(img1, kp1, img2, kp2, matches_gms, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return match_img

    return len(matches_gms)


def main(args: argparse.Namespace):
    # defaults are per gms paper, may change to 224x224 later to fit with deep learning models and RPTM

    match args.dataset:
        case 'vric':
            dataset = VRIC(root='data/vric/', split='train')
        case _:
            raise ValueError

    orb = cv.ORB_create(nfeatures=10000, fastThreshold=0)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

    #dataset.get_dict()

    for i in range(4):
        ax = plt.subplot(2, 2, i+1)
        plt.tight_layout()
        ax.axis('off')
        matches = dataset.get_random_label()
        img1name = matches.to_numpy()[0][-1]
        img2name = matches.to_numpy()[1][-1]

        print(img1name)
    
        img1 = cv.imread(os.path.join('data/vric/train_images/', img1name))
        img2 = cv.imread(os.path.join('data/vric/train_images/', img2name))
        img1 = cv.resize(img1, (args.c.width, args.c.height))
        img2 = cv.resize(img2, (args.c.width, args.c.height))

        img3 = gms_matches(orb, bf, img1, img2)

        #sample_idx = torch.randint(len(vric_dataset), size=(1,)).item()
        #img, label = vric_dataset[sample_idx]
        #plt.title(label)
        plt.imshow(img3)

    plt.show()

