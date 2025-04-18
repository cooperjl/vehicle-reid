from yacs.config import CfgNode as CN

_C = CN()

# ---------------------------
#           Model
# ---------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.ARCH = "resnet50"
_C.MODEL.TWO_BRANCH = True
_C.MODEL.CHECKPOINT = None

# ---------------------------
#           Input
# ---------------------------
_C.INPUT = CN()
_C.INPUT.HEIGHT = 128
_C.INPUT.WIDTH = 128
_C.INPUT.AUGMENT = True
_C.INPUT.FLIP_PROB = 0.5
_C.INPUT.RAND_ERASE = False
_C.INPUT.MEAN = [0.485, 0.456, 0.406]
_C.INPUT.STD = [0.229, 0.224, 0.225]

# ---------------------------
#          Dataset
# ---------------------------
_C.DATASET = CN()
_C.DATASET.PATH = "data"
_C.DATASET.NAME = "veri"

# ---------------------------
#        Data Loader
# ---------------------------
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 8

# ---------------------------
#          Solver
# ---------------------------
_C.SOLVER = CN()
_C.SOLVER.TRIPLET_SELECT = "mean"
_C.SOLVER.OPTIMIZER = "adam"
_C.SOLVER.EPOCHS = 50
_C.SOLVER.BASE_LR = 0.005
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.DECAY_BN_BIAS = True
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.BATCH_SIZE = 8
_C.SOLVER.MILESTONES = [20, 40, 60]
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.AMP = False
_C.SOLVER.CHECKPOINT_PERIOD = 10

# ---------------------------
#           Loss
# ---------------------------
_C.LOSS = CN()
_C.LOSS.MARGIN = 1.0
_C.LOSS.LAMBDA_CE = 1.0
_C.LOSS.LAMBDA_TRI = 1.0

# ---------------------------
#           Test
# ---------------------------
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 8


# ---------------------------
#       Miscellaneous
# ---------------------------
_C.MISC = CN()
_C.MISC.GMS_PATH = "gms"
_C.MISC.LOG_DIR = "logs"
_C.MISC.SAVE_DIR = "checkpoints"
_C.MISC.CACHE_PATH = "cache"
_C.MISC.SAVE_FREQ = 10
_C.MISC.SEED = None
