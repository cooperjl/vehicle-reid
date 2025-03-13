import unittest
from argparse import Namespace

import torch

import vehicle_reid.args
import vehicle_reid.eval
import vehicle_reid.model
from vehicle_reid.datasets import VeRi


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TestEval(unittest.TestCase):
    def test_eval_model(self):
        args = Namespace()

        args.width=224
        args.height=224
        args.data_path="data"
        args.gms_path="gms"
        args.batch_size = 16
        
        args.dataset = "veri"
        
        model = vehicle_reid.model.cresnet50(VeRi.num_classes, pretrain=False)
        model = model.to(device)
        
        try:
            vehicle_reid.eval.eval_model(model)
        except Exception as e:
            self.fail(f"eval function faced an error: {e}")


if __name__ == '__main__':
    unittest.main()
