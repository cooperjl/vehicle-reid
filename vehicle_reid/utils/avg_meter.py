class AverageMeter(object):
    """Computes and stores the average and current value

    Credit: https://github.com/pytorch/examples/blob/5dfeb46902baf444010f2f54bcf4dfbea109ae4d/imagenet/main.py#L423-L441
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

