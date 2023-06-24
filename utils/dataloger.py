from torch.utils.tensorboard import SummaryWriter


class DataLoger():
    def __init__(self, dir):
        self.writer = SummaryWriter(dir)

    def log(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
