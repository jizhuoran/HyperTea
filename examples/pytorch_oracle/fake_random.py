import numpy as np

class FakeRandomGenerator(object):
    """docstring for FakeRandomGenerator"""
    def __init__(self):
        super(FakeRandomGenerator, self).__init__()
        with open('random_number.txt', 'r') as f:
            self.source_vec = [float(n) for n in f.read().split(' ')]
        self.pos = 0


    def generator_random_list(self, item_numbers):
        l = []
        for _ in range(item_numbers):
            l.append(self.source_vec[self.pos])
            self.pos = (self.pos + 1) % len(self.source_vec)
        return l

    def rn(self, item_shape):
        return np.array(self.generator_random_list(np.prod(item_shape))).reshape(item_shape)