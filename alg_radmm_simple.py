
import alg_radmm_base
import numpy as np

class AlgRadMMSimple(alg_radmm_base.AlgRadMMBase):
    def __init__(self, base_path):
        alg_radmm_base.AlgRadMMBase.__init__(self, base_path)
    def get_train_x(self, ids):
        pass
    def get_test_x(self, ids):
        pass
    def train(self, x, y, ids):
        pass
    def predict(self, x, ids):
        ret = np.zeros((len(ids), 2))
        return ret
