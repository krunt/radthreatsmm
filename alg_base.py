
class AlgBase(object):
    def get_train_ids(self):
        pass
    def get_test_ids(self):
        pass
    def get_train_x(self, ids, validation):
        pass
    def get_train_y(self, ids):
        pass
    def get_test_x(self, ids):
        pass
    def train(self, x, y, ids):
        pass
    def predict(self, x, ids):
        pass
    def score(self, pred_y, ids):
        pass
    def write_submission(self, pred_y, ids):
        pass
