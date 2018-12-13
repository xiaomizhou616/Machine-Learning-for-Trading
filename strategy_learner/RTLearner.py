import numpy as np
from scipy import stats
'''
import RTLearner as rt
learner = rt.RTLearner(leaf_size = 1, verbose = False) # constructor
learner.addEvidence(Xtrain, Ytrain) # training step
Y = learner.query(Xtest) # query
'''

class RTLearner(object):

    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.root = None

    def author(self):
        return 'xhan306' # replace tb34 with your Georgia Tech username

    def addEvidence(self, data_x, data_y):
        self.tree = self.build_tree(data_x, data_y)

    def build_tree(self, data_x, data_y):
        def build_leaf(y_arr):
            return np.array([[-1, stats.mode(y_arr, axis=None)[0][0], np.nan, np.nan]])
            # return np.array([[-1, np.mean(y_arr), np.nan, np.nan]])

        if len(np.unique(data_y)) == 1 or data_x.shape[0] <= self.leaf_size:
            return build_leaf(data_y)

        i = np.random.randint(0, data_x.shape[1])

        rand_samples = np.random.choice(data_x[:, i], 2, replace=False)

        split_val = np.mean(rand_samples)

        is_left = data_x[:, i] <= split_val

        uni = np.unique(is_left)
        if len(uni) == 1:
            return build_leaf(data_y[is_left == uni[0]])

        left_tree = self.build_tree(data_x[is_left], data_y[is_left])
        right_tree = self.build_tree(data_x[is_left != True], data_y[is_left != True])

        root = np.array([[i, split_val, 1, len(left_tree) + 1]])
        tree = np.concatenate((root, left_tree, right_tree), axis=0)
        return tree

    def query(self, points):
        if self.verbose:
            pass
        return np.array([self.predict(point, 0) for point in points])

    def predict(self, point, row):
        if int(self.tree[int(row), 0]) == -1:
            return self.tree[int(row), 1]
        if point[int(self.tree[int(row), 0])] <= self.tree[int(row), 1]:
            return self.predict(point, row + self.tree[int(row), 2])
        else:
            return self.predict(point, row + self.tree[int(row), 3])

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
