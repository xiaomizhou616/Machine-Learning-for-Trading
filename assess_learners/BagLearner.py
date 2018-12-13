import numpy as np
import DTLearner as dt

class BagLearner(object):

    def __init__(self, learner=dt.DTLearner, kwargs={"leaf_size":1, "verbose":False}, bags=20, boost=False, verbose=False):
        self.learners = [learner(**kwargs) for _ in range(0, bags)]
        self.boost = boost
        self.verbose = verbose

    def author(self):
        return 'xhan306' # replace tb34 with your Georgia Tech username

    def addEvidence(self, data_x, data_y):
        if self.boost:
            if self.verbose:
                pass
        else:
            for learner in self.learners:
                indices = np.random.choice(data_x.shape[0], data_x.shape[0])
                learner.addEvidence(data_x[indices], data_y[indices])

    def query(self, points):
        predicts = [learner.query(points) for learner in self.learners]
        return np.mean(predicts, axis=0)

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
