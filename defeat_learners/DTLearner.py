import numpy as np

'''
import DTLearner as dt
learner = dt.DTLearner(leaf_size = 1, verbose = False) # constructor
learner.addEvidence(Xtrain, Ytrain) # training step
Y = learner.query(Xtest) # query

'''

class DTLearner(object):

    def __init__(self, leaf_size = 1, verbose = False):
        '''
        @param leaf_size
        @param verbose. If True, your code can print out information for debugging. If verbose = False your code should not generate ANY output.
        
        '''
        self.leaf_size = leaf_size
        self.verbose = verbose

    def author(self):
        return 'jchen779' # replace tb34 with your Georgia Tech username

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: nparrays, each row represents an X1, X2, X3...XN
        @param dataY: single dimension ndarrays that indicate the value we are attempting to predict with X.

        """
        # build and save the model
        dataY = np.reshape(dataY, (dataX.shape[0], 1))
        data = np.concatenate((dataX, dataY), axis=1)
        #print("Origin", data)
        self.root = self.build_tree(data)


    def build_tree(self, data):
        if(data.shape[0] <= self.leaf_size):
            return np.array([[-1.0, np.mean(data[:,-1]), np.nan, np.nan],])
        if(len(set(data[:, -1])) == 1):
            return np.array([[-1.0, np.mean(data[:, -1]), np.nan , np.nan],])
        ## find the best feature's column
        i = self.find_bestFeature(data)
        # sort the data according column i
        data = data[data[:,i].argsort()]
        # calculate the median of column i
        SplitVal = np.median(data[:, i])
        left,right = list(), list()
        for r in data:
            if(r[i] <= SplitVal):
                left.append(r)
            else:
                right.append(r)
        left = np.array(left)
        right = np.array(right)
        if(left.shape[0] == 0):
            return np.array([[-1.0, np.mean(right[:, -1]), np.nan, np.nan],])
        if(right.shape[0] == 0):
            return np.array([[-1.0, np.mean(left[:, -1]), np.nan, np.nan],])

        lefttree = self.build_tree(left)
        righttree = self.build_tree(right)
        root = np.array([[i, SplitVal, 1, len(lefttree) + 1],])
        tree = np.concatenate((root, lefttree, righttree), axis=0)
        return tree

    def find_bestFeature(self, data):
        dataX = data[:, 0:-1]
        dataY = data[:, -1]
        corr = list()
        for col in dataX.T:
            corr.append(np.absolute(np.corrcoef(col, y=dataY)[0,1]))
        return corr.index(np.nanmax(corr))

    def query(self, Xtest):
        """
        @summary: Estimate a set of test data given the model we built.
        @param Xtest: Xtest: nparrays, each row represents an X1, X2, X3...XN
        @returns the estimated values according to the saved model.

        """
        predY = list()
        for rows in Xtest:
            predY.append(self.predict(self.root, rows, 0))

        if self.verbose:
            pass

        return np.array(predY)

    def predict(self, root, val, i):
        if(int(root[int(i), 0]) == -1):
            return root[int(i), 1]
        if(val[int(root[int(i), 0])] <= root[int(i), 1]):
            return self.predict(root, val, i + root[int(i), 2])
        if(val[int(root[int(i), 0])] > root[int(i), 1]):
            return self.predict(root, val, i + root[int(i), 3])
        

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"


    
