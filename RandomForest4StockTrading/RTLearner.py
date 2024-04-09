""""""  		  	   		  		 			  		 			     			  	 
"""  		  	   		  		 			  		 			     			  	 
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		  		 			  		 			     			  	 
All Rights Reserved  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Template code for CS 4646/7646  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		  		 			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		  		 			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		  		 			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 			  		 			     			  	 
or edited.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		  		 			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		  		 			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 			  		 			     			  	 
GT honor code violation.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
-----do not edit anything above this line---  		  	   		  		 			  		 			     			  	 
"""  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
import numpy as np  		  	   		  		 			  		 			     			  	 
import random
from scipy import stats
  		  	   		  		 			  		 			     			  	 
class RTLearner(object):
    """  		  	   		  		 			  		 			     			  	 
    This is a RT Learner. It is implemented correctly.
  		  	   		  		 			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  		 			  		 			     			  	 
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		  		 			  		 			     			  	 
    :type verbose: bool  		  	   		  		 			  		 			     			  	 
    """  		  	   		  		 			  		 			     			  	 
    def __init__(self, leaf_size = 1, verbose=False):
        """  		  	   		  		 			  		 			     			  	 
        Constructor method  		  	   		  		 			  		 			     			  	 
        """
        self.leaf_size = leaf_size
        self.verbose = verbose
  		  	   		  		 			  		 			     			  	 
    def author(self):  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        :return: The GT username of the student  		  	   		  		 			  		 			     			  	 
        :rtype: str  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        return "ycheng345"  # replace tb34 with your Georgia Tech username
  		  	   		  		 			  		 			     			  	 
    def add_evidence(self, data_x, data_y):  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        Add training data to learner  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
        :param data_x: A set of feature values used to train the learner  		  	   		  		 			  		 			     			  	 
        :type data_x: numpy.ndarray  		  	   		  		 			  		 			     			  	 
        :param data_y: The value we are attempting to predict given the X data  		  	   		  		 			  		 			     			  	 
        :type data_y: numpy.ndarray  		  	   		  		 			  		 			     			  	 
        """

        self.tree = self.build_tree(data_x, data_y)

    def build_tree(self, data_x, data_y):
        # Base case for the recursion, return leaf node if there are only leafs or all the values in y are the same
        if data_x.shape[0] <= self.leaf_size or len(np.unique(data_y)) == 1:
            return np.array([-1, stats.mode(data_y)[0][0], np.nan, np.nan]) # change the mean to mode

        # Get the best feature and split value from the helper function
        bf = self.get_best_feature(data_x, data_y)
        sv = self.get_split_val(data_x, bf)
        # Partition the data based on the split value
        belongs_to_left = data_x[:, bf] <= sv

        # Another base case, if all samples end up in the left branch, return the leaf node
        if np.all(belongs_to_left):
            return np.array([-1, stats.mode(data_y)[0][0], np.nan, np.nan]) # change the mean to mode

        # Recursively build the left and right branches
        left_tree = self.build_tree(data_x[belongs_to_left], data_y[belongs_to_left])
        right_tree = self.build_tree(data_x[~belongs_to_left], data_y[~belongs_to_left])

        # Determine the starting index of the right branch
        if left_tree.ndim <= 1:
            root = np.array([bf, sv, 1, 2])
        else:
            root = np.array([bf, sv, 1, left_tree.shape[0] + 1])

        # Assemble tree, with the root node, left subtree and right subtree
        tree = np.vstack((root, left_tree, right_tree))
        return tree

    def get_best_feature(self, data_x, data_y):
        # Randomly choose a col as the best feature col and the rest should be same
        random_col = random.randint(0, data_x.shape[1] - 1)
        return random_col

    def get_split_val(self, data_x, bf_col):
        split_value = np.median(data_x[:, bf_col])
        return split_value

    def query(self, points):  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        Estimate a set of test points given the model we built.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		  		 			  		 			     			  	 
        :type points: numpy.ndarray  		  	   		  		 			  		 			     			  	 
        :return: The predicted result of the input data according to the trained model  		  	   		  		 			  		 			     			  	 
        :rtype: numpy.ndarray  		  	   		  		 			  		 			     			  	 
        """
        tree = self.tree

        pred_y = np.empty((points.shape[0],), dtype=np.float32)
        for i, point in enumerate(points):
            treenode = 0
            while int(tree[treenode, 0]) != -1:
                feature = int(tree[treenode, 0])

                if point[feature] <= tree[treenode, 1]:
                    treenode = treenode + int(tree[treenode, 2])
                else:
                    treenode = treenode + int(tree[treenode, 3])

            pred_y[i] = tree[treenode, 1]
        return pred_y
