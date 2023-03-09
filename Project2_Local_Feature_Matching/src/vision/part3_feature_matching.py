#!/usr/bin/python3

import numpy as np

from typing import Tuple


def compute_feature_distances(
    features1: np.ndarray,
    features2: np.ndarray
) -> np.ndarray:
    """
    This function computes a list of distances from every feature in one array
    to every feature in another.

    Using Numpy broadcasting is required to keep memory requirements low.

    Note: Using a double for-loop is going to be too slow. One for-loop is the
    maximum possible. Vectorization is needed.
    See numpy broadcasting details here:
        https://cs231n.github.io/python-numpy-tutorial/#broadcasting

    Args:
        features1: A numpy array of shape (n1,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
        features2: A numpy array of shape (n2,feat_dim) representing a second
            set of features (n1 not necessarily equal to n2)

    Returns:
        dists: A numpy array of shape (n1,n2) which holds the distances (in
            feature space) from each feature in features1 to each feature in
            features2
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    
    # print(features1.shape)
    # print(features2.shape)
    n1 = features1.shape[0]
    n2 = features2.shape[0]

    dists = np.zeros((n1, n2), dtype = float)
    for i in range(0, n1):
        for j in range(0, n2):
            dists[i, j] = np.linalg.norm(features1[i,:] - features2[j, :])

    # raise NotImplementedError('`compute_feature_distances` function in ' +
    #     '`part3_feature_matching.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dists


def match_features_ratio_test(
    features1: np.ndarray,
    features2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """ Nearest-neighbor distance ratio feature matching.

    This function does not need to be symmetric (e.g. it can produce different
    numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 7.18 in section
    7.1.3 of Szeliski. There are a lot of repetitive features in these images,
    and all of their descriptors will look similar. The ratio test helps us
    resolve this issue (also see Figure 11 of David Lowe's IJCV paper).

    You should call `compute_feature_distances()` in this function, and then
    process the output.

    Args:
        features1: A numpy array of shape (n1,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
        features2: A numpy array of shape (n2,feat_dim) representing a second
            set of features (n1 not necessarily equal to n2)

    Returns:
        matches: A numpy array of shape (k,2), where k is the number of matches.
            The first column is an index in features1, and the second column is
            an index in features2
        confidences: A numpy array of shape (k,) with the real valued confidence
            for every match

    'matches' and 'confidences' can be empty, e.g., (0x2) and (0x1)
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    #initialize the return variables
    matches = []
    confidences = []
    dists = compute_feature_distances(features1, features2)
    # print(dists)
    #print(dists.shape)
    threshold = 0.81 #from paper about nndr

    for i in range(features1.shape[0]):
        sorted = np.sort(dists[i, :])
        d1 = sorted[0]
        d2 = sorted[1]
        ratio = d1/d2
        # print(ratio)

        if ratio < threshold:
            j = np.argmin(dists[i,:])
            matches.append([i,j])
            confidences.append(ratio)
    matches = np.asarray(matches)
    confidences = np.asarray(confidences)

    if len(matches) == 0:
        matches = np.array([[], []]).reshape(0,2)
        confidences = np.array([]).reshape(0,1)





    # raise NotImplementedError('`match_features_ratio_test` function in ' +
    #     '`part3_feature_matching.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return matches, confidences
