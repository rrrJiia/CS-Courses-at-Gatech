#!/usr/bin/python3

from telnetlib import X3PAD
import numpy as np


def compute_normalized_patch_descriptors(
    image_bw: np.ndarray, X: np.ndarray, Y: np.ndarray, feature_width: int
) -> np.ndarray:
    """Create local features using normalized patches.

    Normalize image intensities in a local window centered at keypoint to a
    feature vector with unit norm. This local feature is simple to code and
    works OK.

    Choose the top-left option of the 4 possible choices for center of a square
    window.

    Args:
        image_bw: array of shape (M,N) representing grayscale image
        X: array of shape (K,) representing x-coordinate of keypoints
        Y: array of shape (K,) representing y-coordinate of keypoints
        feature_width: size of the square window

    Returns:
        fvs: array of shape (K,D) representing feature descriptors
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    fvs = np.empty((Y.shape[0], feature_width * feature_width))

    for i in range(X.shape[0]):
        #print(Y[i]-(feature_width-1)//2)
        #print((Y[i]+(feature_width//2) +1))
        #print(X[i]-(feature_width-1)//2)
        #print((X[i]+(feature_width//2) +1))
        Y_prev = Y[i]-(feature_width-1)//2
        Y_latter = Y[i]+feature_width//2+1
        X_prev = X[i]-(feature_width-1)//2
        X_latter = X[i]+feature_width//2 +1
        room = image_bw[Y_prev:Y_latter, X_prev:X_latter] #each room within fvs
        if (room.shape != (16, 16)):
            #print(room.shape)
            room = image_bw[X_prev:X_latter, Y_prev:Y_latter]
        
        norm = np.linalg.norm(room)
        room = (1/norm) * room.astype(np.float32).reshape(1,feature_width * feature_width)
        fvs[i] = room

        
    # raise NotImplementedError('`compute_normalized_patch_descriptors` ' +
    #     'function in`part2_patch_descriptor.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return fvs
