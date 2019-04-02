import numpy as np
from scipy.io import loadmat

from utils.utils import convert_label_10_to_0

###############################################################################

def load_svhn():

    '''
    load svhn dataset
    
    input: N/A
      
    output:
      svhn_train_im = training images; (73257 x 3 x 32 x 32)
      svhn_label = {0...9}-valued training labels; 73257-dim
      svhn_test_im = test images; (26032 x 3 x 32 x 32)
      svhn_label_test = {0...9}-valued test labels; 26032-dim    
    '''
    
    svhn_train = loadmat('data/svhn/train_32x32.mat')
    svhn_test = loadmat('data/svhn/test_32x32.mat')
    svhn_train_im = svhn_train['X']
    svhn_train_im = svhn_train_im.transpose(3, 2, 0, 1).astype(np.float32)
    svhn_label = convert_label_10_to_0(svhn_train['y'])
    svhn_test_im = svhn_test['X']
    svhn_test_im = svhn_test_im.transpose(3, 2, 0, 1).astype(np.float32)
    svhn_label_test = convert_label_10_to_0(svhn_test['y'])

    return svhn_train_im, svhn_label, svhn_test_im, svhn_label_test
