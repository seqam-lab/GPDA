import numpy as np
from scipy.io import loadmat

###############################################################################

def load_mnist(scale=True, usps=False, all_use='no'):
    
    '''
    load mnist dataset
    
    input:
      scale = whether to scale up images to (32 x 32) or not (28 x 28)
        if scale==True, also duplicate channels to (32 x 32 x 3)
      usps, all_use = whether of not to take subsamples from traning set 
        use 2000 random subsamples from training if usps==True & all_use='no'
        use ALL training samples otherwise 
        
    output:
      mnist_train = training images; 
        (55000 x 3 x 32 x 32) or (55000 x 1 x 28 x 28)
      train_label = {0...9}-valued training labels; 55000-dim
      mnist_test = test images;
        (10000 x 3 x 32 x 32) or (10000 x 1 x 28 x 28)
      test_label = = {0...9}-valued training labels; 10000-dim
    '''
    
    mnist_data = loadmat('data/mnist/mnist_data.mat')
      # load the following dict composed of:
      #   mnist_data['train_32', 'test_32'] = (n x 32 x 32)
      #   mnist_data['train_28', 'test_28'] = (n x 28 x 28 x 1)
      #   mnist_data['label_train', 'label_test'] = (n x 10) one-hot
    
    if scale:  # scale up and channel-duplicate images to (32 x 32 x 3)
        
        mnist_train = np.reshape(mnist_data['train_32'], (55000, 32, 32, 1))
        mnist_test = np.reshape(mnist_data['test_32'], (10000, 32, 32, 1))
        
        # duplicate channels
        mnist_train = np.concatenate(
          [mnist_train, mnist_train, mnist_train], 3 )
        mnist_test = np.concatenate(
          [mnist_test, mnist_test, mnist_test], 3 )

        # reshape to (n x C x H x W) format
        mnist_train = mnist_train.transpose(0, 3, 1, 2).astype(np.float32)
        mnist_test = mnist_test.transpose(0, 3, 1, 2).astype(np.float32)
        
    else:  # use original (28 x 28 x 1)
        
        mnist_train = mnist_data['train_28']
        mnist_test =  mnist_data['test_28']

        # reshape to (n x C x H x W) format
        mnist_train = mnist_train.transpose((0, 3, 1, 2)).astype(np.float32)
        mnist_test = mnist_test.transpose((0, 3, 1, 2)).astype(np.float32)
    
    # labels in one-hot format
    mnist_labels_train = mnist_data['label_train']
    mnist_labels_test = mnist_data['label_test']
        
    # convert one-hot to 0~9 labels
    train_label = np.argmax(mnist_labels_train, axis=1)
    test_label = np.argmax(mnist_labels_test, axis=1)
    
    # randomly shuffle training data
    inds = np.random.permutation(mnist_train.shape[0])
    mnist_train = mnist_train[inds]
    train_label = train_label[inds]
    
    # subsample training images
    if usps and all_use != 'yes':
        mnist_train = mnist_train[:2000]
        train_label = train_label[:2000]

    return mnist_train, train_label, mnist_test, test_label
