import sys

from data_manager.unaligned_data_loader import UnalignedDataLoader
from data_manager.svhn import load_svhn
from data_manager.mnist import load_mnist
#from datasets.usps import load_usps
#from datasets.gtsrb import load_gtsrb
#from datasets.synth_traffic import load_syntraffic

###############################################################################

def return_dataset(data, scale=False, usps=False, all_use='no'):
    
    '''
    load a specified dataset
    
    input:
      data = dataset to load (eg, 'svhn', 'mnist'); string
      scale = whether to scale up images to (32 x 32) or not (28 x 28)
      usps, all_use = whether of not to take subsamples from traning set
      
    output:
      train_image = train images; (ntr x C x H x W) 
      train_label = {0...9}-valued train labels; ntr-dim
      test_image = test images; (nte x C x H x W)
      test_label = {0...9}-valued test labels; nte-dim
    '''
    
    if data == 'svhn':
        train_image, train_label, test_image, test_label = load_svhn()
    
    if data == 'mnist':
        train_image, train_label, test_image, test_label = \
          load_mnist( scale=scale, usps=usps, all_use=all_use )
        sys.stdout.write('mnist image shape = ');  print(train_image.shape)
    
#    if data == 'usps':
#        train_image, train_label, test_image, test_label = \
#          load_usps(all_use=all_use)
#    
#    if data == 'synth':
#        train_image, train_label, test_image, test_label = \
#          load_syntraffic()
#    
#    if data == 'gtsrb':
#        train_image, train_label, test_image, test_label = load_gtsrb()

    return train_image, train_label, test_image, test_label

###############################################################################
    
def dataset_read( source, target, batch_size, scale=False, all_use='no' ):

    if source == 'usps' or target == 'usps':
        usps = True
    else:
        usps = False
    
    S = {};  S_test = {}
    T = {};  T_test = {}

    # read source data
    train_source, s_label_train, test_source, s_label_test = \
      return_dataset( source, scale=scale, usps=usps, all_use=all_use )
      
    # read target data
    train_target, t_label_train, test_target, t_label_test = \
      return_dataset( target, scale=scale, usps=usps, all_use=all_use )

    # prepare source/target data
    S['imgs'] = train_source
    S['labels'] = s_label_train
    T['imgs'] = train_target
    T['labels'] = t_label_train

    # test samples for source/target 
    S_test['imgs'] = test_target
    S_test['labels'] = t_label_test
    T_test['imgs'] = test_target
    T_test['labels'] = t_label_test
    
    scale = 40 if source == 'synth' else 28 if usps else 32

    # (train) do some image transform and create a minibatch generator
    train_loader = UnalignedDataLoader()
    train_loader.initialize(S, T, batch_size, batch_size, scale=scale)
    dataset = train_loader.load_data()
    
    # (test) do some image transform and create a minibatch generator
    test_loader = UnalignedDataLoader()
    test_loader.initialize(S_test, T_test, batch_size, batch_size, scale=scale)
    dataset_test = test_loader.load_data()
    
    return dataset, dataset_test
