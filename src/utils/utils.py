import numpy as np

###############################################################################

def weights_init(m):

    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)

###############################################################################

def convert_label_10_to_0(labels):

    '''
    convert class label 10 to 0
    '''
    
    labels2 = np.zeros((len(labels),))
    labels = list(labels)
    for i, t in enumerate(labels):
        if t == 10:
            labels2[i] = 0
        else:
            labels2[i] = t
            
    return labels2
