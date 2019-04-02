import model.svhn2mnist as svhn2mnist
#import model.usps as usps
#import model.syn2gtrsb as syn2gtrsb

###############################################################################

def PhiGnet(source, target):
    if source == 'usps' or target == 'usps':
        return usps.PhiGnetwork()
    elif source == 'svhn':
        return svhn2mnist.PhiGnetwork()
    elif source == 'synth':
        return syn2gtrsb.PhiGnetwork()

###############################################################################

def QWnet(source, target):
    if source == 'usps' or target == 'usps':
        return usps.QWnetwork()
    elif source == 'svhn':
        return svhn2mnist.QWnetwork()
    elif source == 'synth':
        return syn2gtrsb.QWnetwork()

###############################################################################

#def Generator(source, target):
#    if source == 'usps' or target == 'usps':
#        return usps.Feature()
#    elif source == 'svhn':
#        return svhn2mnist.Feature()
#    elif source == 'synth':
#        return syn2gtrsb.Feature()

###############################################################################
        
#def Classifier(source, target):
#    if source == 'usps' or target == 'usps':
#        return usps.Predictor()
#    if source == 'svhn':
#        return svhn2mnist.Predictor()
#    if source == 'synth':
#        return syn2gtrsb.Predictor()
