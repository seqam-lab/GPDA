import torch
import torch.nn as nn
import torch.nn.functional as F

#from model.grad_reverse import grad_reverse

###############################################################################

#
# PhiGnetwork retuns u = phi(G(x)) where
#
#   x = image
#   z = G(x) = exactly Feature() in MCD-DA
#   u = phi(z) = the last hidden layer of Predictor() in MCD-DA
#

class PhiGnetwork(nn.Module):
    
    def __init__(self):
        
        super(PhiGnetwork, self).__init__()
        
        self.g_conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.g_bn1 = nn.BatchNorm2d(64)
        self.g_conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.g_bn2 = nn.BatchNorm2d(64)
        self.g_conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.g_bn3 = nn.BatchNorm2d(128)
        self.g_fc1 = nn.Linear(8192, 3072)
        self.g_bn1_fc = nn.BatchNorm1d(3072)
        
        #self.phi_fc1 = nn.Linear(8192, 3072)
        #self.phi_bn1_fc = nn.BatchNorm1d(3072)
        self.phi_fc2 = nn.Linear(3072, 2048)
        self.phi_bn2_fc = nn.BatchNorm1d(2048)
        
        self.p = 2048

    def forward(self, x):
        
        x = F.max_pool2d( F.relu(self.g_bn1(self.g_conv1(x))), 
          stride=2, kernel_size=3, padding=1 )
        x = F.max_pool2d( F.relu(self.g_bn2(self.g_conv2(x))), 
          stride=2, kernel_size=3, padding=1 )
        x = F.relu(self.g_bn3(self.g_conv3(x)))
        x = x.view(x.size(0), 8192)
        x = F.relu(self.g_bn1_fc(self.g_fc1(x)))
        z = F.dropout(x, training=self.training)
        
        u = F.relu(self.phi_bn2_fc(self.phi_fc2(z)))
        
        return u
    
###############################################################################

#
# QWnetwork retuns w^m_j = mu_j + sd_j.*eps^m_j, for m=1...M samples from 
#   q(w) = \prod_{j=1}^K N(w_j; mu_j, diag(sd_j)^2) with dim(w_j) = p
#
#   eps = samples from N(0,1); (M x p x K) -- input
#   mu = K mean vectors of q(w); (p x K) -- model params
#   logsd = K log-stdev vectors of q(w); (p x K) -- model params
#     (sd = exp(logsd)) 
#

class QWnetwork(nn.Module):
    
    def __init__(self):
        
        super(QWnetwork, self).__init__()
        
        self.mu = nn.Parameter(0.01*torch.randn(2048, 10))
        self.logsd = nn.Parameter(0.01*torch.randn(2048, 10))

    def forward(self, eps):
        
        mu3 = self.mu.unsqueeze(0)  # (1 x p x K)
        sd3 = torch.exp(self.logsd).unsqueeze(0)  # (1 x p x K)
        w = mu3 + sd3*eps  # (M x p x K) 
        
        return w

###############################################################################

#class Feature(nn.Module):
#    
#    def __init__(self):
#        
#        super(Feature, self).__init__()
#        
#        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
#        self.bn1 = nn.BatchNorm2d(64)
#        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
#        self.bn2 = nn.BatchNorm2d(64)
#        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
#        self.bn3 = nn.BatchNorm2d(128)
#        self.fc1 = nn.Linear(8192, 3072)
#        self.bn1_fc = nn.BatchNorm1d(3072)
#
#    def forward(self, x):
#        
#        x = F.max_pool2d( F.relu(self.bn1(self.conv1(x))), 
#          stride=2, kernel_size=3, padding=1 )
#        x = F.max_pool2d( F.relu(self.bn2(self.conv2(x))), 
#          stride=2, kernel_size=3, padding=1 )
#        x = F.relu(self.bn3(self.conv3(x)))
#        x = x.view(x.size(0), 8192)
#        x = F.relu(self.bn1_fc(self.fc1(x)))
#        x = F.dropout(x, training=self.training)
#        
#        return x

###############################################################################

#class Predictor(nn.Module):
#    
#    def __init__(self, prob=0.5):
#        
#        super(Predictor, self).__init__()
#        
#        self.fc1 = nn.Linear(8192, 3072)
#        self.bn1_fc = nn.BatchNorm1d(3072)
#        self.fc2 = nn.Linear(3072, 2048)
#        self.bn2_fc = nn.BatchNorm1d(2048)
#        self.fc3 = nn.Linear(2048, 10)
#        self.bn_fc3 = nn.BatchNorm1d(10)
#        self.prob = prob
#
#    def set_lambda(self, lambd):
#        
#        self.lambd = lambd
#
#    def forward(self, x, reverse=False):
#        
#        if reverse:
#            x = grad_reverse(x, self.lambd)
#        x = F.relu(self.bn2_fc(self.fc2(x)))
#        x = self.fc3(x)
#        
#        return x
