import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

from model.build_gen import PhiGnet, QWnet
from data_manager.dataset_read import dataset_read

###############################################################################

class Solver(object):
    
    ########
    def __init__( self, args, batch_size=128, source='svhn', target='mnist', 
      nsamps_q=50, lamb_marg_loss=10.0,
      learning_rate=0.0002, interval=100, optimizer='adam', num_k=4, num_kq=4, 
      all_use=False, checkpoint_dir=None, save_epoch=10 ):
        
        # set hyperparameters
        self.batch_size = batch_size
        self.source = source
        self.target = target
        self.num_k = num_k
        self.num_kq = num_kq
        self.checkpoint_dir = checkpoint_dir
        self.save_epoch = save_epoch
        self.use_abs_diff = args.use_abs_diff
        self.all_use = all_use
        if self.source == 'svhn':
            self.scale = True
        else:
            self.scale = False
        self.lamb_marg_loss = lamb_marg_loss
            
        # load data, do image transform, and create a mini-batch generator 
        print('dataset loading')
        self.datasets, self.dataset_test = \
          dataset_read( source, target, self.batch_size, 
            scale=self.scale, all_use=self.all_use )
        print('load finished!')
        
        if source == 'svhn':
          self.Ns = 73257
        
        # create models
        self.phig = PhiGnet(source=source, target=target)
        self.qw = QWnet(source=source, target=target)
        
        # load the previously learned models from files (if evaluations only)
        if args.eval_only:
            self.phig = torch.load( '%s/model_epoch%s_phig.pt' % 
              (self.checkpoint_dir , args.resume_epoch) )
            self.qw = torch.load( '%s/model_epoch%s_qw.pt' % 
              (self.checkpoint_dir, args.resume_epoch) )
        
        # move models to GPU
        self.phig.cuda()
        self.qw.cuda()
        
        # create optimizer objects (one for each model)
        self.set_optimizer(which_opt=optimizer, lr=learning_rate)
        
        # print stats every interval (default: 100) minibatch iters
        self.interval = interval

        self.lr = learning_rate
        
        # some dimensions
        self.p = self.phig.p  # dim(phi(G(x)))
        self.M = nsamps_q  # number of samples from variational density q(w)
        

    ########
    def set_optimizer(self, which_opt='momentum', lr=0.001, momentum=0.9):
        
        if which_opt == 'momentum':
            
            self.opt_phig = optim.SGD( self.phig.parameters(),
              lr=lr, weight_decay=0.0005, momentum=momentum )

            self.opt_qw = optim.SGD( self.qw.parameters(),
              lr=lr, weight_decay=0.0005, momentum=momentum )
            
        if which_opt == 'adam':
            
            self.opt_phig = optim.Adam( self.phig.parameters(),
              lr=lr, weight_decay=0.0005 )

            self.opt_qw = optim.Adam( self.qw.parameters(),
              lr=lr, weight_decay=0.0005 )
                        
            
    ########
    def reset_grad(self):
        
        # zero out all gradients of model params registered in the optimizers
        self.opt_phig.zero_grad()
        self.opt_qw.zero_grad()
 

    ########
    def ent(self, output):
        
        return -torch.mean(output * torch.log(output + 1e-6))


    ########
    def kl_loss(self):
        
        kl = 0.5 * ( -self.p*10 + 
          torch.sum( (torch.exp(self.qw.logsd))**2 + self.qw.mu**2 - 
            2.0*self.qw.logsd ) 
        ) 
        
        return kl
            
    
    ########
    def train(self, epoch, record_file=None):
        
        '''
        train models for one epoch (ie, one pass of whole training data)
        '''
        
        criterion = nn.CrossEntropyLoss().cuda()
        
        # turn models into "training" mode 
        #   (required if models contain "BatchNorm"-like layers)
        self.phig.train()
        self.qw.train()
        
        torch.cuda.manual_seed(1)

        # for each batch
        for batch_idx, data in enumerate(self.datasets):
            
            img_t = data['T']
            img_s = data['S']
            label_s = data['S_label']
            if img_s.size()[0] < self.batch_size or \
               img_t.size()[0] < self.batch_size:
                break
            img_s = img_s.cuda()
            img_t = img_t.cuda()
            # imgs = Variable(torch.cat((img_s, img_t), 0))
            label_s = Variable(label_s.long().cuda())
            img_s = Variable(img_s)
            img_t = Variable(img_t)
            
            # (M x p x K) samples from N(0,1)
            eps = Variable(torch.randn(self.M, self.p, 10))
            eps = eps.cuda()
            
            #### step A: min_{qw} (nll + kl)
            
            self.reset_grad()

            for i in range(self.num_kq):
            
                phig_s = self.phig(img_s)  # phi(G(xs))
                wsamp = self.qw(eps)  # samples from q(w)
                
                # w'*phi(G(xs)) = (M x B x K)
                wphig_s = torch.sum( 
                  wsamp.unsqueeze(1) * phig_s.unsqueeze(0).unsqueeze(3), 
                  dim=2 )
                
                # nll loss
                loss_nll = criterion( 
                  wphig_s.view(-1,10), label_s.repeat(self.M) ) * self.Ns

                # kl loss
                loss_kl = self.kl_loss()
                
                loss = loss_nll + loss_kl
                
                # compute gradient of the loss
                loss.backward()
                
                # update models
                self.opt_qw.step()
                
                self.reset_grad()
            
            #### step B: min_{phig} (nll + kl + marg)
            
            self.reset_grad()
            
            for i in range(self.num_k):

                phig_s = self.phig(img_s)  # phi(G(xs))
                phig_t = self.phig(img_t)  # phi(G(xt))
                wsamp = self.qw(eps)  # samples from q(w)
                
                # w'*phi(G(xs)) = (M x B x K)
                wphig_s = torch.sum( 
                  wsamp.unsqueeze(1) * phig_s.unsqueeze(0).unsqueeze(3), 
                  dim=2 )
                
                # nll loss
                loss_nll = criterion( 
                  wphig_s.view(-1,10), label_s.repeat(self.M) ) * self.Ns
    
                # kl loss
                loss_kl = self.kl_loss()
                
                # margin loss on target
                f_t = torch.mm(phig_t, self.qw.mu)  # (B x K)
                top2 = torch.topk(f_t, k=2, dim=1)[0]  # (B x 2)
                  # top2[i,0] = max_j f_t[i,j], top2[:,1] = max2_j f_t[i,j] 
                gap21 = top2[:,1] - top2[:,0]  # B-dim
                std_f_t = torch.sqrt( 
                  torch.mm(phig_t**2, torch.exp(self.qw.logsd)**2) )  # (B x K)
                max_std = torch.max(std_f_t, dim=1)[0]  # B-dim
                loss_marg = torch.mean( F.relu(1.0 + gap21 + 1.96*max_std) )
                    
                loss = loss_nll + loss_kl + self.lamb_marg_loss*loss_marg
                
                # compute gradient of the loss
                loss.backward()
                
                # update models
                self.opt_phig.step()
                
                self.reset_grad()
                
            #### wrap up
            
            if batch_idx > 500:
                return batch_idx

            if batch_idx % self.interval == 0:
                prn_str = ('Train Epoch: %d [batch-idx: %d]  ' + \
                  'nll: %.6f,  kl: %.6f,  marg: %.6f') % \
                  ( epoch, batch_idx, loss_nll.item(), loss_kl.item(), 
                    loss_marg.item() )
                print(prn_str)
                if record_file:
                    record = open(record_file, 'a')
                    record.write('%s\n' % (prn_str,))
                    record.close()
                    
        return batch_idx


    ########
    def test(self, epoch, record_file=None, save_model=False):
        
        '''
        evaluate the current models on the entire test set
        '''
        
        criterion = nn.CrossEntropyLoss().cuda()
        
        # turn models into evaluation mode
        self.phig.eval()
        self.qw.eval()
        
        test_loss = 0  # test nll loss
        corrects = 0  # number of correct predictions by MAP
        size = 0  # total number of test samples
        
        # turn off autograd feature (no evaluation history tracking)
        with torch.no_grad():
            
            for batch_idx, data in enumerate(self.dataset_test):
                
                img = data['T']
                label = data['T_label']
                
                img, label = img.cuda(), label.long().cuda()
                
                #img, label = Variable(img, volatile=True), Variable(label)
                img, label = Variable(img), Variable(label)
                
                # (M x p x K) samples from N(0,1)
                #eps = Variable(torch.randn(self.M, self.p, 10))
                #eps = eps.cuda()
                
                phig = self.phig(img)  # phi(G(x))
                wmode = self.qw.mu  # mode of q(w)
                #wsamp = self.qw(eps)  # samples from q(w)
                
                # w'*phi(G(x)) = (B x K)
                output = torch.mm(phig, wmode)
                
                # w'*phi(G(x)) = (M x B x K)
                #wphig = torch.sum( 
                #  wsamp.unsqueeze(1) * phig.unsqueeze(0).unsqueeze(3), dim=2 )
                
                # nll loss (equivalent to cross entropy loss)
                test_loss += criterion(output, label).item()
                
                # class prediction
                pred = output.data.max(1)[1]  # n-dim {0,...,K-1}-valued
                  # tensor.max(j) returns a list (A, B) where
                  #   A = max of tensor over j-th dim
                  #   B = argmax of tensor over j-th dim
                
                corrects += pred.eq(label.data).cpu().numpy().sum()
                
                size += label.data.size()[0]
                
        test_loss = test_loss / size
        
        prn_str = ( 'Test set: Average nll loss: %.4f, ' + \
          'Accuracy: %d/%d (%.4f%%)\n' ) % \
          ( test_loss, corrects, size, 100. * corrects / size )
        print(prn_str)
        
        # save (append) the test scores/stats to files
        if record_file:
            record = open(record_file, 'a')
            print('recording %s\n' % record_file)
            record.write('%s\n' % (prn_str,))
            record.close()
        
        # save the models as files
        if save_model and epoch % self.save_epoch == 0:
            torch.save( self.phig,
              '%s/model_epoch%s_phig.pt' % (self.checkpoint_dir, epoch) )
            torch.save( self.qw,
              '%s/model_epoch%s_qw.pt' % (self.checkpoint_dir, epoch) )
            
        

