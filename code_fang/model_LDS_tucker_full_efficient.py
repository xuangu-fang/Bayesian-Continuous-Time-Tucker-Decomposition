import numpy as np
from numpy.lib import utils
import torch 
# import matplotlib.pyplot as plt
from model_LDS import LDS_GP
import tensorly as tl
from utils import build_id_key_table
from utils import kronecker_product_einsum_batched
tl.set_backend('pytorch')

# full tucker form based dynamic model, efficient version

#  gamma is coorespoding to the tucker-core W in draft
# z is coorespoding to b_i =  \kronecker_prod u in draft
# z_del  is coorespoding to b^{\k} = \kronecker_prod_{j \neq k} u_j in draft
# we don't explicity encode/store msg_gamma_eta.lam for each data-likelihood


class LDS_CEP_tucker_full_efficient(LDS_GP):

    def __init__(self,hyper_para_dict):

        super(LDS_CEP_tucker_full_efficient,self).__init__(hyper_para_dict)
        
        self.DAMPPING_gamma = hyper_para_dict['DAMPPING_gamma']
        self.DAMPPING_U = hyper_para_dict['DAMPPING_U']

        self.train_time_ind =hyper_para_dict['train_time_ind'] 
        self.test_time_ind = hyper_para_dict['test_time_ind']   
        self.N_time = hyper_para_dict['N_time'] 
        self.time_data_table = hyper_para_dict['time_data_table']

        # training data
        self.ind_tr = hyper_para_dict['ind_tr']
        self.y_tr = hyper_para_dict['y_tr'].to(self.device) # N*1
        
        # some hyper-paras
        self.epoch = hyper_para_dict['epoch'] # passing epoch
        self.ndims = hyper_para_dict['ndims'] 
        self.U = [item.to(self.device) for item in hyper_para_dict['U']] # list of mode embedding, fixed and known in this setting
        self.nmod = len(self.U)
        self.R_U = hyper_para_dict['R_U'] # rank of latent factor of embedding
        
        self.H= hyper_para_dict['H'].double().to(self.device) # the fixed (nmod*R_U) * (nmod*2 R_U) size sde obv-mat to extract latent vars Gamma   
        
        
        # prior of noise
        self.v = hyper_para_dict['v'] # prior varience of embedding (scaler)
        self.a0 = hyper_para_dict['a0']
        self.b0 = hyper_para_dict['b0']


        self.nmod_list = [self.R_U for k in range(self.nmod)]
        self.gamma_size = np.prod(self.nmod_list) # R^K


        # init the message/llk factor

        # nature paras of msg_gamma, lam = v_{-1}, eta =  v_{-1}*m, as 
        
        # self.msg_gamma_lam = 1e-4*torch.eye(self.gamma_size).reshape((1,self.gamma_size,self.gamma_size)).repeat(self.N_time,1,1).double().to(self.device) # N*(R^K)*(R^K)
        # self.msg_gamma_eta = torch.zeros(self.N_time,self.gamma_size,1).double().to(self.device) # N*(R^K)*1

        # self.msg_gamma_m =  torch.zeros(self.N_time,self.gamma_size,1).double().to(self.device) #  (N*(R^K)*1) 
        # self.msg_gamma_v = torch.eye(self.gamma_size).reshape((1,self.gamma_size,self.gamma_size)).repeat(self.N_time,1,1).double().to(self.device) # (N*(R^K)*(R^K)) 

        # msg of tau
        self.msg_a = 1.5*torch.ones(self.N,1).double().to(self.device) # N*1
        self.msg_b = torch.ones(self.N,1).double().to(self.device) # N*1

        # nature paras of msg_U, lam = v_{-1}, eta =  v_{-1}*m, as 
        self.msg_U_lam = [1e-4*torch.eye(self.R_U).reshape((1,self.R_U,self.R_U)).repeat(self.N,1,1).double().to(self.device) for i in range(self.nmod) ] # (N*R_U*R_U)*nmod
        self.msg_U_eta =  [torch.zeros(self.N,self.R_U,1).double().to(self.device) for i in range(self.nmod)] # (N*R_U*1)*nmod

        # init the approx. post. factor (over all data points)        
        
        # post. of tau
        self.post_a = self.a0
        self.post_b = self.b0        

        # post. of gamma, value will be updated after filter&smooth
        self.post_gamma_m = torch.zeros(self.N_time,self.gamma_size,1).double().to(self.device) # N*(R^K)*1
        self.post_gamma_v = torch.eye(self.gamma_size).reshape((1,self.gamma_size,self.gamma_size)).repeat(self.N_time,1,1).double().to(self.device) # N*(R^K)*(R^K)

        # post.of U     
        self.post_U_m = [item.unsqueeze(-1) for item in self.U] # nmod(list) * (ndim_i * R_U *1) 
        self.post_U_v = [(self.v) *torch.eye(self.R_U).reshape((1,self.R_U,self.R_U)).repeat(ndim,1,1).double().to(self.device) \
            for ndim in self.ndims]# nmod(list) * (ndim_i * R_U *R_U ) 

        
        # Expectation terms over current approx. post

        self.E_tau = 1.0

        # directly use self.post_gamma_m? 
        # self.E_gamma = torch.ones(self.N_time,self.gamma_size,1).double().to(self.device) # N*(R^K)*1

        # self.E_gamma_2 = torch.ones(self.N,self.gamma_size,self.gamma_size).double().to(self.device) # N*(R^K)*(R^K)


        # does not store E_U,E_U_2 explictly, which it's trival form the post.U, store E_z instead
        # no need to store the conditional moment for each mode(just compute, use and drop for saving memory)
        # if we won't, use list here 
        
        
        # for efficient memory, we don't store moment of z or z_del, but just compute,use and drop
        self.E_z_del = None  # N* (R^{K-1}) *1
        self.E_z_del_2 = None # N* (R^{K-1}) * (R^{K-1})

        self.E_z = None # N* (R^K) *1
        self.E_z_2 = None # N* (R^K) * (R^K)

        # self.expectation_update_z()

        # for mode in range(self.nmod):
        #     self.expectation_update_z_del(mode)

        # uid-data table 
        self.uid_table, self.data_table = build_id_key_table(self.nmod,self.ind_tr) 

        # some placeholder vars:
        self.unfold_gamma = torch.zeros(self.N_time,self.R_U,int(self.gamma_size/self.R_U)).double().to(self.device) # N*R*R^{k-1}
        self.ones_const = torch.ones(self.N,1).to(self.device)


    def moment_kronecker_tucker(self,modes,ind,order='first'):
       # computhe first and second moments of kronecker_prod_{k \in given modes} u_k (denote as b/b^{\k} in draft)
        
        ind = ind.reshape(-1,self.nmod) # N * nmod
        last_mode = modes[-1]

        assert order in {'first','second'}
        if order == 'first':
            # only compute the first order moment
            E_z = self.post_U_m[last_mode][ind[:,last_mode]] # N*R_u*1
            
            for mode in reversed(modes[:-1]):
                E_u = self.post_U_m[mode][ind[:,mode]] # N*R_u*1
            
                E_z = kronecker_product_einsum_batched(E_z,E_u)
                
            return E_z

        elif order == 'second':
            #  compute the second order moment E_z / E_z_2
            E_z = self.post_U_m[last_mode][ind[:,last_mode]] # N*R_u*1
            E_z_2 = self.post_U_v[last_mode][ind[:,last_mode]] + torch.bmm(E_z,E_z.transpose(dim0=1,dim1=2)) # N*R_u*R_U

            for mode in reversed(modes[:-1]):
                E_u = self.post_U_m[mode][ind[:,mode]] # N*R_u*1
                E_u_2 = self.post_U_v[mode][ind[:,mode]] + torch.bmm(E_u,E_u.transpose(dim0=1,dim1=2)) # N*R_u*R_U

                E_z = kronecker_product_einsum_batched(E_z,E_u)
                E_z_2 = kronecker_product_einsum_batched(E_z_2,E_u_2)

            return E_z,E_z_2

        else: raise Exception('wrong arg')

    def expectation_update_z_del(self,del_mode):
        # compute E_z_del,E_z_del_2 by current post.U and deleting the info of mode_k
        
        other_modes = [i for i in range(self.nmod)]
        other_modes.remove(del_mode)
        self.E_z_del = self.moment_kronecker_tucker(other_modes,self.ind_tr,'first')

    def expectation_update_z(self): 
        # compute E_z,E_z_2 for given datapoints by current post.U (merge info of all modes )

        all_modes = [i for i in range(self.nmod)]        
        self.E_z = self.moment_kronecker_tucker(all_modes,self.ind_tr,'first')  

    def expectation_update_tau(self):
        self.E_tau = self.post_a/self.post_b
        
    def expectation_update_gamma(self):

        self.E_gamma = self.post_gamma_m # N*(R^K)*1
        # self.E_gamma_2 = self.post_gamma_v + torch.bmm(self.E_gamma,self.E_gamma.transpose(dim0=1, dim1=2)) # N*(R^K)*(R^K)
    
    def msg_update_U(self):
        
        for mode in range(self.nmod):
            
            # self.expectation_update_z_del(mode)
            other_modes = [i for i in range(self.nmod)]
            other_modes.remove(mode)

            # unfolding the gamma/W (vec->tensor->matrix) at each state
            for i in range(self.N_time):

                # E_gamma_tensor = tl.tensor(self.E_gamma[i].reshape(self.nmod_list)) # (R^k *1)-> (R * R * R ...)
                E_gamma_tensor = tl.tensor(self.post_gamma_m[i].reshape(self.nmod_list)) # (R^k *1)-> (R * R * R ...)
                self.unfold_gamma[i] = tl.unfold(E_gamma_tensor,mode).double()  #  (R * R * R ...) -> (R * (R^{K-1}))
                
            for j in range(len(self.uid_table[mode])):
                uid = self.uid_table[mode][j] # id of embedding
                eid = self.data_table[mode][j] # id of associated entries
                tid = self.train_time_ind[eid] # id time states of such entries
                # compute msg of associated entries, update with damping

                E_z_del, E_z_del_2 = self.moment_kronecker_tucker(other_modes,self.ind_tr[eid],order='second')

                msg_U_lam_new = self.E_tau * torch.bmm(self.unfold_gamma[tid],\
                                    torch.bmm(E_z_del_2,self.unfold_gamma[tid].transpose(dim0=1,dim1=2))) # num_eid * R_U * R_U
                
                # to compute E_a = W_k * z\ 
                E_a = torch.bmm(self.unfold_gamma[tid],E_z_del)
                msg_U_eta_new =  self.E_tau * torch.bmm(E_a, self.y_tr[eid].unsqueeze(-1)) # num_eid * R_U *1
                
                self.msg_U_lam[mode][eid] = self.DAMPPING_U * self.msg_U_lam[mode][eid] + (1- self.DAMPPING_U ) * msg_U_lam_new # num_eid * R_U * R_U
                self.msg_U_eta[mode][eid] = self.DAMPPING_U * self.msg_U_eta[mode][eid] + (1- self.DAMPPING_U ) * msg_U_eta_new # num_eid * R_U * 1
      
    def msg_update_tau(self,time_id):

        all_modes = [i for i in range(self.nmod)]
        data_id = self.time_data_table[time_id]
        E_z = self.moment_kronecker_tucker(all_modes,self.ind_tr[data_id],order='first')

        # self.msg_a[data_id] = 1.5*torch.ones(len(data_id),1).to(self.device)
         
        term1 = 0.5*torch.square(self.y_tr[data_id]) # N*1
        term2 = self.y_tr[data_id] * torch.squeeze(torch.bmm(torch.transpose(self.post_gamma_m[self.train_time_ind[data_id]],dim0=1,dim1=2) , E_z),dim=-1)# N*1
        
        # to compute term3 = 0.5* E[gamma^T * z* z^T * gamma] w(o) using E_z_2, 
        # we use first-order expanda here,  term3 = 0.5 * (gamma.T * E_z)^2
        term3 = 0.5 * torch.square(torch.bmm(self.post_gamma_m[self.train_time_ind[data_id]].transpose(dim0=1,dim1=2),E_z).squeeze(-1))# N*1
        self.msg_b[data_id] =  term1 - term2 + term3 # N*1

    def msg_update_gamma(self,time_id):

        all_modes = [i for i in range(self.nmod)]


        data_id = self.time_data_table[time_id]

        E_z, E_z_2 = self.moment_kronecker_tucker(all_modes,self.ind_tr[data_id],order='second')

        # no damping
        msg_gamma_v = torch.linalg.inv((self.E_tau*E_z_2).sum(dim=0))  
        msg_gamma_m = torch.mm(msg_gamma_v,(self.E_tau*torch.bmm(E_z,self.y_tr[data_id].unsqueeze(-1))).sum(dim=0))

        return msg_gamma_m, msg_gamma_v
        # msg_gamma_lam_new = (self.E_tau*E_z_2).sum(dim=0) # (R_U)^K*(R_U)^K
        # msg_gamma_eta_new = (self.E_tau*torch.bmm(E_z,self.y_tr[data_id].unsqueeze(-1))).sum(dim=0)

        # self.msg_gamma_lam[i] = self.DAMPPING_gamma * self.msg_gamma_lam[i] + (1-self.DAMPPING_gamma)*msg_gamma_lam_new 
        # self.msg_gamma_eta[i] = self.DAMPPING_gamma * self.msg_gamma_eta[i] + (1-self.DAMPPING_gamma)*msg_gamma_eta_new

        # self.msg_gamma_v[i] = torch.linalg.inv(self.msg_gamma_lam[i])  
        # self.msg_gamma_m[i] = torch.mm(self.msg_gamma_v[i],self.msg_gamma_eta[i])
        




    def post_update_U(self):
    
        # merge such msgs to get post.U

        for mode in range(self.nmod):
            for j in range(len(self.uid_table[mode])):

                uid = self.uid_table[mode][j] # id of embedding
                eid = self.data_table[mode][j] # id of associated entries

                self.post_U_v[mode][uid] = torch.linalg.inv(self.msg_U_lam[mode][eid].sum(dim=0) + (1.0/self.v)*torch.eye(self.R_U).to(self.device)) # R_U * R_U
                self.post_U_m[mode][uid] = torch.mm(self.post_U_v[mode][uid],self.msg_U_eta[mode][eid].sum(dim=0)) # R_U *1
                
               
    def post_update_tau(self):
        # update post. factor of tau based on current msg. factors
    
        self.post_a = self.a0 + self.msg_a.sum() - self.N
        self.post_b = self.b0 + self.msg_b.sum()  

    def post_update_gamma(self):
        # update post. factor of gamma based on latest results from LDS system (RTS-smoother)

        self.post_gamma_m =  torch.matmul(self.H,torch.cat(self.m_smooth_list,dim=1)).T.unsqueeze(dim=-1)  # N_time*(R^k)*1

        #  batch mat-mul to compute H*P*H^T 
        P = torch.stack(self.P_smooth_list,dim=0)# N*(2*R^k)*(2*R^k)
        # step1: A=P*H.T
        tensor1 = torch.matmul(P,self.H.T) # N* (2*R^k) * (R^k)        
        # step2: B= H*A = H*P*H.T = (A^T * H^T )^T
        self.post_gamma_v = torch.matmul(tensor1.transpose(dim0=1,dim1=2), self.H.T).transpose(dim0=1,dim1=2)# N_time* (R^k) * (R^k)

    def filter_update_simple(self,y,R,z):
        # for one-order expand based simple approx. case,
        #  emission formula is N(y_tr|H*x*E_z,E_tau), gamma = H*x
        
        H = torch.mm(z.T,self.H)
        V = y-torch.mm(H,self.m)
        S = torch.mm(torch.mm(H,self.P),H.T)+R
        K = torch.mm(torch.mm(self.P,H.T),torch.linalg.pinv(S))
        self.m = self.m + torch.mm(K,V)
        # self.P = self.P - torch.mm(torch.mm(K,S),K.T)
        self.P = self.P - torch.mm(torch.mm(K,H),self.P)
        
        self.m_list.append(self.m)
        self.P_list.append(self.P)

    def filter_update_general(self,y,R):

        V = y-torch.mm(self.H,self.m)
        S = torch.mm(torch.mm(self.H,self.P),self.H.T)+R
        K = torch.mm(torch.mm(self.P,self.H.T),torch.linalg.pinv(S))
        self.m = self.m + torch.mm(K,V)
        # self.P = self.P - torch.mm(torch.mm(K,S),K.T)
        self.P = self.P - torch.mm(torch.mm(K,self.H),self.P)
        
        self.m_list.append(self.m)
        self.P_list.append(self.P)


    def model_test(self,test_ind,test_y,test_time):
            
        MSE_loss = torch.nn.MSELoss()
        MAE_loss = torch.nn.L1Loss()
        smooth_result = torch.cat(self.m_smooth_list,dim=1) # size: (2*R_U)*N

        # train_loss

        # y_pred_train = torch.bmm(self.E_z.transpose(dim0 = 1,dim1 = 2),self.post_gamma_m[self.train_time_ind]).squeeze()
        # y_true_train = self.y_tr.squeeze()
        # loss_train =  torch.sqrt(MSE_loss(y_pred_train,y_true_train))

        # test_loss
        all_modes = [i for i in range(self.nmod)]        
        E_z_test = self.moment_kronecker_tucker(all_modes,test_ind,'first') 

        # find the closest state

        base_state = smooth_result[:,self.test_X_state].squeeze()  # 2R_U * N 
        base_gamma = torch.matmul(self.H,base_state).T.unsqueeze(dim=-1) # N*R_U*1
        
        # test_loss_base: 
        tid = self.test_time_ind
        y_pred_test_base = torch.bmm(E_z_test.transpose(dim0 = 1,dim1 = 2),base_gamma[tid]).squeeze()
        loss_test_rmse =  torch.sqrt(MSE_loss(y_pred_test_base,test_y.squeeze().to(self.device)))
        loss_test_MAE = MAE_loss(y_pred_test_base,test_y.squeeze().to(self.device))

        # test_loss_transfer: no need for current dataset, as all timestamps of test data 
        # # have shown in training data

        # get the time-gap
        # test_X_tau = (test_time - self.train_time[self.test_X_state]).to(self.device) 

        # # there's no efficient way to do the "apply" on tensor for both gpu/cpu
        # trans_mat_list = [torch.matrix_exp(self.F*test_X_tau[i]) for i in range(len(test_X_tau))]
        # trans_mat_batch = torch.stack(trans_mat_list) # size: N_test * 2R_u * 2R_U      
        # trans_state = torch.bmm(trans_mat_batch,base_state.T.unsqueeze(-1)).squeeze().T # size: R_u * N_test 

        # trans_gamma = torch.matmul(self.H,trans_state).T.unsqueeze(dim=-1) # N*R_U*1  

        # y_pred_test_trans = torch.bmm(E_z_test.transpose(dim0 = 1,dim1 = 2),trans_gamma[tid]).squeeze()
        # loss_test_trans =  torch.sqrt(MSE_loss(y_pred_test_trans,test_y.squeeze().to(self.device)))

        return loss_test_rmse,loss_test_MAE
