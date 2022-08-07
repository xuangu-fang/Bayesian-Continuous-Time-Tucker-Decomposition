import numpy as np
# import scipy
import torch
import utils

'''
SDE represent of f(1-d)  with stationary kernel GP prior:

dx/dt = Fx(t) + Lw(t)

the coorespoding LDS model is :

transition: x_k = A_{k-1} * x_{k-1} + q_{k-1}
enission: y_k = H*x_k + noise(R)

where: A_{k-1} = mat_exp(F*(t_k-t_{k-1})), q_{k-1}~N(0,P_{\inf}-A_{k-1}P_{\inf}A_{k-1}^T)

Attention, with Matern /nu=1, x is 2-d vector = (f, df/dt), H = (1,0)

'''
# SDE-inference
class LDS_GP():
    def __init__(self,hyper_para_dict):
        self.device = hyper_para_dict['device'] # add the cuda version later 

        self.N = hyper_para_dict['N'] # number of data-llk
        self.N_time = hyper_para_dict['N_time'] # number of total states

        self.F = hyper_para_dict['F'].double().to(self.device) # transition mat-SDE
        self.H = hyper_para_dict['H'].double().to(self.device) # emission mat-SDE
        self.R = hyper_para_dict['R'].double().to(self.device) # emission noise

        self.P_inf = hyper_para_dict['P_inf'].double().to(self.device) # P_inf 
        
        self.fix_int = hyper_para_dict['fix_int'] # whether the time interval is fixed
        
        

        if self.fix_int:
            self.A = torch.matrix_exp(self.F*self.fix_int)
            self.Q = self.P_inf - torch.mm(torch.mm(self.A,self.P_inf),self.A.T)
        else:
            # pre-compute and store the transition-mats for dynamic gap
            # we can also do this in each filter predict-step, but relative slow 
            self.time_int_list = hyper_para_dict['time_int_list'].to(self.device)
            # self.A_list = [torch.matrix_exp(self.F*time_int).double() for time_int in self.time_int_list]
            # self.Q_list = [self.P_inf - torch.mm(torch.mm(A,self.P_inf),A.T) for A in self.A_list]

        self.m_0 = hyper_para_dict['m_0'].double().to(self.device) # init mean
        self.P_0 = hyper_para_dict['P_0'].double().to(self.device) # init var

        self.reset_list()

        # pre-match the test state
        self.train_time = hyper_para_dict['train_time']
        self.test_time = hyper_para_dict['test_time']

        self.test_X_state = self.test_state_match()
        
    def test_state_match(self):
            
        # find the closest training state for each test data
        
        # if self.fix_int:
        #     test_X_state = (self.test_time/self.fix_int).floor()
        #     test_X_state = torch.where(test_X_state.long()>self.N_time-1,self.N_time-1,test_X_state.long())

        # else:
        test_X_state = []
        for test_X in self.test_time.numpy():
                test_X_state.append(utils.bisect_search(self.train_time.cpu().numpy(),test_X)-1)

        return torch.tensor(test_X_state).long()

    def reset_list(self):
        self.m = self.m_0 # store the current state(mean)
        self.P = self.P_0 # store the current state(var)


        self.m_list = [] # store the filter-update state(mean)
        self.P_list = [] # store the filter-update state(var)

        self.m_pred_list = [] # store the filter-pred state(mean)
        self.P_pred_list = [] # store the filter-pred state(var)

        self.m_smooth_list = [] # store the smoothed state(mean)
        self.P_smooth_list = [] # store the smoothed state(mean)
 
    def filter_predict(self,ind=None,time_int=None):
        if self.fix_int:
            self.m = torch.mm(self.A,self.m).double()
            self.P = torch.mm(torch.mm(self.A,self.P),self.A.T) + self.Q

            self.m_pred_list.append(self.m)
            self.P_pred_list.append(self.P)

        # none-fix-interval, recompute A,Q based on current time-interval
        else:

            if ind is None:
                raise Exception('need to input the state-index for non-fix-interval case')

            time_int = self.time_int_list[ind]
            self.A = torch.matrix_exp(self.F*time_int).double()
            self.Q = self.P_inf - torch.mm(torch.mm(self.A,self.P_inf),self.A.T)

            # self.A = self.A_list[ind]
            # self.Q = self.Q_list[ind]
            
            self.m = torch.mm(self.A,self.m).double()
            self.P = torch.mm(torch.mm(self.A,self.P),self.A.T) + self.Q

            self.m_pred_list.append(self.m)
            self.P_pred_list.append(self.P)

            # self.tau_list.append(tau) # store all the time interval

    def filter_update(self,y):
        
        V = y-torch.mm(self.H,self.m)
        S = torch.mm(torch.mm(self.H,self.P),self.H.T)+self.R
        K = torch.mm(torch.mm(self.P,self.H.T),torch.linalg.pinv(S))
        self.m = self.m + torch.mm(K,V)
        # self.P = self.P - torch.mm(torch.mm(K,S),K.T)
        self.P = self.P - torch.mm(torch.mm(K,self.H),self.P)
        

        self.m_list.append(self.m)
        self.P_list.append(self.P)

    def smooth(self):
        
        # start from the last end
        m_s = self.m_list[-1]
        P_s = self.P_list[-1]

        self.m_smooth_list.insert(0,m_s)
        self.P_smooth_list.insert(0,P_s)

        if self.fix_int:
            for i in reversed(range(self.N_time-1)):

                m = self.m_list[i]
                P = self.P_list[i]

                m_pred = self.m_pred_list[i+1]
                P_pred = self.P_pred_list[i+1]

                G = torch.mm(torch.mm(P,self.A.T),torch.linalg.pinv(P_pred))
                m_s = m + torch.mm(G,m_s-m_pred)
                P_s = P + torch.mm(torch.mm(G,P_s-P_pred),G.T)
            
                self.m_smooth_list.insert(0,m_s)
                self.P_smooth_list.insert(0,P_s)

        else: # to be verify
            for i in reversed(range(self.N_time-1)):

                m = self.m_list[i]
                P = self.P_list[i]

                m_pred = self.m_pred_list[i+1]
                P_pred = self.P_pred_list[i+1]

                
                time_int = self.time_int_list[i+1]
                A = torch.matrix_exp(self.F*time_int)
                # A = self.A_list[i+1]

                G = torch.mm(torch.mm(P,A.T),torch.linalg.pinv(P_pred))
                m_s = m + torch.mm(G,m_s-m_pred)
                P_s = P + torch.mm(torch.mm(G,P_s-P_pred),G.T)
            
                self.m_smooth_list.insert(0,m_s)
                self.P_smooth_list.insert(0,P_s)
                

class LDS_CEP_fixU(LDS_GP):
    def __init__(self,hyper_para_dict):
        JITTER = 1e-4        
        super(LDS_CEP_fixU,self).__init__(hyper_para_dict)
        # attention, self.H(observed transition), self.R(observed noise) in base class are not fixted but vars to be learnt for each time-point
        
        # training data
        self.ind_tr = hyper_para_dict['ind_tr']
        self.y_tr = hyper_para_dict['y_tr'].to(self.device) # N*1
        
        # some hyper-paras
        self.epoch = hyper_para_dict['epoch'] # passing epoch
        self.U = [item.to(self.device) for item in hyper_para_dict['U']] # list of mode embedding, fixed and known in this setting
        self.nmod = len(self.U)
        self.R_U = hyper_para_dict['R_U'] # rank of latent factor of embedding
        
        self.H= hyper_para_dict['H'].double().to(self.device) # the fixed (R_U * 2R_U) size sde obv-mat to extract latent vars Gamma   
        
        
        # prior of noise
        self.v = hyper_para_dict['v'] # varience of prior on embedding, will not use as u given and fix
        self.a0 = hyper_para_dict['a0']
        self.b0 = hyper_para_dict['b0']
        
        # init the message/llk factor
        self.msg_gamma_m = torch.zeros(self.N,self.R_U,1).double().to(self.device) # N*R_U*1
        # self.msg_gamma_v = torch.eye(self.R_U).reshape((1,self.R_U,self.R_U)).repeat(self.N,1,1).double() # N*R_U*R_U
        self.msg_gamma_v = 0.1*torch.ones(self.N,1).to(self.device) # only for fix U case 
        
        # self.msg_gamma_v_jitter = JITTER * self.msg_gamma_v
        
        self.msg_a = torch.ones(self.N,1).double().to(self.device) # N*1
        self.msg_b = torch.ones(self.N,1).double().to(self.device) # N*1
        
        # init the approx. post factor (over all data points)        
        self.post_a = self.a0
        self.post_b = self.b0
        
        # value will be updated after filter&smooth
        self.post_gamma_m = self.msg_gamma_m.to(self.device) # N*R_U*1
        self.post_gamma_v = self.msg_gamma_v.to(self.device)#0.1*torch.eye(self.R_U).reshape((1,self.R_U,self.R_U)).repeat(self.N,1,1).double() # N*R_U*R_U
        # self.post_gamma_v = torch.ones(self.N,1) * 0.1 # only for 
        
        
        # Expectation terms over current approx. post
        # self.E_u = None
        # self.E_u_2 = None
        self.E_tau = None
        self.E_gamma = None
        self.E_gamma_2 = None
        
        self.E_z = None 
        self.E_z_2 = None
    
    def expectation_update_z(self,ind):
        # compute E_z,E_z_2 for given datapoints by current post.U
        # in fixU setting, only call it once, b/c U is fixed and given
        
        # batch_U = [self.U[k][ind[:, k],:] for k in range(self.nmod)]
        
        self.E_z = torch.ones(self.N,self.R_U).to(self.device) # N*R_U
        self.E_z_2 = torch.ones(self.N,self.R_U,self.R_U).to(self.device)# N*R_U*R_U
        
        for k in range(self.nmod):
            term = self.U[k][ind[:, k],:]
            self.E_z = self.E_z * term # N*R_U
            
            expand1 = torch.unsqueeze(term,2) # N*R_U*1
            expand2 = torch.unsqueeze(term,1) # N*1*R_U
            
            self.E_z_2 = self.E_z_2 * torch.bmm(expand1,expand2) # N*R_U
              
        self.E_z = torch.unsqueeze(self.E_z,-1) # N*R_U*1
        
    def expectation_update_tau(self):
        self.E_tau = self.post_a/self.post_b
        
    def expectation_update_gamma(self):
        self.E_gamma = self.post_gamma_m.double() # N*R_U*1
        E_gamma_T = torch.transpose(self.E_gamma, dim0=1, dim1=2)
        self.E_gamma_2 = self.post_gamma_v + torch.bmm(self.E_gamma,E_gamma_T) # N*R_U*R_U
    
    def msg_update_tau(self):
        self.msg_a = 1.5*torch.ones(self.N,1).to(self.device)
         
        term1 = 0.5*torch.square(self.y_tr) # N*1
        term2 = self.y_tr * torch.squeeze(torch.bmm(torch.transpose(self.E_gamma,dim0=1,dim1=2) , self.E_z),dim=-1)# N*1
        term3 = torch.unsqueeze(0.5* torch.einsum('bii->b',torch.bmm(self.E_gamma_2,self.E_z_2)),dim=-1) # N*1
        
        self.msg_b =  term1 - term2 + term3 # N*1
    
    def msg_update_gamma(self):
        
        
        # general case
        # the (approx.) observation noise in KF update step
        # self.msg_gamma_v = torch.linalg.pinv(self.E_tau*self.E_z_2)  # N*R_U*R_U
        
        # the (approx.) observation values in KF update step
        # self.msg_gamma_m = torch.linalg.solve(self.E_z_2, torch.bmm(self.E_z,self.y_tr.unsqueeze(-1)))# N*R_U*1
        # term1 = torch.bmm(self.E_z,self.y_tr.unsqueeze(-1)) * self.E_tau
        # self.msg_gamma_m = torch.bmm(self.msg_gamma_v,term1)
        
        
        # fixU case
        self.msg_gamma_v = self.msg_b / self.msg_a
        self.msg_gamma_m = self.E_z
        
    # def msg_update_U(self):
    #     # will be very complex..check BASS code, will not be uses in this setting(fix U)
    #     pass 
    
    def post_update_tau(self):
        # update post. factor of tau based on current msg. factors
    
        self.post_a = self.a0 + self.msg_a.sum() - self.N
        self.post_b = self.b0 + self.msg_b.sum()  
        
    def post_update_gamma(self):
        # update post. factor of gamma based on latest results from RTS-smoother
    
        self.post_gamma_m = torch.matmul(self.H,torch.cat(self.m_smooth_list,dim=1)).T # N*R_U
        self.post_gamma_m = torch.unsqueeze(self.post_gamma_m,-1)# N*R_U*1
        
        # tricky point: batch mat-mul to compute H*P*H^T, where P size: N*2R_U*2R_U, H size: R_U*2R_U 
        P = torch.stack(self.P_smooth_list,dim=0)# N*2R_U*2R_U
        
        # step1: A=P*H.T
        tensor1 = torch.matmul(P,self.H.T) # N*2R_U*R_U 
        
        # step2: B= H*A = H*P*H.T = (A^T * H^T )^T
        tensor2 = torch.matmul(torch.transpose(tensor1,dim0=1,dim1=2), self.H.T)# N*R_U*R_U 
        self.post_gamma_v = torch.transpose(tensor2,dim0=1,dim1=2) # N*R_U*R_U

    def filter_update(self,y,R,z):
        # to be overload to implement the dynamic observd y,R, based on msg_gamma
        # for fix U case, emission formula is N(y|) 
        
        H = torch.mm(z.T,self.H)
        V = y-torch.mm(H,self.m)
        S = torch.mm(torch.mm(H,self.P),H.T)+R
        K = torch.mm(torch.mm(self.P,H.T),torch.linalg.pinv(S))
        self.m = self.m + torch.mm(K,V)
        # self.P = self.P - torch.mm(torch.mm(K,S),K.T)
        self.P = self.P - torch.mm(torch.mm(K,H),self.P)
        
        self.m_list.append(self.m)
        self.P_list.append(self.P)
