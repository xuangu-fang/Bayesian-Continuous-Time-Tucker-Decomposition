import numpy as np
import torch
import utils
from torch.utils.data import Dataset
# torch.manual_seed(513)

def data_loader_CEP_base(args,data_file_name):

    # assert args.method in {'CP-LDS','CP-static','CP-discrete',\
                            # 'Tucker-LDS','Tucker-static','Tucker-discrete'}
    
    data = np.load(data_file_name, allow_pickle=True)

    train_ind = data.item().get('train_ind')
    train_y = data.item().get('train_y')
    train_time = data.item().get('train_time_uni') # sorted & unique timestamps
    train_time_disct = data.item().get('train_time_disct')
    train_time_ind = data.item().get('train_time_ind')


    test_ind = data.item().get('test_ind')
    test_y = data.item().get('test_y')
    test_time = data.item().get('test_time_uni')
    test_time_ind = data.item().get('test_time_ind')
    test_time_disct = data.item().get('test_time_disct')

    ndims = data.item().get('ndims')
    nmod = len(ndims)
    N = train_y.size
    N_time = train_time.size

    nepoch = args.epoch

    R_U = args.R_U

    if args.device == 'cpu':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    if 'discrete' in args.method:
        # add discrete time as extra mode
        train_ind =  np.concatenate([train_ind,train_time_disct.reshape(-1,1)],1)
        test_ind =  np.concatenate([test_ind,test_time_disct.reshape(-1,1)],1)
        nmod = nmod+1
        ndims = data.item().get('ndims_time') 
      

    if 'CP' in args.method:
        # CP-based model
        gamma_size = R_U 
        # U = [3*torch.rand(ndim,R_U).double() for ndim in ndims]
    elif 'Tucker' in args.method:
        # Tucker-based model
        gamma_size = np.prod([R_U for k in range(nmod)]) # R^K
        # U = [torch.rand(ndim,R_U).double() for ndim in ndims]
    else:
        raise Exception("only support CP/Tucker based method")


    if 'LDS' in args.method:
        time_data_table = utils.build_time_data_table(train_time_ind)
    else:
        time_data_table = None


    # array to tensor
    U = [torch.rand(ndim,R_U).double() for ndim in ndims]
    # U = [torch.randn(ndim,R_U).double() for ndim in ndims]

    # if args.dataset == 'dblp':
    #     U = [3*torch.randn(ndim,R_U).double() for ndim in ndims]

    train_time = torch.tensor(train_time)
    test_time = torch.tensor(test_time)

    if args.fix_int:
        # fix-time-interval setting
        fix_int = torch.abs(train_time[1]-train_time[0]).squeeze()
        time_int_list = fix_int*torch.ones(N) 
    else:
        # non-fix-time-interval setting, compute the gap between each two time-stamps
        fix_int = None
        time_int_list_follow = [train_time[i+1] - train_time[i] for i in range(N_time-1)]
        time_int_list =torch.tensor([0.0] + time_int_list_follow)

    train_y = torch.tensor(train_y).reshape(-1,1)
    test_y = torch.tensor(test_y).reshape(-1,1)


    # hyper-paras setting for LDS system
    lengthscale = torch.tensor(args.ls)
    variance = torch.tensor(0.1)
    noise = torch.tensor(0.1)

    # matern 32 kernel
    lamb = np.sqrt(3)/lengthscale
    F_base = torch.tensor([[0,1.0],
                    [-lamb*lamb,-2*lamb]])
    P_inf_base = torch.tensor([[variance,0.0],
                        [0,lamb*lamb*variance]])
    H_base = torch.tensor([[1.0,0]])

    F_list =[F_base for i in range(gamma_size)]
    P_inf_list =[P_inf_base for i in range(gamma_size)]
    H_list = [H_base for i in range(gamma_size)]

    F = torch.block_diag(*F_list)
    P_inf = torch.block_diag(*P_inf_list)
    H = torch.block_diag(*H_list)


    m_0 = torch.zeros(2*gamma_size,1)
    P_0 = P_inf

    # package the hypyerparas

    hyper_para_dict = {}
    hyper_para_dict['N'] = N
    hyper_para_dict['F'] = F
    hyper_para_dict['H'] = H
    hyper_para_dict['R'] = noise
    hyper_para_dict['P_inf'] = P_inf
    hyper_para_dict['fix_int'] = fix_int # set as None if it's not fix 
    hyper_para_dict['m_0'] = m_0
    hyper_para_dict['P_0'] = P_0

    hyper_para_dict['U'] = U
    hyper_para_dict['device'] = device
    hyper_para_dict['ind_tr'] = train_ind
    hyper_para_dict['y_tr'] = train_y

    hyper_para_dict['test_ind'] = test_ind
    hyper_para_dict['test_y'] = test_y

    hyper_para_dict['epoch'] = nepoch
    hyper_para_dict['R_U'] = R_U
    hyper_para_dict['v'] = 1
    hyper_para_dict['a0'] = 10.0
    hyper_para_dict['b0'] = 1.0
    hyper_para_dict['time_int_list'] = time_int_list
    hyper_para_dict['ndims'] = ndims
    hyper_para_dict['train_time'] = train_time
    hyper_para_dict['test_time'] = test_time

    hyper_para_dict['train_time_ind'] = train_time_ind
    hyper_para_dict['test_time_ind'] = test_time_ind
    hyper_para_dict['N_time'] = N_time
    hyper_para_dict['time_data_table'] = time_data_table

    hyper_para_dict['gamma_size'] = gamma_size

    hyper_para_dict['DAMPPING_U'] = args.DAMPPING_U
    hyper_para_dict['DAMPPING_gamma'] = args.DAMPPING_gamma

    return hyper_para_dict

def data_loader_GPTF_base(args,data_file_name):
    
    assert args.method in {'GPTF-discrete','GPTF-time','GPTF-static','CPTF-time'}

    data = np.load(data_file_name, allow_pickle=True)

    train_ind =data.item().get('train_ind')
    train_y = data.item().get('train_y')
    train_time = data.item().get('train_time')
    train_time_disct = data.item().get('train_time_disct')
    
    test_ind = data.item().get('test_ind')
    test_y = torch.tensor(data.item().get('test_y'))
    test_time = torch.tensor(data.item().get('test_time'))
    test_time_disct = data.item().get('test_time_disct')


    ndims = data.item().get('ndims')
    
    R_U = args.R_U

    if args.method == 'GPTF-discrete':
        # add discrete time as extra mode
        train_ind =  np.concatenate([train_ind,train_time_disct.reshape(-1,1)],1)
        test_ind =  np.concatenate([test_ind,test_time_disct.reshape(-1,1)],1)
        ndims = data.item().get('ndims_time')

    U = [np.random.rand(ndim,R_U) for ndim in ndims]

    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # package the hypyerparas
    hyper_para_dict = {}

    hyper_para_dict['U'] = U
    hyper_para_dict['device'] = device

    hyper_para_dict['train_ind'] = train_ind
    hyper_para_dict['train_y'] = train_y
    hyper_para_dict['train_time'] = train_time

    hyper_para_dict['test_ind'] = test_ind
    hyper_para_dict['test_y'] = test_y
    hyper_para_dict['test_time'] = test_time

    hyper_para_dict['ndims'] = ndims


    return hyper_para_dict

def data_loader_SVI_base(args,data_file_name):
    
    assert args.method in {'CP-SVI','Tucker-SVI'}

    data = np.load(data_file_name, allow_pickle=True)

    train_ind =data.item().get('train_ind')
    train_y = data.item().get('train_y')
    # train_time = data.item().get('train_time')
    train_time_disct = data.item().get('train_time_disct')
    
    test_ind = data.item().get('test_ind')
    test_y = torch.tensor(data.item().get('test_y'))
    # test_time = torch.tensor(data.item().get('test_time'))
    test_time_disct = data.item().get('test_time_disct')

    train_y = torch.tensor(train_y)
    test_y = torch.tensor(test_y)

    ndims = data.item().get('ndims')
    nmod = len(ndims)
    
    N = train_y.size
    
    R_U = args.R_U



    # add discrete time as extra mode
    train_ind =  np.concatenate([train_ind,train_time_disct.reshape(-1,1)],1)
    test_ind =  np.concatenate([test_ind,test_time_disct.reshape(-1,1)],1)
    ndims = data.item().get('ndims_time')
    nmod = nmod + 1

    U = [np.random.rand(ndim,R_U,1) for ndim in ndims]

    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if 'CP' in args.method:
        # CP-based model
        gamma_size = R_U 
    elif 'Tucker' in args.method:
        # Tucker-based model
        gamma_size = np.prod([R_U for k in range(nmod)]) # R^K
    # package the hypyerparas
    hyper_para_dict = {}

    hyper_para_dict['U'] = U
    hyper_para_dict['R_U'] = R_U
    hyper_para_dict['device'] = device

    hyper_para_dict['train_ind'] = train_ind
    hyper_para_dict['train_y'] = train_y
    # hyper_para_dict['train_time'] = train_time

    hyper_para_dict['test_ind'] = test_ind
    hyper_para_dict['test_y'] = test_y
    # hyper_para_dict['test_time'] = test_time

    hyper_para_dict['ndims'] = ndims
    hyper_para_dict['batct_size'] = args.batch_size
    hyper_para_dict['lr'] = args.lr
    hyper_para_dict['gamma_size'] = gamma_size

    hyper_para_dict['v'] = 1
    hyper_para_dict['v_time'] = 1

    return hyper_para_dict

class dataset_SVI_base(Dataset):
    def __init__(self,x_id,y):
        self.x = x_id
        self.y = y
        self.length = len(self.y)

    def __getitem__(self, index):
        return self.x[index,:],self.y[index]
    
    def __len__(self):
        return self.length 

def data_loader_simu_base(args,data_file_name):
    
    
    data = np.load(data_file_name, allow_pickle=True)

    train_ind = data.item().get('train_ind')
    train_y = data.item().get('train_y')
    train_time = data.item().get('train_time_uni') # sorted & unique timestamps
    train_time_ind = data.item().get('train_time_ind')


    test_ind = data.item().get('test_ind')
    test_y = data.item().get('test_y')
    test_time = data.item().get('test_time_uni')
    test_time_ind = data.item().get('test_time_ind')

    ndims = data.item().get('ndims')
    nmod = len(ndims)
    N = train_y.size
    N_time = train_time.size

    nepoch = args.epoch

    R_U = args.R_U

    if args.device == 'cpu':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      

    if 'CP' in args.method:
        # CP-based model
        gamma_size = R_U 
        # U = [3*torch.rand(ndim,R_U).double() for ndim in ndims]
    elif 'Tucker' in args.method:
        # Tucker-based model
        gamma_size = np.prod([R_U for k in range(nmod)]) # R^K
        # U = [torch.rand(ndim,R_U).double() for ndim in ndims]
    else:
        raise Exception("only support CP/Tucker based method")


    if 'LDS' in args.method:
        time_data_table = utils.build_time_data_table(train_time_ind)
    else:
        time_data_table = None


    # array to tensor
    U = [torch.rand(ndim,R_U).double() for ndim in ndims]
    # U = [torch.randn(ndim,R_U).double() for ndim in ndims]

    # if args.dataset == 'dblp':
    #     U = [3*torch.randn(ndim,R_U).double() for ndim in ndims]

    train_time = torch.tensor(train_time)
    test_time = torch.tensor(test_time)

    if args.fix_int:
        # fix-time-interval setting
        fix_int = torch.abs(train_time[1]-train_time[0]).squeeze()
        time_int_list = fix_int*torch.ones(N) 
    else:
        # non-fix-time-interval setting, compute the gap between each two time-stamps
        fix_int = None
        time_int_list_follow = [train_time[i+1] - train_time[i] for i in range(N_time-1)]
        time_int_list =torch.tensor([0.0] + time_int_list_follow)

    train_y = torch.tensor(train_y).reshape(-1,1)
    test_y = torch.tensor(test_y).reshape(-1,1)


    # hyper-paras setting for LDS system
    lengthscale = torch.tensor(args.ls)
    variance = torch.tensor(0.1)
    noise = torch.tensor(0.1)

    # matern 32 kernel
    lamb = np.sqrt(3)/lengthscale
    F_base = torch.tensor([[0,1.0],
                    [-lamb*lamb,-2*lamb]])
    P_inf_base = torch.tensor([[variance,0.0],
                        [0,lamb*lamb*variance]])
    H_base = torch.tensor([[1.0,0]])

    F_list =[F_base for i in range(gamma_size)]
    P_inf_list =[P_inf_base for i in range(gamma_size)]
    H_list = [H_base for i in range(gamma_size)]

    F = torch.block_diag(*F_list)
    P_inf = torch.block_diag(*P_inf_list)
    H = torch.block_diag(*H_list)


    m_0 = torch.zeros(2*gamma_size,1)
    P_0 = P_inf

    # package the hypyerparas

    hyper_para_dict = {}
    hyper_para_dict['N'] = N
    hyper_para_dict['F'] = F
    hyper_para_dict['H'] = H
    hyper_para_dict['R'] = noise
    hyper_para_dict['P_inf'] = P_inf
    hyper_para_dict['fix_int'] = fix_int # set as None if it's not fix 
    hyper_para_dict['m_0'] = m_0
    hyper_para_dict['P_0'] = P_0

    hyper_para_dict['U'] = U
    hyper_para_dict['device'] = device
    hyper_para_dict['ind_tr'] = train_ind
    hyper_para_dict['y_tr'] = train_y

    hyper_para_dict['test_ind'] = test_ind
    hyper_para_dict['test_y'] = test_y

    hyper_para_dict['epoch'] = nepoch
    hyper_para_dict['R_U'] = R_U
    hyper_para_dict['v'] = 1
    hyper_para_dict['a0'] = 10.0
    hyper_para_dict['b0'] = 1.0
    hyper_para_dict['time_int_list'] = time_int_list
    hyper_para_dict['ndims'] = ndims
    hyper_para_dict['train_time'] = train_time
    hyper_para_dict['test_time'] = test_time

    hyper_para_dict['train_time_ind'] = train_time_ind
    hyper_para_dict['test_time_ind'] = test_time_ind
    hyper_para_dict['N_time'] = N_time
    hyper_para_dict['time_data_table'] = time_data_table

    hyper_para_dict['gamma_size'] = gamma_size

    hyper_para_dict['DAMPPING_U'] = args.DAMPPING_U
    hyper_para_dict['DAMPPING_gamma'] = args.DAMPPING_gamma

    return hyper_para_dict