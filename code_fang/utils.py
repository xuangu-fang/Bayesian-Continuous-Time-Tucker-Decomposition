import numpy as np
import torch
# import bisect
import argparse

def parse_args_CEP_based_model():
    
    description = "SDE-GP dynamic tensor factorization"                   
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument('--epoch', type=int,default=20,help='number of training epoch')
    parser.add_argument('--R_U', type=int,default=5,help='dim of mode embeddings')
    parser.add_argument('--num_fold', type=int,default=1,help='number of folds(random split) and take average,min:1,max:5')
    parser.add_argument('--fix_int', type=int,default=False,help='whether timestamps are with fix interal')
    parser.add_argument('--ls', type=float,default=0.1,help='length-scale of LDS system')
    parser.add_argument('--DAMPPING_U', type=float,default=0.95,help='Damping ratio of U undate')
    parser.add_argument('--DAMPPING_gamma', type=float,default=0.5,help='Damping ratio of gamma undate')
    parser.add_argument('--expand_odrer', type=str,default='two',help='one or two order based expantion of Expecatation')
    parser.add_argument('--device', type=str,default='cpu',help='cpu or gpu')
    parser.add_argument('--machine', type=str,default='zeus',help='machine_name')
    parser.add_argument('--method', type=str,default='LDS-CP',help='methods name:Tucker/CP-LDS')
    parser.add_argument('--dataset', type=str,default='mvlens',help='dataset name: mvlens')
    
    return parser.parse_args()

def parse_args_GPTF_based_model():
    
    description = "(sparse)GP based tensor factorization"                   
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument('--epoch', type=int,default=20,help='number of training epoch')
    parser.add_argument('--R_U', type=int,default=5,help='dim of mode embeddings')
    parser.add_argument('--num_fold', type=int,default=1,help='number of folds(random split) and take average,min:1,max:5')
    parser.add_argument('--m', type=int,default=100,help='number of pseudo input')
    parser.add_argument('--batch_size', type=int,default=248,help='batch size for SVI')
    parser.add_argument('--lr', type=float,default=1e-3,help='learning rate')
    parser.add_argument('--device', type=str,default='cpu',help='cpu or gpu')
    parser.add_argument('--machine', type=str,default='zeus',help='machine_name')
    parser.add_argument('--method', type=str,default='GPTF-time',help='methods name:GPTF-time/static/discrete')
    parser.add_argument('--dataset', type=str,default='mvlens',help='dataset name: mvlens, ufo, or twists')
    
    return parser.parse_args()

def parse_args_SVI_based_model():
    
    description = "(sparse)GP based tensor factorization"                   
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument('--epoch', type=int,default=20,help='number of training epoch')
    parser.add_argument('--R_U', type=int,default=5,help='dim of mode embeddings')
    parser.add_argument('--num_fold', type=int,default=1,help='number of folds(random split) and take average,min:1,max:5')
    parser.add_argument('--batch_size', type=int,default=1024,help='batch size for SVI')
    parser.add_argument('--lr', type=float,default=3e-3,help='learning rate')
    parser.add_argument('--device', type=str,default='cpu',help='cpu or gpu')
    parser.add_argument('--machine', type=str,default='zeus',help='machine_name')
    parser.add_argument('--method', type=str,default='CP-SVI',help='methods name:CP-SVI or Tucker-SVI')
    parser.add_argument('--dataset', type=str,default='mvlens',help='dataset name: mvlens, ufo, or twists')
    
    return parser.parse_args()

def make_log(args,result_dict):
    fname = "result_log/"+ args.dataset + '_' + args.method + '_' + args.machine +".txt"
    f= open(fname,"a+")
    f.write('\n take %.1f seconds to finish %d folds. avg time: %.1f seconds'%(result_dict['time'],args.num_fold,result_dict['time']/args.num_fold))
    f.write('\n Setting:R_U = %d, epoch = %d, ls = %.2f,  \n'\
    %(args.R_U, args.epoch,args.ls))
    f.write('\n final test RMSE, avg is %.4f, std is %.4f \n'%(result_dict['rmse_avg'],result_dict['rmse_std']))
    f.write('\n final test MAE, avg is %.4f, std is %.4f \n'%(result_dict['MAE_avg'],result_dict['MAE_std']))
    f.write('\n\n\n')
    f.close()

def make_log_GPTF(args,result_dict):
    fname = "result_log/"+ args.dataset + '_' + args.method + '_' + args.machine +".txt"
    f= open(fname,"a+")
    f.write('\n take %.1f seconds to finish %d folds. avg time: %.1f seconds'%(result_dict['time'],args.num_fold,result_dict['time']/args.num_fold))
    f.write('\n Setting:R_U = %d, epoch = %d, lr = %.3f, batch_size = %d, m = %d \n'\
    %(args.R_U, args.epoch,args.lr,args.batch_size,args.m))
    f.write('\n final test RMSE, avg is %.4f, std is %.4f \n'%(result_dict['rmse_avg'],result_dict['rmse_std']))
    f.write('\n final test MAE, avg is %.4f, std is %.4f \n'%(result_dict['MAE_avg'],result_dict['MAE_std']))
    f.write('\n\n\n')
    f.close()

def make_log_SVI(args,result_dict):
    fname = "result_log/"+ args.dataset + '_' + args.method + '_' + args.machine +".txt"
    f= open(fname,"a+")
    f.write('\n take %.1f seconds to finish %d folds. avg time: %.1f seconds'%(result_dict['time'],args.num_fold,result_dict['time']/args.num_fold))
    f.write('\n Setting:R_U = %d, epoch = %d, lr = %.3f, batch_size = %d,  \n'\
    %(args.R_U, args.epoch,args.lr,args.batch_size))
    f.write('\n final test RMSE, avg is %.4f, std is %.4f \n'%(result_dict['rmse_avg'],result_dict['rmse_std']))
    f.write('\n final test MAE, avg is %.4f, std is %.4f \n'%(result_dict['MAE_avg'],result_dict['MAE_std']))
    f.write('\n\n\n')
    f.close()

# def bisect_search(nuclears, x):
#         idx = bisect.bisect_right(nuclears, x)
#         return idx

# batch knorker product  
def kronecker_product_einsum_batched(A: torch.Tensor, B: torch.Tensor): 
    """ 
    Batched Version of Kronecker Products 
    :param A: has shape (b, a, c) 
    :param B: has shape (b, k, p) 
    :return: (b, ak, cp) 
    """ 
    assert A.dim() == 3 and B.dim() == 3 

    res = torch.einsum('bac,bkp->bakcp', A, B).view(A.size(0), 
                                                    A.size(1)*B.size(1), 
                                                    A.size(2)*B.size(2) 
                                                    ) 
    return res 

def Hadamard_product_batch(A: torch.Tensor, B: torch.Tensor):
    """ 
    Batched Version of Hadamard Products 
    :param A: has shape (N, a, b) 
    :param B: has shape (N, a, b) 
    :return: (N, a, b) 
    """ 
    assert A.dim() == 3 and B.dim() == 3 
    assert A.shape == B.shape
    res = A*B
    return res

def build_time_data_table(time_ind):
    # input: sorted time-stamp seq (duplicated items exists) attached with data seq 
    # output: table (list) of associated data points of each timestamp
    # ref: https://stackoverflow.com/questions/38013778/is-there-any-numpy-group-by-function/43094244
    # attention, here the input "time-stamps" can be either (repeating) id, or exact values, but seq length must match data seq
    # in out table, order of item represents the time id in order 
    time_data_table = np.split(np.array([i for i in range(len(time_ind))]),np.unique(time_ind,return_index=True)[1][1:])
    return time_data_table


def build_id_key_table(nmod,ind):
    # build uid-data_key_table, implement by nested list
    
    # given indices of unique rows of each mode/embed (store in uid_table)  
    uid_table = []
    
    # we could index which data points are associated through data_table
    data_table = []

    for i in range(nmod):
        values,inv_id = np.unique(ind[:,i],return_inverse=True)
        uid_table.append(list(values))

        sub_data_table = []
        for j in range(len(values)):
            data_id = np.argwhere(inv_id==j)
            if len(data_id)>1:
                data_id = data_id.squeeze().tolist()
            else:
                data_id = [[data_id.squeeze().tolist()]]
            sub_data_table.append(data_id)
            
        data_table.append(sub_data_table)
        
    return uid_table,data_table
