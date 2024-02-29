import numpy as np
import torch 
# import matplotlib.pyplot as plt
# from model_LDS_CP import LDS_CEP_full_v3
# from model_LDS_tucker_full_old import LDS_CEP_tucker_full_v2
from model_LDS_tucker_full_efficient import LDS_CEP_tucker_full_efficient
import os
import tqdm
import utils
import data_loader
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
JITTER = 1e-4

torch.manual_seed(2)

args = utils.parse_args_CEP_based_model()
print(args.dataset, '  ',args.method)
if args.dataset == 'mvlens':
    file_base_name = '../data/mvlens_small/mv_small_week_' #.npy'
elif args.dataset == 'ufo':
    file_base_name = '../data/ufo/ufo_week_'
elif args.dataset == 'twitch':   
    file_base_name = '../data/twitch/twitch_sub_hour_'
elif args.dataset == 'dblp':  
    file_base_name = '../data/dblp/dblp_year_'
elif args.dataset == 'ctr':  
    file_base_name = '../data/ctr/ctr_hour_'


# model_class = LDS_CEP_full_v3
# model_class = LDS_CEP_tucker_full_v2
model_class = LDS_CEP_tucker_full_efficient

result_dict = {}
rmse_test = []
MAE_test = []
time_use = []

start_time = time.time()


for fold in range(args.num_fold):
    file_name = file_base_name + str(fold) + '.npy'
    hyper_para_dict = data_loader.data_loader_CEP_base(args,file_name)

    model = model_class(hyper_para_dict)
    # model.msg_update_gamma()

    for i in tqdm.tqdm(range(args.epoch)):
    # for i in range(args.epoch):
        
        # LDS
        model.reset_list()
                
        # for k in tqdm.tqdm(range(model.N_time)):
        for k in range(model.N_time):

            model.msg_update_tau(time_id=k)
            model.filter_predict(ind=k)

            if args.expand_odrer == 'two':
                # two-order expand of conditional moment
                msg_gamma_m,msg_gamma_v = model.msg_update_gamma(time_id=k)
                model.filter_update_general(y= msg_gamma_m,R = msg_gamma_v )

            else:
                # one-order expand of conditional moment
                data_id = model.time_data_table[k]
                if len(data_id)>1:
                    # more that one data-llk 
                    R = torch.diag((model.msg_b[data_id]/model.msg_a[data_id]).squeeze())
                    z = model.E_z[data_id].squeeze(-1).T
                else:
                    data_id = data_id[0]
                    R = model.msg_b[data_id]/model.msg_a[data_id]
                    z = model.E_z[data_id]
                # print(z.shape)
                model.filter_update_simple(y= model.y_tr[data_id], R= R,z = z)

        model.smooth()
        del model.P_pred_list
        del model.P_list
        model.post_update_gamma()
        # model.reset_list()
        # gamma 
        # model.expectation_update_gamma()

        # tau
        model.post_update_tau()
        model.expectation_update_tau()
        # CEP 
        # U
        model.msg_update_U()
        model.post_update_U() 
        # model.expectation_update_z()

        if i % 2 == 0:
            loss_test_rmse,loss_test_MAE = model.model_test(hyper_para_dict['test_ind'],hyper_para_dict['test_y'],hyper_para_dict['test_time'])
            print('loss_test_rmse: %.4f,loss_test_MAE: %.4f'%(loss_test_rmse,loss_test_MAE) )

    loss_test_rmse,loss_test_MAE = model.model_test(hyper_para_dict['test_ind'],hyper_para_dict['test_y'],hyper_para_dict['test_time'])
    # loss_train,loss_test_base,loss_test_trans = model.model_test(hyper_para_dict['test_ind'],hyper_para_dict['test_y'],hyper_para_dict['test_time'])
    rmse_test.append(loss_test_rmse.cpu().numpy().squeeze())
    MAE_test.append(loss_test_MAE.cpu().numpy().squeeze())


rmse_array = np.array(rmse_test)
MAE_array = np.array(MAE_test)

result_dict['time'] = time.time() - start_time
result_dict['rmse_avg'] = rmse_array.mean()
result_dict['rmse_std'] = rmse_array.std()
result_dict['MAE_avg'] = MAE_array.mean()
result_dict['MAE_std'] = MAE_array.std()

utils.make_log(args,result_dict)