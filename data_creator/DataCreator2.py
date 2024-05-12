import sys
sys.path.insert(0, '../')
import numpy as np
import time
import random
import math
import networkx as nx
from lib.baseNCST2 import *
import joblib
import pandas as pd
import os
from copy import deepcopy
import yaml
import argparse
import psutil
from lib.utils import check_exists, mph2kph, mph2kph_dict

try:
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict

def print_memory():
    mem_dict = psutil.virtual_memory()._asdict()
    print("Total memory (MB): ", {mem_dict['total'] / (10 ** 6)})  # in bytes
    print("Available memory (MB): ", {mem_dict['available'] / (10 ** 6)})  # in bytes

def main(args):
    with open(args.config_filename, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    region_ = config_dict['region_']
    start_hour = config_dict['start_hour']
    start_hour_opt = config_dict['start_hour_opt']
    start_minute_opt = config_dict['start_minute_opt']
    n_times_test_horizon = config_dict['n_times_test_horizon']
    fileNameSetting1 = config_dict['fileNameSetting1']
    pad_str = config_dict['pad_str']
    interval_t = config_dict['interval_t']
    analysis_month = config_dict['analysis_month']
    analysis_day = config_dict['analysis_day']
    num_paths = config_dict['num_paths']
    theta_temp = config_dict['theta_temp']
    OD_add = config_dict['OD']
    step_size = config_dict['step_size']

    col_width = max(len(word) for word in config_dict.keys()) + 4
    for a, b in config_dict.items():
        print("".join([a.ljust(col_width), str(b)]))

    # DetStoch = args.DetStoch
    iterRun = args.iterRun+1
    # budget = args.b
    seedData = args.sD
    # seedADMM = args.sA
    # rho = args.r
    # MaxIter = args.it + args.sA
    # MaxIter = args.it
    # VOT = args.VOT
    # nC = args.nC
    # fairness = args.f
    # percC = args.percC
    # initSB = args.initSB
    # percNonU = args.percNonU
    initEta = args.initEta

    folderADMM = 'Det_MultT' + '_sD' + str(seedData) + \
                    '_ss' + str(step_size)

    # folderADMM = DetStoch+'_MultT' + '_sD' + str(seedData) +  '_nC' + str(nC) + \
    #             '_f' + fairness + '_percC' + percC + '_percNonU' + str(percNonU) +  \
    #             '_ss' + str(step_size)


    if initEta!=-1:
        ADMM_address2 = os.path.join('../data', region_, fileNameSetting1, folderADMM, 'Run_'+str(iterRun)+'_initEta')
        check_exists(add_=ADMM_address2, create_=True)
    else:
        ADMM_address2 = os.path.join('../data', region_, fileNameSetting1, folderADMM, 'Run_'+str(iterRun))
        check_exists(add_=ADMM_address2, create_=True)

    ADMM_address_new0 = os.path.join('../data', region_, fileNameSetting1, folderADMM, 'Run_'+str(iterRun+1))
    check_exists(add_=ADMM_address_new0, create_=True)
    if initEta!=-1:
        ADMM_address_new = os.path.join('../data', region_, fileNameSetting1, folderADMM, 'Run_'+str(iterRun)+'_initEta')
    else:
        ADMM_address_new = os.path.join('../data', region_, fileNameSetting1, folderADMM, 'Run_'+str(iterRun+1))
    check_exists(add_=ADMM_address_new, create_=True)


    # Check day
    analysis_date = datetime.date(random.randint(2018, 2018), random.randint(analysis_month, analysis_month), random.randint(analysis_day, analysis_day))  # !!
    n_times_per_hr = int(60 / interval_t)
    h_start_test_temp = n_times_per_hr * (start_hour_opt - start_hour) + math.floor(start_minute_opt/interval_t)
    theta_temp_str = "{:.0e}".format(theta_temp)

    OD_est_file_add_temp = os.path.join(ADMM_address2, 'POpt2018-05-01.pickle')  # 10AM-5PM
    od_list_address = os.path.join('../data', 'graph_NCST2', region_, 'my_od_list_' + region_ + '_original' + OD_add + '.pickle')
    graph_address = os.path.join('../data', 'graph_NCST2', region_, 'my_graph_' + region_ + '_original.gpickle')
    pck_address_ub_temp_csv = os.path.join('../data', 'capacity', region_, 'Mar2May_2018_new_5-22_link_capacity_' + region_ + '.csv')  # !!

    s_0_csv_address = os.path.join('../data', 'capacity', region_, 'Mar2May_2018_new_5-22_link_s_0_' + region_ + '.csv')
    link_s_0_dict0 = pd.read_csv(s_0_csv_address, header=0, index_col=0).to_dict()['speed']
    link_s_0_dict = dict()
    for i, (j, k) in enumerate(link_s_0_dict0.items()):
        link_s_0_dict[i] = mph2kph(k)

    link_cap_csv_address = os.path.join('../data', 'capacity', region_, 'Mar2May_2018_new_5-22_link_capacity_' + region_ + '.csv')
    link_cap_dict = pd.read_csv(link_cap_csv_address, header=0, index_col=0).to_dict()['capacity']

    # Load list of OD pairs
    with open(od_list_address, 'rb') as handle:  # !!
        (O_list, D_list) = pickle.load(handle, encoding='latin1')
    # Load graph
    G = nx.read_gpickle(graph_address)  # !!
    G = nx.freeze(G)

    G2 = nx.read_gpickle(graph_address)  # !!

    start_time = time.time()
    OD_paths = OrderedDict()
    # Information of all links that are traversed, set of all link classes, keys are link ids
    link_dict = OrderedDict()
    # Set of all of the Path objects, one object for each path between OD pairs
    # OD_paths also include all the paths but with specifying the OD pair of the path but in path_list all path objects are together
    path_list = list()
    for O in O_list:
        for D in D_list:
            # if not diff_idx[counter]:
            #     counter += 1
            #     continue

            # paths = list(k_shortest_paths(G, O, D, num_paths_temp)) # !! My code
            # print "From ", O, " To ", D, "there is/are ", len(paths), "path(s)"

            G_temp = deepcopy(G2)
            OD_temp = [O, D]
            paths = list()
            # path_temp = list(k_shortest_paths(G_temp, O, D, 1)) # !! My code
            # paths.append(path_temp[0])
            # if len(path_temp[0]) > 2:
            for iter_path in range(num_paths):
                try:
                    path_temp = list(k_shortest_paths(G_temp, O, D, 1))[0]  # !! My code

                    length_path = 0
                    for iter_edge in range(len(path_temp) - 1):
                        length_path += \
                        G_temp.adj[path_temp[iter_edge]][path_temp[iter_edge + 1]]['length']
                    # print('length_path:', length_path)

                    paths.append(path_temp)
                    if len(path_temp) <= 2:
                        break
                    bool_idx = [path_temp[iter] not in OD_temp for iter in range(len(path_temp))]
                    l_remove_nodes = [path_temp[iter] for iter in range(len(path_temp)) if bool_idx[iter]==True]
                    # if the list is not empty
                    if l_remove_nodes:
                        for iter_remove in range(len(l_remove_nodes)):
                            G_temp.remove_node(l_remove_nodes[iter_remove])
                except nx.NetworkXNoPath:
                    # print('No more path between ' + str(O) + ' and ' + str(D))
                    break
            # print('paths:', paths)
            # print("From ", O, " To ", D, "there is/are ", len(paths), "path(s)\n\n")

            # If the number of paths between O and D is at least 1
            if len(paths) != 0:
                # We create tmp_path_list and fill it with the path objects in the 'base.py' code
                # Next we add all these path objects for O and D to OD_paths[(O, D)]
                tmp_path_list = list()
                for path in paths:
                    # path_o is a Path object
                    path_o = Path()
                    # this path is now an attribute of Path object
                    path_o.node_list = path
                    # Constructs the
                    path_o.node_to_list(G, link_dict)
                    tmp_path_list.append(path_o)
                    path_list.append(path_o)
                # Add the list of all Path classes for the pair O & D, Each Path class includes all the attributes
                OD_paths[(O, D)] = tmp_path_list
            # counter += 1
    print((time.time() - start_time) / 60)

    num_OD = len(OD_paths)
    link_list = list(link_dict.values())

    # Sample of first 3 elements of link_dict:
    # OrderedDict([(25, < base.Link at 0x1d512b199b0 >),
    #              (55, < base.Link at 0x1d512b19a90 >),
    #              (53, < base.Link at 0x1d512b19cc0 >), ...

    # Sample of fist 3 elements of link_key_list:
    # [25,
    #  55,
    #  53, ...

    link_key_list = list(link_dict.keys())
    num_link = len(link_list)

    # .itervalues(): returns an iterator over the values of dictionary dictionary
    # vector of number of paths between each OD pair
    num_path_v = [len(x) for x in OD_paths.values()]
    num_path_v_arr = np.array(num_path_v)
    # Total number of paths
    num_path = np.sum(num_path_v)
    max_num_path = max(num_path_v)
    # Number of intervals
    assert (len(path_list) == num_path)

    pck_address_ub = pck_address_ub_temp_csv  # !!

    h_start_test = h_start_test_temp
    date_str = analysis_date.strftime("%Y-%m-%d")


    r = joblib.load(os.path.join(ADMM_address2, "R" + date_str + ".pickle")).tocsr()
    r_ = np.array(r.todense())

    # P_dict = dict()
    # try:
    #     P_dict[0] = joblib.load(os.path.join(ADMM_address2, "POpt" + date_str + "_0.pickle")).tocsr()
    # except:
    #     # P_dict[0] = joblib.load(os.path.join(folder_DPFE, "P_matrix_opt", new_file, date_str + "_0_pck.pickle")).tocsr()
    #     with open(os.path.join(ADMM_address2, "POpt" +  date_str + "_0_pck.pickle"), 'rb') as f:
    #         P_dict[0] = pickle.load(f, encoding='latin1').tocsr()
    # P_dict[0] = P_dict[0].todense()


    for iter_t in range(n_times_test_horizon):
        # print(f'iter_t: {iter_t}')
        r_2 = deepcopy(r_[num_link * (h_start_test + iter_t):num_link * (h_start_test + n_times_test_horizon),
                            (num_path * (h_start_test + iter_t)):(num_path * (h_start_test + iter_t + 1))])
        r_2 = np.array(r_2).astype("float32")
        np.savetxt(os.path.join(ADMM_address_new, 'R' + str(iter_t+1) + '_StartHour_' + str(start_hour_opt) + '_AVG' + str(
            interval_t) + pad_str + '_theta' + theta_temp_str + '.txt'), r_2, delimiter=',')
        # # np.savetxt(ADMM_address2 + '/R' + str(iter_t+1) + '_StartHour_' + str(start_hour_opt) + '_AVG' + str(
        # #     interval_t) + pad_str + '_theta' + theta_temp_str + '_' + incentive_add + '.txt', r_2, delimiter=',')
        # # del r_2
        # P_0_tempppp = deepcopy(
        #     P_dict[0][(num_path * (h_start_test + iter_t)):(num_path * (h_start_test + iter_t + 1)), :])
        # # print(P_0_tempppp[:5, :5])
        # if iter_t == 0:
        #     P_dict2 = dict()
        #     P_0_temp = deepcopy(P_dict[0][(num_path * (h_start_test + iter_t)):(num_path * (h_start_test + iter_t + 1)), :])

        #     P_dict2[0] = np.zeros_like(P_0_temp)
        #     # In this loop, we make the format of probability matrix of no incentive, similar to P matrices of positive incentives
        #     # P matrix of no incentive has one column per OD but P matrix of positive incentives has d columns for OD if it has d paths.
        #     # Iterating through the ODs
        #     path_index = list()
        #     for iter in range(len(num_path_v)):
        #         path_index.append(sum(num_path_v_arr[:iter]))
        #         # Iterating through the paths of OD
        #         for iter2 in range(num_path_v[iter]):
        #             idx = sum(num_path_v_arr[:iter])
        #             P_dict2[0][:, idx + iter2] = P_0_temp[:, iter]

        #     P_baseline = deepcopy(P_0_temp[:, :num_OD])
        #     # P_baseline_address = ADMM_address2 + '/P_baseline_StartHour_' + str(start_hour_opt) + '_AVG' + str(interval_t) + pad_str + '_theta' + theta_temp_str + '_' + incentive_add + '.pickle'
        #     # pickle.dump(P_baseline, open(P_baseline_address, "wb"))
        #     np.savetxt(os.path.join(ADMM_address_new, 'P_baseline_StartHour_' + str(start_hour_opt) + '_AVG' + str(
        #         interval_t) + pad_str + '_theta' + theta_temp_str + '.txt'), P_baseline, delimiter=',')

        #     path_index = np.array(path_index, dtype='int32')
        #     np.savetxt(os.path.join(ADMM_address_new, 'pathIndex_StartHour_' + str(start_hour_opt) + '_AVG' + str(
        #         interval_t) + pad_str + '_theta' + theta_temp_str + '.txt'), path_index, fmt='%i', delimiter=',')


        # P_temp2 = deepcopy(P_dict[0][(num_path * (h_start_test + iter_t)):(num_path * (h_start_test + iter_t + 1)), :])
        # # np.savetxt(
        # #     os.path.join(ADMM_address_new, 'P' + str(iter_t + 1) + '_StartHour_' + str(start_hour_opt) + '_AVG' + str(
        # #         interval_t) + pad_str + '_theta' + theta_temp_str + '.txt'), P_temp2, delimiter=',')

        # P2save = np.zeros_like(P_temp2)
        # for iter in range(len(num_path_v)):
        #     # Iterating through the paths of OD
        #     for iter2 in range(num_path_v[iter]):
        #         idx = sum(num_path_v_arr[:iter])
        #         P2save[:, idx + iter2] = P_temp2[:, iter]
        # np.savetxt(
        #     os.path.join(ADMM_address_new, 'P' + str(iter_t + 1) + '_StartHour_' + str(start_hour_opt) + '_AVG' + str(
        #         interval_t) + pad_str + '_theta' + theta_temp_str + '.txt'), P2save, delimiter=',')
        # # print(P2save[0, :])


        # print(f'Computing matrix A')
        # # old one
        # # A = r_2 @ P_dict2[0]
        # # New one
        # A = r_2 @ P2save
        # del P2save

        # del r_2
        # A = np.array(A).astype("float32")
        # np.savetxt(os.path.join(ADMM_address_new, 'A' + str(iter_t+1) + '_StartHour_' + str(start_hour_opt) + '_AVG' + str(
        #     interval_t) + pad_str + '_theta' + theta_temp_str + '.txt'), A, delimiter=',')

    del r_

    n_OD = num_path_v_arr.shape[0]

    # Data of travel time
    with open(os.path.join(ADMM_address2, 'tt' + date_str + "_pck.pickle"), 'rb') as f:
        tt_complete = pickle.load(f, encoding='latin1')
    tt = tt_complete[max_num_path*h_start_test:max_num_path*(h_start_test+n_times_test_horizon), :]
    # print(tt_complete.shape)
    # print(tt.shape)
    # print(max_num_path)
    # print(h_start_test)
    # print(max_num_path*h_start_test)
    # print(n_times_test_horizon)
    # print(max_num_path*(h_start_test+n_times_test_horizon))
    np.savetxt(os.path.join(ADMM_address_new, 'tt_StartHour_' + str(start_hour_opt) + '_AVG' + str(interval_t) + pad_str + '_theta' + theta_temp_str + '.txt'), tt, delimiter=',', fmt='%.3f')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename',
                        default='../data/YAML/region_toy.yaml',
                        type=str,
                        help='Configuration filename for the region.')
    parser.add_argument('--iterRun',
                        default=1,
                        type=int,
                        help='Iteration number.')
    parser.add_argument('--sD',
                        default=2,
                        type=int,
                        help='Seed of data.')
    parser.add_argument('--initEta',
                        default=-1,
                        type=int,
                        help='-1: not initEta else it is initEta')
    # parser.add_argument('--DetStoch',
    #                     default='Det',
    #                     type=str,
    #                     help='Deterministic or Stochastic setting.')
    # parser.add_argument('--b',
    #                     default=1,
    #                     type=int,
    #                     help='Budget.')
    # parser.add_argument('--sA',
    #                     default=2,
    #                     type=int,
    #                     help='Seed of ADMM.')
    # parser.add_argument('--r',
    #                     default=20,
    #                     type=int,
    #                     help='Rho.')
    # parser.add_argument('--it',
    #                     default=1000,
    #                     type=int,
    #                     help='Max # of iterations.')
    # parser.add_argument('--VOT',
    #                     default=0.46667,
    #                     type=float,
    #                     help='Value of time.')
    # parser.add_argument('--nC',
    #                     default=2,
    #                     type=int,
    #                     help='Number of companies.')
    # parser.add_argument('--f',
    #                     default="0_0_0_100_0",
    #                     type=str,
    #                     help='Fairness.')
    # parser.add_argument('--percC',
    #                     default="50_50",
    #                     type=str,
    #                     help='Percentage of each companies.')
    # parser.add_argument('--initSB',
    #                     default="T",
    #                     type=str,
    #                     help='Initialization based on baseline S matrix.')
    args = parser.parse_args()
    main(args)
