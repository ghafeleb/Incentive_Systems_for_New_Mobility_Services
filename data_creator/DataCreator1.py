# from __future__ import print_function
import sys
sys.path.insert(0, '../')
import numpy as np
from lib.baseNCST2 import *
import time
import random
import math
import networkx as nx
import joblib
import pandas as pd
import os
from copy import deepcopy
import yaml
import argparse
import pickle
import psutil
from lib.utils import check_exists, mph2kph, mph2kph_dict, m2km

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
    n_hours = config_dict['n_hours']
    pad_hr = config_dict['pad_hr']
    pad_str = config_dict['pad_str']
    n_times_test_horizon = config_dict['n_times_test_horizon']
    interval_t = config_dict['interval_t']
    analysis_month = config_dict['analysis_month']
    analysis_day = config_dict['analysis_day']
    num_paths = config_dict['num_paths']
    new_file_temp = config_dict['new_file_temp']
    theta_temp = config_dict['theta_temp']
    folder_DPFE = config_dict['folder_DPFE']
    fileNameSetting1 = config_dict['fileNameSetting1']
    OD_add = config_dict['OD']
    col_width = max(len(word) for word in config_dict.keys()) + 4  # padding
    for a, b in config_dict.items():
        print("".join([a.ljust(col_width), str(b)]))


    # Check day
    analysis_date = datetime.date(random.randint(2018, 2018), random.randint(analysis_month, analysis_month), random.randint(analysis_day, analysis_day))  # !!
    n_times_per_hr = int(60 / interval_t)
    # h_start_test_temp = n_times_per_hr * (start_hour_opt - start_hour) + math.floor(start_minute_opt/interval_t)
    h_start_test_temp = n_times_per_hr * (start_hour_opt - start_hour) + math.floor(start_minute_opt/interval_t)
    num_interval = int(n_times_per_hr * (n_hours+pad_hr))  # 7 hours, 12 5-minute intervals in each hour >> data 10AM-5PM
    analysis_start_time_temp = datetime.time(start_hour, 0, 0)  # !! Starting time = 3 AM # !!
    time_interval_temp = datetime.timedelta(minutes=interval_t)  # !! Time interval = 5 minutes
    # test_horizon = n_hr_test_horizon*n_times_per_hr
    test_horizon = n_times_test_horizon
    theta_temp_str = "{:.0e}".format(theta_temp)

    ADMM_address1 = os.path.join('../data', region_,)
    check_exists(add_=ADMM_address1, create_=True)
    ADMM_address2 = os.path.join('../data', region_, fileNameSetting1)
    # ADMM_address2 = 'data/ADMM_matrix_NCST2/' + region_ + '/' + incentive_add + '_th_' + str(theta_temp)
    check_exists(add_=ADMM_address2, create_=True)
    ADMM_address_MATLAB = os.path.join('../data', region_, fileNameSetting1, 'initialData')
    check_exists(add_=ADMM_address_MATLAB, create_=True)
    OD_est_file_add_temp = folder_DPFE + '/Q_vector/' + new_file_temp + '/python3/2018-05-01.pickle'  # 10AM-5PM
    od_list_address = '../data/graph_NCST2/' + region_ + '/my_od_list_' + region_ + '_original' + OD_add + '.pickle'
    graph_address = '../data/graph_NCST2/' + region_ + '/my_graph_' + region_ + '_original.gpickle'


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
            G_temp = deepcopy(G2)
            OD_temp = [O, D]
            paths = list()
            for iter_path in range(num_paths):
                try:
                    path_temp = list(k_shortest_paths(G_temp, O, D, 1))[0]  # !! My code

                    length_path = 0
                    for iter_edge in range(len(path_temp) - 1):
                        length_path += \
                        G_temp.adj[path_temp[iter_edge]][path_temp[iter_edge + 1]]['length']
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

    # %%
    # ################################################ OD estimation ######################################
    start_time = time.time()
    OD_est_file_add = OD_est_file_add_temp

    with open(OD_est_file_add, 'rb') as f:
        q0 = pickle.load(f, encoding="latin1")
        # q_temp = np.array([math.floor(q_element) for q_element in q0])
        q_temp1 = np.array([round(q_element) for q_element in q0])

    q_temp2 = q_temp1.astype(int)

    # ? hours
    # Row >> time intervals, columns >> OD pairs
    q_temp3 = q_temp2.reshape(num_interval, -1)
    q = pd.DataFrame(q_temp3, copy=True)

    ##### Create ROW indecies of dataframe in Pandas
    month_dict = dict()
    # Business days of May 2018
    # month_dict[5] = [1, 2, 3, 4, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 29, 30, 31]  # !!
    month_dict[analysis_month] = [analysis_day]  # !!
    analysis_start_time = analysis_start_time_temp
    time_interval = time_interval_temp
    date_need_to_finish = list()
    for iter_month in month_dict.keys():
        for iter_day in month_dict[iter_month]:
            for h in range(num_interval):
                date_temp = datetime.date(2018, iter_month, iter_day)
                time_basis = (datetime.datetime.combine(date_temp, analysis_start_time) + h * time_interval).time()
                cur_date_time = datetime.datetime.combine(date_temp, time_basis)
                # single_date = cur_date_time.date()
                date_need_to_finish.append(cur_date_time)
    q.index = date_need_to_finish

    h_start_test = h_start_test_temp
    # q for only 12 horizons
    date_need_to_finish_test = list()
    for iter_month in month_dict.keys():
        for iter_day in month_dict[iter_month]:
            date_temp = datetime.date(2018, iter_month, iter_day)  # !!
            time_basis = (datetime.datetime.combine(date_temp, analysis_start_time) + (h_start_test) * time_interval).time()
            cur_date_time = datetime.datetime.combine(date_temp, time_basis)
            print(cur_date_time)
            # single_date = cur_date_time.date()
            date_need_to_finish_test.append(cur_date_time)
    q_test_single_time = q.loc[date_need_to_finish_test, :]
    q_test_single_time = q_test_single_time.values.reshape((-1, 1)).astype("float32")

    time_temp_ = (time.time() - start_time) / 60
    print('Time of processing OD estimation vector (q): %f minutes' % time_temp_)
    # ################################################## optimization model ###########################################
    print_memory()
    print('1')
    h_start_test = h_start_test_temp
    date_str = analysis_date.strftime("%Y-%m-%d")
    new_file = new_file_temp

    r = joblib.load(os.path.join(folder_DPFE, "R_matrix", new_file, date_str + ".pickle")).tocsr()
    r_ = np.array(r.todense())
    print_memory()
    print('2')

    print_memory()
    print('4')
    for iter_t in range(test_horizon):
        # print(f'iter_t: {iter_t}')
        r_2 = deepcopy(r_[num_link * (h_start_test + iter_t):num_link * (h_start_test + test_horizon),
                            (num_path * (h_start_test + iter_t)):(num_path * (h_start_test + iter_t + 1))])
        r_2 = np.array(r_2).astype("float32")
        np.savetxt(ADMM_address_MATLAB + '/R' + str(iter_t+1) + '_StartHour_' + str(start_hour_opt) + '_AVG' + str(
            interval_t) + pad_str + '_theta' + theta_temp_str + '.txt', r_2, delimiter=',')
        h_start_test_temp2 = h_start_test_temp + iter_t
        # q for only 12 horizons
        date_need_to_finish_test_temp = list()
        for iter_month in month_dict.keys():
            for iter_day in month_dict[iter_month]:
                date_temp = datetime.date(2018, iter_month, iter_day)  # !!
                time_basis = (datetime.datetime.combine(date_temp, analysis_start_time) + (
                    h_start_test_temp2) * time_interval).time()
                cur_date_time = datetime.datetime.combine(date_temp, time_basis)
                # print(cur_date_time)
                # single_date = cur_date_time.date()
                date_need_to_finish_test_temp.append(cur_date_time)
        q_test_single_time_temp = q.loc[date_need_to_finish_test_temp, :]
        q_test_single_time_temp = q_test_single_time_temp.values.reshape((-1, 1)).astype("float32")
        np.savetxt(ADMM_address_MATLAB + '/q' + str(iter_t+1) + '_StartHour_' + str(start_hour_opt) + '_AVG' + str(interval_t) + pad_str + '_theta' + theta_temp_str + '.txt', q_test_single_time_temp, delimiter=',', fmt='%i')
        del q_test_single_time_temp
    del r_

    print_memory()
    print('7-2')

    n_OD = num_path_v_arr.shape[0]

    D_temp = np.zeros((n_OD*test_horizon, num_path*test_horizon))
    for iter_time in range(test_horizon):
        for iter_OD in range(n_OD):
            D_temp[iter_OD + iter_time*n_OD, (sum(num_path_v_arr[:iter_OD])+iter_time*num_path):(sum(num_path_v_arr[:iter_OD+1])+iter_time*num_path)] = 1
    # D = np.tile(D_temp, n_incentive).astype("float32")
    np.savetxt(ADMM_address_MATLAB + '/D_StartHour_' + str(start_hour_opt) + '_AVG' + str(interval_t) + pad_str + '_theta' + theta_temp_str + '.txt', D_temp, delimiter=',', fmt='%i')

    # Data of travel time
    np.savetxt(ADMM_address_MATLAB + '/num_path_v.txt', num_path_v, delimiter=',', fmt='%i')
    with open(os.path.join(folder_DPFE, "tt", new_file, date_str + "_pck.pickle"), 'rb') as f:
        tt_complete = pickle.load(f, encoding='latin1')
    tt = tt_complete[max_num_path*h_start_test:max_num_path*(h_start_test+test_horizon), :]
    np.savetxt(ADMM_address_MATLAB + '/tt_StartHour_' + str(start_hour_opt) + '_AVG' + str(interval_t) + pad_str + '_theta' + theta_temp_str + '.txt', tt, delimiter=',', fmt='%.3f')
    print_memory()
    print('9')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename',
                        default='../data/YAML/region_toy.yaml',
                        type=str,
                        help='Configuration filename for the region.')
    args = parser.parse_args()
    main(args)
