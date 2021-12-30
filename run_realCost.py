import time
from copy import deepcopy
import numpy as np
import datetime
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import pickle
from collections import OrderedDict
import joblib
from base_NCST1 import *
from joblib import Parallel, delayed
import data_loader_NCST1
# from pfe import *  # !! My code
import argparse
import yaml


def AM_PM_f(t):
    if t > 12:
        return str(t - 12) + 'PM'
    elif t == 12:
        return '12PM'
    else:
        return str(t) + 'AM'


def hr_str(start_hr, finish_hr, AM_PM):
    if AM_PM:
        return AM_PM_f(start_hr), AM_PM_f(finish_hr)
    else:
        return str(start_hr), str(finish_hr)


def check_exists(add_, create_=True):
    if not os.path.exists(add_):
        print('Does not exist: \n' + add_)
        if create_:
            print('Creating the folder...')
            os.mkdir(add_)


def main(args):
    with open(args.config_filename, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    region_ = config_dict['region_']
    fileNameSetting1 = config_dict['fileNameSetting1']
    interval_t = config_dict['interval_t']
    n_hr = config_dict['n_hr']
    AM_PM = config_dict['AM_PM']
    start_hr_0 = config_dict['start_hr_0']
    finish_hr = config_dict['finish_hr']
    pad_hr = config_dict['pad_hr']
    num_paths = config_dict['num_paths']
    theta_temp = config_dict['theta_temp']
    month_dict = config_dict['month_dict']
    OD_add = config_dict['OD']
    altered_speed = config_dict['altered_speed']
    start_hour_speed_data = config_dict['start_hour_speed_data']
    data_year = config_dict['data_year']
    step_size = config_dict['step_size']

    col_width = max(len(word) for word in config_dict.keys()) + 4  # padding
    for a, b in config_dict.items():
        print "".join([a.ljust(col_width), str(b)])

  
    iterRun = args.iterRun+1
    budget = args.b
    seedData = args.sD
    seedADMM = args.sA
    rho = args.r
    MaxIter = args.it + args.sA
    nC = args.nC
    fairness = args.f
    percC = args.percC
    initSB = args.initSB
    percNonU = args.percNonU
    nTIS = args.nTIS
    nTIE = args.nTIE

    VOT = 2.63

    if budget - int(budget) == 0:
        budget = int(budget)
 
    folderRun = ""
    folderADMM = 'Det_initAll2_MultT_b' + str(budget) + '_sD' + str(seedData) + \
                '_sA' + str(seedADMM) + '_r' + str(rho) + '_it' + str(MaxIter) + \
                '_VOT' + str(VOT) + '_nC' + str(nC) + '_f' + fairness + \
                '_percC' + percC + '_initSB_' + initSB + '_percNonU' + str(percNonU) +  \
                '_nTIS' + str(nTIS) + '_nTIE' + str(nTIE) + \
                '_ss' + str(step_size) +'_itN' + str(iterRun)

    theta_OD_Estimation_temp = theta_temp
    theta_opt_temp = theta_temp
    if pad_hr:
        pad_str = '_pad'
        start_hr = start_hr_0 - pad_hr  # 5 AM
    else:
        pad_str = ''
        start_hr = start_hr_0

    analysis_start_time = datetime.time(start_hr, 0, 0)  # !! Starting time = 3 AM # !!
    time_basis = datetime.time(start_hr, 0, 0)  # !! 3 AM
    time_interval = datetime.timedelta(minutes=interval_t)  # !! Time interval = 15 minutes # !!
    n_times_per_hr = int(60 / interval_t)
    N = int(60 / interval_t * n_hr)  # !!

    start_hr_str, finish_hr_str = hr_str(start_hr_0, finish_hr, AM_PM)

    ##### Create ROW indecies of dataframe in Pandas
    # Number of days for each ID
    num_days = sum([len(x) for x in month_dict.itervalues()])
    date_need_to_finish = list()
    for iter_month in month_dict.keys():
        for iter_day in month_dict[iter_month]:
            # print '\n'
            date_temp = datetime.date(2018, iter_month, iter_day)  # !!
            # time_basis = datetime.time(3, 0, 0) # !! 3 AM
            cur_date_time = datetime.datetime.combine(date_temp, time_basis)
            # print cur_date_time
            single_date = cur_date_time.date()
            # print single_date
            date_need_to_finish.append(single_date)

    def data_days_f(time_basis_temp, data_year, month_dict):
        date_need_to_finish = list()
        for iter_month in month_dict.keys():
            for iter_day in month_dict[iter_month]:
                # print '\n'
                date_temp = datetime.date(data_year, iter_month, iter_day)  # !!
                time_basis = time_basis_temp
                cur_date_time = datetime.datetime.combine(date_temp, time_basis)
                # print cur_date_time
                single_date = cur_date_time.date()
                # print single_date
                date_need_to_finish.append(single_date)
        return date_need_to_finish

    if altered_speed:
        sv_folder = os.path.join('data', region_, fileNameSetting1, folderADMM, folderRun)
        
        spd_data_address = os.path.join(sv_folder, 's_inc.csv')
        spd_data_raw_Det = pd.read_csv(spd_data_address, header=None, index_col=False)

        count_data_address = os.path.join(sv_folder, 'v_inc.csv')
        count_data_raw_Det = pd.read_csv(count_data_address, header=None, index_col=False)

        n_link = spd_data_raw_Det.shape[0]
        num_time = spd_data_raw_Det.shape[1]
        N = num_time
        analysis_start_time = datetime.time(start_hour_speed_data, 0, 0)  # !! Starting time = 3 AM # !!
        spd_data_Det = dict()
        count_data_Det = dict()
        t = [datetime.time(start_hour_speed_data + int(iter / n_times_per_hr), iter % n_times_per_hr * interval_t, 00) for iter in range(num_time)]  # Python 3
        time_basis_temp = datetime.time(start_hour_speed_data, 0, 0)
        date_need_to_finish = data_days_f(time_basis_temp, data_year, month_dict)
        for iter_link in range(n_link):
            # Speed data
            temp_data = pd.DataFrame(spd_data_raw_Det.iloc[iter_link, :]).T
            temp_data.index = date_need_to_finish
            temp_data.columns = t
            spd_data_Det[iter_link] = temp_data
            # Speed data
            temp_data = pd.DataFrame(count_data_raw_Det.iloc[iter_link, :]).T
            temp_data.index = date_need_to_finish
            temp_data.columns = t
            count_data_Det[iter_link] = temp_data

    else:
        spd_data_address = 'data/speed_volume/Mar2May_2018_' + region_ + '_AVG' + str(
            interval_t) + '_' + start_hr_str + '-' + finish_hr_str + '_with_linkID_pad/my_link_avg_spd_data_AVG' + str(
            interval_t) + 'min_' + start_hr_str + '-' + finish_hr_str + '_' + region_ + pad_str + '.pickle'
        spd_data = pd.read_pickle(spd_data_address)

        count_data_address = 'data/speed_volume/Mar2May_2018_' + region_ + '_AVG' + str(
            interval_t) + '_' + start_hr_str + '-' + finish_hr_str + '_with_linkID_pad/my_link_avg_count_data_AVG' + str(
            interval_t) + 'min_' + start_hr_str + '-' + finish_hr_str + '_' + region_ + pad_str + '.pickle'
        count_data = pd.read_pickle(spd_data_address)

        count_data = pd.read_pickle(count_data_address)
        spd_data = pd.read_pickle(spd_data_address)

    od_list_address = 'data/graph_NCST1/' + region_ + '/my_od_list_' + region_ + '_original' + OD_add + '.pickle'
    graph_address = 'data/graph_NCST1/' + region_ + '/my_graph_' + region_ + '_original.gpickle'

    # with open(count_data_address, 'rb') as handle:  # !!
    #     count_data = pickle.load(handle)
    # with open(spd_data_address, 'rb') as handle:  # !!
    #     spd_data = pickle.load(handle)


    print 'First 5 rows of count_data_Det of link ' + str(count_data_Det.keys()[0]) + ':', count_data_Det[0].head()
    print 'First 5 rows of spd_data_Det of link ' + str(spd_data_Det.keys()[0]) + ':', spd_data_Det[0].head()

    ### Read graph data
    with open(od_list_address, 'rb') as handle:  # !!
        (O_list, D_list) = pickle.load(handle)
    O_list = list(np.array(O_list).astype(int))
    D_list = list(np.array(D_list).astype(int))

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
                    # print('length_path:', length_path)

                    paths.append(path_temp)

                    if len(path_temp) <= 2:
                        break
                    bool_idx = [path_temp[iter] not in OD_temp for iter in range(len(path_temp))]
                    l_remove_nodes = [path_temp[iter] for iter in range(len(path_temp)) if bool_idx[iter] == True]
                    if l_remove_nodes:
                        for iter_remove in range(len(l_remove_nodes)):
                            G_temp.remove_node(l_remove_nodes[iter_remove])

                except nx.NetworkXNoPath:
                    # print 'No more path between ' + str(O) + ' and ' + str(D)
                    break
                # if O == 10082 and D == 10806:
                #     print path_temp
                #     print 'STOP'
                #     print 'STOP'
            # print 'paths:', paths
            # print "From ", O, " To ", D, "there is/are ", len(paths), "path(s)"

            # If the number of paths between O and D is at least 1
            if len(paths) != 0:
                # We create tmp_path_list and fill it with the path objects in the 'base.py' code
                # Next we add all these path objects for O and D to OD_paths[(O, D)]
                tmp_path_list = list()
                for path in paths:
                    # path_o is a Path object
                    path_o = Path();
                    # this path is now an attribute of Path object
                    path_o.node_list = path;
                    # Constructs the
                    path_o.node_to_list(G, link_dict);

                    tmp_path_list.append(path_o);
                    path_list.append(path_o);
                # Add the list of all Path classes for the pair O & D, Each Path class includes all the attributes
                OD_paths[(O, D)] = tmp_path_list;
            # counter += 1
            # if O==10082 and D==10806:
            #     print path_temp

            #     print 'STOP'
            #     print 'STOP'

    OD_paths_opt = OrderedDict()
    link_dict_opt = OrderedDict()
    path_list_opt = list()
    for O in O_list:
        for D in D_list:
            G_temp = deepcopy(G2)
            OD_temp = [O, D]
            paths_opt = list()
            for iter_path in range(num_paths):
                try:
                    path_temp_opt = list(k_shortest_paths(G_temp, O, D, 1))[0]  # !! My code
                    paths_opt.append(path_temp_opt)
                    if len(path_temp_opt) <= 2:
                        break
                    bool_idx = [path_temp_opt[iter] not in OD_temp for iter in range(len(path_temp_opt))]
                    l_remove_nodes = [path_temp_opt[iter] for iter in range(len(path_temp_opt)) if bool_idx[iter] == True]
                    if l_remove_nodes:
                        for iter_remove in range(len(l_remove_nodes)):
                            G_temp.remove_node(l_remove_nodes[iter_remove])
                except nx.NetworkXNoPath:
                    # print 'No more path between ' + str(O) + ' and ' + str(D)
                    break
                    # print 'paths:', paths
            # print "From ", O, " To ", D, "there is/are ", len(paths_opt), "path(s)"
            if len(paths_opt) != 0:
                tmp_path_list_opt = list()
                for path_opt in paths_opt:
                    path_o_opt = Path();
                    path_o_opt.node_list = path_opt;
                    path_o_opt.node_to_list(G2, link_dict);
                    tmp_path_list_opt.append(path_o_opt);
                    path_list_opt.append(path_o_opt);
                OD_paths_opt[(O, D)] = tmp_path_list_opt;
    print "Generating paths in %.2f minutes." % ((time.time() - start_time) / 60)


    ## Generate Delta
    # Number of OD pairs
    num_OD = len(OD_paths)
    # print "Number of OD pairs: ", num_OD
    link_list = list(link_dict.values())
    num_link = len(link_list)
    # print "Number of links: ", num_link
    # .itervalues(): returns an iterator over the values of dictionary dictionary
    # vector of number of paths between each OD pair
    num_path_v = [len(x) for x in OD_paths.itervalues()]
    print num_path_v

    # Total number of paths
    num_path = np.sum(num_path_v)
    # print "Number of paths: ", num_path
    assert (len(path_list) == num_path)


    # The delta matrix with bianry elements
    delta = np.zeros((num_link, num_path))
    # Iterate over links (edges)
    for i, link in enumerate(link_list):
        # Iterate over paths
        for j, path in enumerate(path_list):
            # If the path includes the link (edge), we change the element of the matrix to 1
            if link in path.link_list:
                delta[i, j] = 1.0

    link_loc = dict()
    link_loc_list = []
    for idx, link in enumerate(link_list):
        # print link.ID
        link_loc[link] = idx
        link_loc_list.append([link.ID, idx])

    # ############################################################################
    ## Construct travel time matrix
    start_time = time.time()
    for tmp_date in date_need_to_finish:
        Parallel(n_jobs=1)(delayed(save_tt_joblib)(N, spd_data_Det, analysis_start_time, time_interval,
                                            tmp_date, path_list_opt, OD_paths_opt, region_, fileNameSetting1, folderADMM, folderRun)
                        for tmp_date in date_need_to_finish)
    t_ = (time.time() - start_time) / 60
    print "save_tt: %.2f minutes" % t_


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename',
                        default='data/YAML/region_toy_realCost.yaml',
                        type=str,
                        help='Configuration filename for the region.')
    parser.add_argument('--iterRun',
                        default=1,
                        type=int,
                        help='Iteration number.')
    parser.add_argument('--b',
                        default=1,
                        type=float,
                        help='Budget.')
    parser.add_argument('--sD',
                        default=2,
                        type=int,
                        help='Seed of data.')
    parser.add_argument('--sA',
                        default=2,
                        type=int,
                        help='Seed of ADMM.')
    parser.add_argument('--r',
                        default=20,
                        type=int,
                        help='Rho.')
    parser.add_argument('--it',
                        default=1000,
                        type=int,
                        help='Max # of iterations.')
    parser.add_argument('--nC',
                        default=2,
                        type=int,
                        help='Number of companies.')
    parser.add_argument('--f',
                        default="0_0_0_100_0",
                        type=str,
                        help='Fairness.')
    parser.add_argument('--percC',
                        default="50_50",
                        type=str,
                        help='Percentage of each companies.')
    parser.add_argument('--initSB',
                        default="T",
                        type=str,
                        help='Initialization based on baseline S matrix.')
    parser.add_argument('--percNonU',
                        default=90,
                        type=int,
                        help='Percentage of nonuser drivers.')
    parser.add_argument('--nTIS',
                        default=13,
                        type=int,
                        help='Starting time of incentivization')
    parser.add_argument('--nTIE',
                        default=24,
                        type=int,
                        help='Ending time of incentivization')
    parser.add_argument('--folderADMM',
                        default="Det",
                        type=str,
                        help='Folder of incentivization policy')
    args = parser.parse_args()
    main(args)
