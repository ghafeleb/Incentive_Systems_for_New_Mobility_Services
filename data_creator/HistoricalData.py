import sys
sys.path.insert(0, '../')
import time
from copy import deepcopy
import numpy as np
import datetime
import os
import networkx as nx
import pickle
from collections import OrderedDict
from lib.baseNCST1 import *
from joblib import Parallel, delayed
import argparse
import yaml
from lib.utils import check_exists, AM_PM_f, hr_str


def main(args):
    with open(args.config_filename, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    region_ = config_dict['region_']
    fileNameSetting1 = config_dict['fileNameSetting1']
    interval_t = config_dict['interval_t']
    n_hours = config_dict['n_hours']
    AM_PM = config_dict['AM_PM']
    start_hour = config_dict['start_hour']
    finish_hr = config_dict['finish_hr']
    pad_hr = config_dict['pad_hr']
    num_paths = config_dict['num_paths']
    month_dict = config_dict['month_dict']
    start_hour = config_dict['start_hour']
    data_year = config_dict['data_year']
    OD_add = config_dict['OD']
    col_width = max(len(word) for word in config_dict.keys()) + 4  # padding
    
    for a, b in config_dict.items():
        print "".join([a.ljust(col_width), str(b)])

    ADMM_address0 = os.path.join('../data', region_, fileNameSetting1)
    check_exists(add_=ADMM_address0, create_=True)
    ADMM_address1 = os.path.join('../data', region_, fileNameSetting1, 'initialData')
    check_exists(add_=ADMM_address1, create_=True)

    if pad_hr:
        pad_str = '_pad'
        start_hr_w_pad = start_hour - pad_hr  # 5 AM
    else:
        pad_str = ''
        start_hr_w_pad = start_hour

    time_basis_w_pad = datetime.time(start_hr_w_pad, 0, 0)  # !! 3 AM
    time_interval = datetime.timedelta(minutes=interval_t)  # !! Time interval = 15 minutes # !!
    n_times_per_hr = int(60 / interval_t)
    N = int(60 / interval_t * n_hours)  # !!

    start_hr_str, finish_hr_str = hr_str(start_hour, finish_hr, AM_PM)

    ##### Create ROW indecies of dataframe in Pandas
    # Number of days for each ID
    num_days = sum([len(x) for x in month_dict.itervalues()])
    date_need_to_finish = list()
    for iter_month in month_dict.keys():
        for iter_day in month_dict[iter_month]:
            # print '\n'
            date_temp = datetime.date(2018, iter_month, iter_day)  # !!
            # time_basis = datetime.time(3, 0, 0) # !! 3 AM
            cur_date_time = datetime.datetime.combine(date_temp, time_basis_w_pad)
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
                cur_date_time = datetime.datetime.combine(date_temp, time_basis_temp)
                # print cur_date_time
                single_date = cur_date_time.date()
                # print single_date
                date_need_to_finish.append(single_date)
        return date_need_to_finish
        
    spd_data_address = '../data/speed_volume/Mar2May_2018_' + region_ + '_AVG' + str(
        interval_t) + '_' + start_hr_str + '-' + finish_hr_str + '_with_linkID_pad/my_link_avg_spd_data_AVG' + str(
        interval_t) + 'min_' + start_hr_str + '-' + finish_hr_str + '_' + region_ + pad_str + '.pickle'
    spd_data = pd.read_pickle(spd_data_address)

    count_data_address = '../data/speed_volume/Mar2May_2018_' + region_ + '_AVG' + str(
        interval_t) + '_' + start_hr_str + '-' + finish_hr_str + '_with_linkID_pad/my_link_avg_count_data_AVG' + str(
        interval_t) + 'min_' + start_hr_str + '-' + finish_hr_str + '_' + region_ + pad_str + '.pickle'
    count_data = pd.read_pickle(count_data_address)


    print 'First 5 rows of count_data of link ' + str(count_data.keys()[0]) + ':', count_data[0].head()
    print 'First 5 rows of spd_data of link ' + str(spd_data.keys()[0]) + ':', spd_data[0].head()



    analysis_start_time2 = datetime.time(start_hour, 0, 0)  # !! Starting time = 3 AM # !!

    num_o_link = len(count_data.keys())
    o_link_list = [x for x in range(num_o_link)]
  
    start_time = time.time()
    for iter_month in month_dict.keys():
        for iter_day in month_dict[iter_month]:
            date_temp = datetime.date(2018, iter_month, iter_day)  # !!
            cur_date_time = datetime.datetime.combine(date_temp, time_basis_w_pad)
            single_date = cur_date_time.date()
            date_str = single_date.strftime("%Y-%m-%d")

            histVolume = np.zeros(num_o_link * N)
            for h in xrange(N):
                start_time = (datetime.datetime.combine(single_date, analysis_start_time2) + h * time_interval).time()
                for a, link in enumerate(o_link_list):
                    data = np.float(count_data[link].loc[single_date][start_time])
                    # print start_time, a, link, data
                    histVolume[h * num_o_link + a] = data            
            print '\n\n'
            histSpeed = np.zeros(num_o_link * N)
            for h in xrange(N):
                start_time = (datetime.datetime.combine(single_date, analysis_start_time2) + h * time_interval).time()
                for a, link in enumerate(o_link_list):
                    data = np.float(spd_data[link].loc[single_date][start_time])
                    # print start_time, a, link, data
                    histSpeed[h * num_o_link + a] = data


            np.savetxt(os.path.join(ADMM_address1, 'histSpeed.txt'), list(histSpeed), delimiter=',', fmt='%.3f')
            np.savetxt(os.path.join(ADMM_address1, 'histVolume.txt'), list(histVolume), delimiter=',', fmt='%.3f')



    od_list_address = '../data/graph_NCST1/' + region_ + '/my_od_list_' + region_ + '_original' + OD_add + '.pickle'
    graph_address = '../data/graph_NCST1/' + region_ + '/my_graph_' + region_ + '_original.gpickle'

    # with open(count_data_address, 'rb') as handle:  # !!
    #     count_data = pickle.load(handle)
    # with open(spd_data_address, 'rb') as handle:  # !!
    #     spd_data = pickle.load(handle)



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

    # fig = plt.figure()
    # plt.hist(num_path_v, edgecolor='black', linewidth=1.2)
    # plt.xlabel('Number of paths')
    # plt.ylabel('Number of OD pairs')
    # plt.title('Histogram of number of paths of OD pairs')
    # #    plt.show()
    # fig.savefig("data/plot/" + new_file + "/hist_num_paths.png")


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

    np.savetxt(os.path.join('../data', region_, 'link_loc.txt'), np.array(link_loc_list), delimiter=',', fmt='%d')
    link_loc_pd = pd.DataFrame(link_loc_list, columns=['idx', 'link_id'])
    link_loc_pd.to_csv(os.path.join('../data', region_, 'link_loc.csv'), header=True, index=False)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename',
                        default='../data/YAML/region_toy.yaml',
                        type=str,
                        help='Configuration filename for the region.')
    args = parser.parse_args()
    main(args)
