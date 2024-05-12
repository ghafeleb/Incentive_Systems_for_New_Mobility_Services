import sys
sys.path.insert(0, '../')
import networkx as nx
import os
import pandas as pd
import argparse
import yaml
import pickle
from lib.utils import check_exists, O_list_D_list, od_list_f


def main(args):
    with open(args.config_filename, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    nPath = args.nPath
    region_ = config_dict['region_']

    ODNodes = [i+1 for i in range(nPath+2)]

    # OD list
    O_list, D_list = O_list_D_list(nPath, ODNodes)
    assert(len(O_list)==len(D_list))
    nODPair = len(O_list)

    od_list = od_list_f(nPath, ODNodes, region_)
    od_list_address = '../data/graph_NCST2/' + region_ + '/my_od_list_' + region_ + '_original.pickle'
    check_exists(add_=os.path.join('../data', 'graph_NCST2'), create_=True)
    check_exists(add_=os.path.join('../data', 'graph_NCST2', region_), create_=True)
    pickle.dump(od_list, open(od_list_address, 'wb'))

    # Create graph G
    G = nx.DiGraph()
    G.add_nodes_from([i for i in ODNodes])
    length_data_address = os.path.join('../data', region_, 'link_length_meter_' + region_ + '_original.csv')
    length_data = pd.read_csv(length_data_address, header=0, index_col=0)

    for link in range(nODPair):
        print('Link ID: ', link, ', O: ', O_list[link], ', D: ', D_list[link], ', length: ', length_data.loc[link, "length_meter"], 'meters')
        G.add_edge(O_list[link], D_list[link], ID=link,
                    length=length_data.loc[link, 'length_meter'], fft=0)

    # nx.draw(G)
    # plt.show()

    graph_address = '../data/graph_NCST2/' + region_ + '/my_graph_' + region_ + '_original.gpickle'
    nx.write_gpickle(G, graph_address)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename',
                        default='../data/YAML/region_toy.yaml',
                        type=str,
                        help='Configuration filename for the region.')
    parser.add_argument('--nPath',
                        default=2,
                        type=int,
                        help='Number of paths.')
    args = parser.parse_args()
    main(args)