import os
import networkx as nx
import dgl
import re
import pandas as pd

def get_network_graph(name, basedir):
    s = os.path.join(basedir, f'{name}.graphml')
    g = nx.read_graphml(s)
    return g


def get_dgl_graph(name, basedir):
    s = os.path.join(basedir, f'{name}.graph.bin')
    g = dgl.load_graphs(s)[0][0]
    return g

def get_inst_from_verilog(name, basedir):
    s = os.path.join(basedir, f'{name}.v')
    with open(s, 'r') as file:
        netlist = file.read()
        file.close()
    # read a dictionary of instance name and instance type
    pattern = r'sky130_fd_sc_hd__(\w+)_(\w+) (_\w+_)'
    new_sizes = {m[2]: m[:2] for m in re.findall(pattern, netlist)}
    return new_sizes

def get_inst_names(nx_graph, dgl_graph):
    # get ids of virtual nodes (instances)
    ids = dgl_graph.nodes['virtual_node'].data['id']
    # get id -> name map
    inst_table = {attributes.get('id'): node for node, attributes in nx_graph.nodes(data=True) if
             attributes.get('_TYPE') == 1}
    # get names list
    names_list = [inst_table[n] for n in ids.tolist()]
    return names_list

def get_inst_sizes(cell_size_ind, dgl_graph, libdata):
    cell_type_ind = dgl_graph.nodes['virtual_node'].data['feat1'].tolist()
    cell_sizes = [libdata['size_list'][libdata['cell_types'][ci]][si] for ci, si in
                      zip(cell_type_ind, cell_size_ind)]
    return cell_sizes


def make_table(list_name, map_init, map_dc, list_dds):
    # index     inst_name   inst_type   Initial size    DC size     DDS size    #
    table = {}
    for i, n in enumerate(list_name):
        rec = {'Instance name': n,
               'Instance type': map_init[n][0],
               'Initial size': map_init[n][1],
               'DC opt size': map_dc[n][1],
               'DDS opt size': list_dds[i]
               }
        table[i] = rec
    return table

def store_table(filename, table):
    df = pd.DataFrame.from_dict(table, orient='index')
    df.to_csv(filename, index_label='Index')


'''
Usage example

g = get_dgl_graph(benchname, '/work/tayyeb/dds_related/branch/sa/benchmarks/dgl')

inst_map_dc = get_inst_from_verilog(benchname, '/work/tayyeb/dds_related/branch/sa/benchmarks/dc_results/netlists')
inst_map_inp = get_inst_from_verilog(benchname, '/work/tayyeb/dds_related/branch/sa/benchmarks/verilog')

network = get_network_graph(benchname, '/work/tayyeb/dds_related/branch/sa/benchmarks/network')
inst_names = get_inst_names(network, g)

# main should return the new_size tensor, corresponding to best wns, or any point of interest
best_sizes = main(benchname, g)

opt_cell_sizes = get_inst_sizes(best_sizes.tolist(), g, libdump)

table = make_table(inst_names, inst_map_inp, inst_map_dc, opt_cell_sizes)
store_table('data.csv', table)

'''








