import dgl
import torch
import torch.optim as optim
import dgl.function as fn
import networkx as nx
import pickle as pk
import numpy as np
import torch.nn.functional as F
import re
from steiner_tree import steiner_tree
from propagate import *
import utils
import lib
import plotly.graph_objects as go
from def_parser import *
from gradient_collector import GradientCollector
# Enable anomaly detection to trace in-place operation issues
torch.autograd.set_detect_anomaly(True)






sdc_set_input_slew = 0.012
sdc_set_input_delay = 0.1
sdc_set_load_all_outputs = 17.65/1000
global sdc_create_clock_period
sdc_create_clock_period = 1.0
sdc_clock_period_scaler = 0.8482



parser = lib.LibertyParser()
parser.parse_file('./data/sky130_fd_sc_hd__tt_025C_1v80.lib')
wire_load_models = parser.parse_cap_res()['medium']
unit_cap = wire_load_models[0]
unit_res = wire_load_models[1]

delay_slew = ['n_ats', 'n_slews', 'delay']
net_attr = ['a', 'b', 'imp']
grad_attr = ['imp2_grad', 'a_grad', 'b_grad', 'slew_grad', 'delay_grad', 'load_grad', 'imp_grad']


def initialize(g, libdump, def_file_path):
    # Extract edges from 'g' to create 'ng'
    na, nb = g.edges(etype='net_out', form='uv')
    ca, cb = g.edges(etype='cell_out', form='uv')
    
    # Create 'ng' as a new DGL graph using edges from 'g'
    ng = dgl.graph((torch.cat([na, ca]), torch.cat([nb, cb])))

            
    topo = dgl.topological_nodes_generator(ng)
    levels = [ng.tolist() for ng in topo]


    ndata = g.nodes['node'].data
    na, nb = g.edges(etype='net_out', form='uv')
    ca, cb = g.edges(etype='cell_out', form='uv')
    end_nodes = [n.item() for n in nb if g.nodes['node'].data['feat1'][n] == 1]
    inst, _ = g.in_edges(end_nodes, form='uv', etype='net_shared')
    _, fins = g.out_edges(inst, form='uv', etype='net_shared')
    clks, _ = g.out_edges(fins, form='uv', etype='cell_out')
    clks = set(clks.tolist())
    
    end_nodes = set(end_nodes)
    pos = end_nodes - set(fins.tolist())
    pos = list(pos)

    ep_set = []

    for i in inst.tolist():
        _, fins = g.out_edges(i, form='uv', etype='net_shared')
        fins = set(fins.tolist())
        clk = clks & fins
        ep = end_nodes & fins
        ep_set.append(list(ep) + list(clk))

    pis = torch.tensor(levels[0], device='cpu')
    clk = pis[torch.nonzero(ndata['feat1'][pis] == 1)].flatten()

    # Get positions from DEF file


    basedir = './benchmarks/'  # Replace with the base directory of your data
    # Get instance names from networkx graph
    nx_graph = utils.get_network_graph(benchname, basedir + 'network/')
    inst_names = utils.get_inst_names(nx_graph, g)
    def_positions = parse_def_file(def_file_path)
    
    ndata = g.nodes['node'].data
    vdata = g.nodes['virtual_node'].data
    ndata['x'] = torch.zeros(g.num_nodes('node'), dtype=torch.float32)
    ndata['y'] = torch.zeros(g.num_nodes('node'), dtype=torch.float32)
    vdata['x'] = torch.zeros(g.num_nodes('virtual_node'), dtype=torch.float32)
    vdata['y'] = torch.zeros(g.num_nodes('virtual_node'), dtype=torch.float32)
    
    virtual_node_ids = g.nodes('virtual_node')

    # First: Assign positions to virtual nodes
    for inst_name, node_id in zip(inst_names, virtual_node_ids):
        x_pos, y_pos = def_positions.get(inst_name, (0.0, 0.0))
        
        # If position not found with exact name, try alternative formats
        if (x_pos, y_pos) == (0.0, 0.0):
            continue
        
        # Assign position to virtual node
        vdata['x'][node_id] = x_pos
        vdata['y'][node_id] = y_pos

    # Second: Push positions from virtual nodes to connected nodes through net_shared edges
    g['net_shared'].push(g.nodes('virtual_node'), message_func=lambda edges: {'x': edges.src['x'], 'y': edges.src['y']}, reduce_func=lambda nodes: {'x': nodes.mailbox['x'].mean(dim=1), 'y': nodes.mailbox['y'].mean(dim=1)}, etype='net_shared')
    x = torch.zeros(g.num_nodes('node'), 4)


    ndata['n_ats'] = x[:, 2] + 0
    x[:, [0, 2]] = sdc_set_input_slew 
    x[:, [1, 3]] = sdc_set_input_slew 
    ndata['n_slews'] = x[:, 2] + 0
    x.fill_(sdc_set_input_delay) #input delay for all attributes
    x[clk, [0, 2]] = 0.0 #lr clock has no input delay
    x[clk, [1, 3]] = sdc_create_clock_period/2 #lf clock has half of the clock period as input delay
    ndata['n_ats'] = x[:, 2] + 0
    x.fill_(sdc_set_load_all_outputs) #load for all attributes
    ndata['cell_type'] = torch.zeros_like(ndata['feat1'])
    g.nodes['node'].data['delay'] = torch.zeros(g.num_nodes('node'), dtype=torch.float32, requires_grad=True)
    edata = g.edges['cell_out'].data
    x = torch.zeros((g.num_edges('cell_out'), 4))

    edata['n_ats'] = x[:, 2] + 0
    edata['n_slews'] = x[:, 2] + 0
    

    g.nodes['virtual_node'].data['cell_size'] = g.nodes['virtual_node'].data['feat2'] + 0
    sizes = torch.tensor([len(libdump['size_list'][r]) for r in libdump['cell_types']])
    g.nodes['virtual_node'].data['cell_sizes'] = sizes[g.nodes['virtual_node'].data['feat1']]
    # step p1: push instance type to fanins and fanouts
    g['net_shared'].push(range(g.num_nodes('virtual_node')), lambda eb: {'cell_type': eb.src['feat1']},
                          lambda nb: {'cell_type': torch.sum(nb.mailbox['cell_type'], 1)})

    return levels, ep_set, pos





def get_res(sg):
    edges = sg.edges(etype='snet_out')
    
    # Initialize stored_grads dictionary if it doesn't exist
    if not hasattr(sg, 'stored_grads'):
        sg.stored_grads = {}
    
    def hook_factory(name):
        def hook(grad):
            if grad is not None:
                sg.stored_grads[name] = grad.clone()
                # Print gradient information
                print(f"Gradient for {name}:")
                print(f"Shape: {grad.shape}")
                print(f"Mean: {grad.mean().item():.6f}")
                print(f"Max: {grad.max().item():.6f}")
                print(f"Min: {grad.min().item():.6f}")
            return grad
        return hook
    
    sg.nodes['node'].data['x'].requires_grad_(True)
    sg.nodes['node'].data['y'].requires_grad_(True)
    
    # Create new tensors instead of modifying in-place
    caps_data = sg.nodes['node'].data['caps'].clone()  # Clone to avoid in-place modification
    res_data = torch.zeros(sg.num_edges('snet_out'), dtype=torch.float32)
    
    for i, (u, v) in enumerate(zip(*edges)):
        # Calculate Manhattan distance using smooth_abs
        wire_length = torch.sqrt((sg.nodes['node'].data['x'][u] - sg.nodes['node'].data['x'][v])**2 + (sg.nodes['node'].data['y'][u] - sg.nodes['node'].data['y'][v])**2 + 1e-6)
        
        # Calculate capacitance and resistance
        caps = wire_length * unit_cap
        res = wire_length * unit_res
        
        # Update values using out-of-place operations
        caps_data[v, 2] = caps_data[v, 2] + caps
        caps_data[v, 3] = caps_data[v, 3] + caps
        res_data[i] = res

    # Assign the new tensors back to the graph
    sg.nodes['node'].data['caps'] = caps_data
    sg.nodes['node'].data['caps'].requires_grad_(True)
    sg.nodes['node'].data['caps'].register_hook(hook_factory('capacitance'))
    
    sg.edges['snet_out'].data['res'] = res_data
    sg.edges['snet_out'].data['res'].requires_grad_(True)

def get_delay(sg, g, levels):
    # Initialize gradient collector
    collector = GradientCollector()
    
    # Initialize temporary dictionaries with gradient hooks
    for attr in delay_slew:
        # For g graph
        g.nodes['node'].data[attr].requires_grad_(True)
        g.nodes['node'].data[attr].register_hook(collector.hook_factory(f'g_{attr}'))
        sg.nodes['node'].data[attr] = torch.zeros(sg.num_nodes('node'), dtype=torch.float32, requires_grad=True)
        sg.nodes['node'].data[attr].register_hook(collector.hook_factory(f'sg_{attr}'))
    for attr in net_attr:
        sg.nodes['node'].data[attr] = torch.zeros(sg.num_nodes('node'), requires_grad=True)
        sg.nodes['node'].data[attr] = torch.zeros(sg.num_nodes('node'), dtype=torch.float32, requires_grad=True)
        sg.nodes['node'].data[attr].register_hook(collector.hook_factory(f'sg_{attr}'))
    for attr in grad_attr:
        sg.nodes['node'].data[attr] = torch.zeros(sg.num_nodes('node'), dtype=torch.float32, requires_grad=True)
        g.nodes['node'].data[attr] = torch.zeros(g.num_nodes('node'), dtype=torch.float32, requires_grad=True)
    sg.nodes['node'].data['caps_grad'] = torch.zeros(sg.num_nodes('node'), 4, dtype=torch.float32, requires_grad=True)
    g.nodes['node'].data['caps_grad'] = torch.zeros(g.num_nodes('node'), 4, dtype=torch.float32, requires_grad=True)

    # Add hooks for caps and res
    g.nodes['node'].data['caps'] = torch.zeros(g.num_nodes('node'), 4, requires_grad=True, dtype=torch.float32)
    g.nodes['node'].data['caps'].register_hook(collector.hook_factory('caps'))
    sg.nodes['node'].data['slew_grad'] = torch.ones(sg.num_nodes('node')) *1e-6
    
    
    sg.edges['snet_out'].data['res'] = torch.zeros(sg.num_edges('snet_out'), dtype=torch.float32, requires_grad=True)
    sg.edges['snet_out'].data['res'].register_hook(collector.hook_factory('res'))
    sg.nodes['node'].data['caps'] = torch.zeros(sg.num_nodes('node'),4,  requires_grad=True, dtype=torch.float32)
    sg.nodes['node'].data['caps'].retain_grad()


    # Ensure coordinates require gradients
    sg.nodes['node'].data['x'].requires_grad_(True)
    sg.nodes['node'].data['y'].requires_grad_(True)

    _, fanouts = g.edges(etype='cell_out', form='uv')
    fanouts = torch.unique(fanouts)
    _, cell_pins = g.edges(etype='net_shared', form='uv')
    _, net_outs = g.edges(etype='net_out', form='uv')
    fanins = list(set(cell_pins.tolist()) & set(net_outs.tolist()))
    
    
    

    # Process size choices
    all_cells = torch.arange(g.num_nodes('virtual_node'))
    size_param = F.one_hot(g.nodes['virtual_node'].data['cell_size'], 7) * 0.0001
    new_size = torch.argmax(size_param, dim=1)
    y = F.one_hot(new_size, num_classes=7) * 1.0
    g.nodes['virtual_node'].data['size_choice'] = y
    d = g.nodes['node'].data
    g['net_shared'].push(all_cells, fn.copy_u('size_choice', 'size_choice'), fn.sum('size_choice', 'size_choice'))
    caps_update = torch.sum(d['size_choice'][fanins].unsqueeze(-1) * cap_emb[d['cell_type'][fanins], d['feat2'][fanins]], 1)
    new_caps = d['caps'].clone()
    new_caps[fanins] = caps_update
    d['caps'] = new_caps

    # Step 2-2: Get resistance for both regular nodes and steiner nodes
    get_res(sg)
    # Step 2: Pull capacitances in fanouts
    sg.nodes['node'].data['caps'][:num_pins] = g.nodes['node'].data['caps'][:num_pins] + sg.nodes['node'].data['caps'][:num_pins]
    g['net_in'].pull(fanouts, fn.copy_u('caps', 'caps'), fn.sum('caps', 'caps'))
    for l in range(len(levels)-1,-1, -1):
        level_nodes = torch.tensor(levels[l], dtype=torch.int64)
        sg['snet_in'].pull(level_nodes, fn.copy_u('caps', 'caps'), fn.sum('caps', 'caps'))
    

    # Step 3: Propagate delay & slew across levels
    for l in range(0,len(levels)):
        level_nodes = torch.tensor(levels[l], dtype=torch.int64)
        if l % 2 != 0:  # Destination of net_out
            # Group destinations by their sources
            net_groups = {}
            for dst in level_nodes:
                src_nodes, _ = g.in_edges(dst, etype='net_out')
                for src in src_nodes:
                    if src.item() not in net_groups:
                        net_groups[src.item()] = []
                    net_groups[src.item()].append(dst.item())
            
            nx_g = nx.DiGraph()
            out_edges = sg.edges(etype=('node', 'snet_out', 'node'))
            for i in range(len(out_edges[0])):
                nx_g.add_edge(out_edges[0][i].item(), out_edges[1][i].item())

            #find path for each group
            all_steiner_paths = []
            all_no_src_steiner_paths = []
            all_no_dst_steiner_paths = []
            for src, dsts in net_groups.items():
                group_steiner_paths = []
                no_src_steiner_paths = []
                no_dst_steiner_paths = []
                
                for dst in dsts:
                    try:
                        # Find path using existing edges
                        path = nx.shortest_path(nx_g, source=src, target=dst)
                        no_src_path = path[1:]
                        no_dst_path = path[:-1]
                        group_steiner_paths.extend(path)
                        no_src_steiner_paths.extend(no_src_path)
                        no_dst_steiner_paths.extend(no_dst_path)
                    except nx.NetworkXNoPath:
                        print(f"No valid path found from {src} to {dst}")
                        continue
                if group_steiner_paths:
                    group_steiner_paths = torch.tensor(group_steiner_paths, dtype=torch.int64)
                    all_steiner_paths.append(group_steiner_paths)
                if no_src_steiner_paths:
                    no_src_steiner_paths = torch.tensor(no_src_steiner_paths, dtype=torch.int64)
                    all_no_src_steiner_paths.append(no_src_steiner_paths)
                if no_dst_steiner_paths:
                    no_dst_steiner_paths = torch.tensor(no_dst_steiner_paths, dtype=torch.int64)
                    all_no_dst_steiner_paths.append(no_dst_steiner_paths)
            # Combine all paths
            if all_steiner_paths:
                steiner_paths = torch.cat(all_steiner_paths)
            else:
                steiner_paths = torch.tensor([], dtype=torch.int64)

            if all_no_src_steiner_paths:
                all_no_src_steiner_paths = torch.cat(all_no_src_steiner_paths)
            else:
                all_no_src_steiner_paths = torch.tensor([], dtype=torch.int64)

            if all_no_dst_steiner_paths:
                all_no_dst_steiner_paths = torch.cat(all_no_dst_steiner_paths)
            else:
                all_no_dst_steiner_paths = torch.tensor([], dtype=torch.int64)
            
            #net propagation on steiner path
            delay, slew = NetPropFunction.apply(
                sg.edges['snet_out'].data['res'], 
                sg.nodes['node'].data['caps'],
                sg.nodes['node'].data['n_slews'],
                sg, g, steiner_paths, net_groups, nx_g, level_nodes, all_no_src_steiner_paths, all_no_dst_steiner_paths
            )

            new_values = [delay, slew]
            
            new_slew = g.nodes['node'].data['n_slews'].clone()
            new_slew[:num_pins] = new_slew[:num_pins]
            g.nodes['node'].data['n_slews'] = new_slew

            g.nodes['node'].data['n_ats'][:num_pins] = delay[:num_pins] + g.nodes['node'].data['n_ats'][:num_pins]

            
            

        else :  # Destination of cell_out
            pred_nodes, _ = g.in_edges(level_nodes, etype='cell_out')
            pred_nodes = torch.unique(pred_nodes) 
            new_slew, new_delay = CellPropFunction.apply(g, g.nodes['node'].data['n_slews'], g.nodes['node'].data['caps'], level_nodes)
            
            # Create new tensors instead of in-place operations
            new_values = [new_delay, new_slew]
            for attr, new_value in zip(delay_slew, new_values):
                new_attr = sg.nodes['node'].data[attr].clone()
                new_attr[:num_pins] = new_value
                sg.nodes['node'].data[attr] = new_attr
            

    return collector

def optimize_delay_with_coordinates(sg, g, levels, num_pins, learning_rates=[0.1], iterations=500):
    # Initialize fixed coordinates (pins)
    fixed_x = g.nodes['node'].data['x'][:num_pins].detach()
    fixed_y = g.nodes['node'].data['y'][:num_pins].detach()
    
    # Initialize movable coordinates with requires_grad=True
    movable_x = torch.nn.Parameter(sg.nodes['node'].data['x'][num_pins:].clone().detach())
    movable_y = torch.nn.Parameter(sg.nodes['node'].data['y'][num_pins:].clone().detach())
    
    params = [movable_x, movable_y]
    optimizer = optim.SGD(params, lr=learning_rates[0])
    
    for i in range(iterations):
        optimizer.zero_grad()
        
        # Create new tensors that maintain gradient flow
        x_coords = torch.cat([fixed_x, movable_x])
        y_coords = torch.cat([fixed_y, movable_y])
        
        # Create a copy of the original graph
        temp_sg = sg.clone()
        
        # Update coordinates in the copied graph
        temp_sg.nodes['node'].data['x'] = x_coords
        temp_sg.nodes['node'].data['y'] = y_coords
        
        # Compute delay with gradient tracking
        collector = get_delay(temp_sg, g, levels)
        loss = temp_sg.nodes['node'].data['n_ats'].mean()
        
        print(f"Iteration {i+1}: Loss = {loss.item():.4f}")
        
        loss.backward()
        
        if params[0].grad is not None and params[1].grad is not None:
            optimizer.step()
        else:
            print("Warning: No gradients found")
            print("Current x grad:", params[0].grad)
            print("Current y grad:", params[1].grad)
            
    return torch.stack((
        torch.cat([fixed_x, movable_x.detach()]),
        torch.cat([fixed_y, movable_y.detach()])
    ), dim=1)

inf = 1000000
def preprocess_arcs(a):
    shape = list(a.shape)
    b = a.reshape(-1, shape[-1] >> 3)
    inf_vector = np.full((b.shape[0], 1), inf)
    b = np.hstack((-inf_vector, b[:, 0:5], inf_vector, -inf_vector, b[:, 5:10], inf_vector, b[:, 10:]))
    shape[-1] = -1
    b = b.reshape(shape)
    return b

#1 preprocessing 
techdatafile = '../data/libdump_bilin.bin'
libdump = load_lib(techdatafile)
arc_emb = torch.tensor(preprocess_arcs(libdump['arc_emb']), dtype=torch.float32)
cap_emb = torch.tensor(libdump['cap_emb'], dtype=torch.float32)

benchname = 's44'
g= dgl.load_graphs(f"./benchmarks/dgl/{benchname}.graph.bin")[0][0]


#2. make attributes
def_file_path = f'./benchmarks/{benchname}.def'
levels, ep_set, pos = initialize(g, libdump, def_file_path)
#3 make steiner tree
new_g, steiner_points, steiner_edges, num_pins = steiner_tree(g)
#4. delay propagation
#get_delay(new_g, g, levels)
#print(g.nodes['node'].data['n_ats_lr'])

#5. optimize coordinates

#print("initial x steiner", new_g.nodes['node'].data['x'][num_pins:])
#print("initial y steiner", new_g.nodes['node'].data['y'][num_pins:])
optimized_coords = optimize_delay_with_coordinates(new_g, g, levels, num_pins)

#####################################################################################################
def result_graph(sg, g, levels, num_pins, optimized_coords):
    # Update coordinates with optimized values
    sg.nodes['node'].data['x'] = torch.cat([g.nodes['node'].data['x'][:num_pins], optimized_coords[:, 0]])
    sg.nodes['node'].data['y'] = torch.cat([g.nodes['node'].data['y'][:num_pins], optimized_coords[:, 1]])
    
    # Create a new graph with only the necessary nodes
    new_g = dgl.graph(([], []))
    snode_num = sg.num_nodes('node') - num_pins #steiner nodes
    new_g.add_nodes(snode_num)  # Add nodes from sg[num_pins:] steiner nodes 
    new_g.add_nodes(g.num_nodes('virtual_node'))  # Add all nodes from g's virtual_node
    
    # Copy node features from g to new_g
    sg_keys = ['n_ats', 'n_slews', 'caps']

    for key in sg_keys:
        feature_shape = g.nodes['node'].data[key].shape
        g.nodes['virtual_node'].data[key] = torch.zeros((g.num_nodes('virtual_node'), *feature_shape[1:]), dtype=torch.float32)
        for v_node in range(g.num_nodes('virtual_node')):
            v, pin = g.out_edges(v_node, etype='net_shared')
            first_pin = pin[0]
            g.nodes['virtual_node'].data[key][v_node] = g.nodes['node'].data[key][first_pin]

    for key in g.nodes['virtual_node'].data.keys():
        feature_shape = g.nodes['virtual_node'].data[key].shape
        new_g.ndata[key] = torch.zeros((new_g.num_nodes(), *feature_shape[1:]), dtype=torch.float32)
        new_g.ndata[key][snode_num:] = g.nodes['virtual_node'].data[key].clone()

    # Copy node features from sg[num_pins:] to new_g[snode_num:]
    for key in sg_keys:
        feature_shape = sg.nodes['node'].data[key].shape
        if key not in new_g.ndata.keys():
            new_g.ndata[key] = torch.zeros((new_g.num_nodes(), *feature_shape[1:]), dtype=torch.float32)
        new_g.ndata[key][:snode_num] = sg.nodes['node'].data[key][num_pins:].clone()


    # Establish connections in new_g
    src, dst = sg.edges(etype='snet_out')
    
    # Iterate over each edge
    except_port = 0
    for s, d in zip(src, dst):
        if s < num_pins:  # sg에 있는 g.nodes['node'] : pin
            vnodes, _ = g.in_edges(s, etype='net_shared')
            if vnodes.numel() > 0:
                s = vnodes + snode_num
            else:
                except_port = 1
        else: #sg에 있는 steiner nodes
            s = s - num_pins
        if d < num_pins:
            vnodes, _ = g.in_edges(d, etype='net_shared')
            if vnodes.numel() > 0:
                d = vnodes + snode_num
            else:
                except_port = 1
        else: #sg에 있는 steiner nodes
            d = d - num_pins
        edges = new_g.edges(form='uv')
        if (s, d) not in edges:
            if except_port == 0:
                new_g.add_edges(s, d)
            else:
                except_port = 0


    #get x, y from sg
    new_g.ndata['x'][:snode_num] = sg.nodes['node'].data['x'][num_pins:].clone()
    new_g.ndata['y'][:snode_num] = sg.nodes['node'].data['y'][num_pins:].clone()
    new_g.ndata['x'][snode_num:] = g.nodes['virtual_node'].data['x'].clone()
    new_g.ndata['y'][snode_num:] = g.nodes['virtual_node'].data['y'].clone()
    return new_g, snode_num








########################################################################################################################

def group_and_transform_nodes(g, levels, snode_num):
    # Step 1: Identify nodes at level 0
    level_0_nodes = torch.tensor(levels[0], dtype=torch.int64)

    # Step 2: Group connected nodes
    connected_groups = []

    for node in level_0_nodes:
        current_group = set([node.item()])
        queue = [node.item()]

        while queue:
            current_node = queue.pop(0)

            # Find all neighbors connected via net_out and cell_out
            net_out_neighbors = g.successors(current_node, etype='net_out').tolist()
            cell_out_neighbors = g.successors(current_node, etype='cell_out').tolist()
            neighbors = net_out_neighbors + cell_out_neighbors

            for neighbor in neighbors:
                if neighbor not in current_group:
                    current_group.add(neighbor)
                    queue.append(neighbor)

        connected_groups.append(list(current_group))

    # Step 3: Identify connected virtual nodes and replace node IDs with virtual node IDs
    for group in connected_groups:
        virtual_node_ids = set()  # Use a set to store unique virtual node IDs
        for node in group:
            # Find the virtual node connected via net_shared
            vnodes, _ = g.in_edges(node, etype='net_shared')
            if vnodes.numel() > 0:
                virtual_node_ids.add(vnodes[0].item() + snode_num)  # Add snode_num to the virtual node ID

        # Replace the original group with unique virtual node IDs
        group[:] = list(virtual_node_ids)

    return connected_groups

def process_and_plot_graph(sg, g, levels, num_pins, optimized_coords, benchname):
    # Update coordinates with optimized values
    sg.nodes['node'].data['x'] = torch.cat([g.nodes['node'].data['x'][:num_pins], optimized_coords[:, 0]])
    sg.nodes['node'].data['y'] = torch.cat([g.nodes['node'].data['y'][:num_pins], optimized_coords[:, 1]])
    # Create a new graph with only the necessary nodes
    new_g = dgl.graph(([], []))
    snode_num = sg.num_nodes('node') - num_pins  # steiner nodes
    new_g.add_nodes(snode_num)  # Add nodes from sg[num_pins:] steiner nodes 
    new_g.add_nodes(g.num_nodes('virtual_node'))  # Add all nodes from g's virtual_node
    
    # Copy node features from g to new_g
    sg_keys = ['n_ats', 'n_slews', 'caps']
    g_keys = ['size_choice']
    #arrival time, slew, cap from sg to g's virtual node
    for key in sg_keys:
        feature_shape = g.nodes['node'].data[key].shape
        g.nodes['virtual_node'].data[key] = torch.zeros((g.num_nodes('virtual_node'), *feature_shape[1:]), dtype=torch.float32)
        for v_node in range(g.num_nodes('virtual_node')):
            _, pin = g.out_edges(v_node, etype='net_shared')
            first_pin = pin[0]
            g.nodes['virtual_node'].data[key][v_node] = g.nodes['node'].data[key][first_pin]

    for key in g_keys:
        feature_shape = g.nodes['virtual_node'].data[key].shape
        sg.nodes['node'].data[key] = torch.zeros((sg.num_nodes('node'), *feature_shape[1:]), dtype=torch.float32)
        sg.nodes['node'].data[key][:num_pins, :] = g.nodes['node'].data[key][:num_pins, :]
        


    size_choices = g.nodes['virtual_node'].data['size_choice'].numpy()

    def get_node_size(size_choice):
        size_mapping = [5, 10, 15, 20, 25, 30, 35]
        return size_mapping[np.argmax(size_choice)]

    x_coords = sg.ndata['x'].detach().numpy()
    y_coords = sg.ndata['y'].detach().numpy()
    node_to_inst = {}
    node_to_type = {}
    basedir = './benchmarks/'
    nx_graph = utils.get_network_graph(benchname, basedir + 'network/')
    inst_names = utils.get_inst_names(nx_graph, g)
    virtual_node_ids = g.nodes('virtual_node')
    _, def_types = parse_def_type(def_file_path)
    for inst_name, v_node in zip(inst_names, virtual_node_ids):
        cell_type = def_types.get(inst_name, 'unknown')
        _, connected_nodes = g.out_edges(v_node, etype='net_shared')
        for node in connected_nodes:
            node_to_inst[node.item()] = inst_name
            node_to_type[node.item()] = cell_type

    nx_g = nx.DiGraph()
    out_edges = sg.edges(etype=('node', 'snet_out', 'node'))
    for i in range(len(out_edges[0])):
        nx_g.add_edge(out_edges[0][i].item(), out_edges[1][i].item())

    
    node_to_vnode = {}
    for v_node in range(g.num_nodes('virtual_node')):
        _, connected_nodes = g.out_edges(v_node, etype='net_shared')
        for node in connected_nodes:
            node_to_vnode[node.item()] = v_node
            


    def get_mapped_size(idx):
        if idx < num_pins:
            # If node has a virtual node mapping, use that
            if idx in node_to_vnode:
                vnode = node_to_vnode[idx]
                return get_node_size(size_choices[vnode])
        # For Steiner nodes or unmapped nodes, return minimum size
        return 5

    def get_mapped_size_choice(idx):
        if idx < num_pins:
            # If node has a virtual node mapping, use that
            if idx in node_to_vnode:
                vnode = node_to_vnode[idx]
                return g.nodes['virtual_node'].data['size_choice'][vnode]
        # For Steiner nodes or unmapped nodes, return zero vector
        return torch.zeros(7, dtype=torch.float32)

    # Ensure size_choice is initialized for all nodes
    if 'size_choice' not in g.ndata:
        g.ndata['size_choice'] = torch.zeros((g.num_nodes('node'), 7), dtype=torch.float32)

    for l in range(len(levels)):
        level_nodes = torch.tensor(levels[l], dtype=torch.int64)

        net_groups = {}
        for src in level_nodes:
            _, dsts = g.out_edges(src, etype='net_out')
            for dst in dsts:
                if src.item() not in net_groups:
                    net_groups[src.item()] = []
                net_groups[src.item()].append(dst.item())
            
        #find path for each group
        all_steiner_paths = {}
        path_per_src = {}
        for src, dsts in net_groups.items():
            path_per_src[src] = []
            for dst in dsts:
                try:
                    # Find path using existing edges
                    path = nx.shortest_path(nx_g, source=src, target=dst)
                    all_steiner_paths[(src, dst)] = path
                    path_per_src[src].extend(path)
                except nx.NetworkXNoPath:
                    print(f"No valid path found from {src} to {dst}")
                    continue
            
        for i, (src, dsts) in enumerate(net_groups.items()):

            fig = go.Figure()
            src_path = path_per_src[src]
            nodes_for_src = list(set(src_path))
            group_x_coords = x_coords[nodes_for_src]
            group_y_coords = y_coords[nodes_for_src]


            fig.add_trace(go.Scatter(
                x=group_x_coords,
                y=group_y_coords,
                mode='markers',
                marker=dict(
                    color=[
                        'yellow' if idx == src else 
                        'blue' if idx in dsts else 
                        'red' 
                        for idx in nodes_for_src
                    ],
                    size=[get_mapped_size(idx) for idx in nodes_for_src],
                    opacity=0.6
                ),
                text=[
                    f'Node {idx}' + (
                        f"<br>n_ats: {sg.ndata['n_ats'][idx]:.4f}"
                        f"<br>n_slew: {sg.ndata['n_slews'][idx]:.4f}"
                        f"<br>Type: {node_to_type.get(idx, 'Steiner or port')}"
                        f"<br>Size: {np.argmax(get_mapped_size_choice(idx)) if idx < num_pins else 0}"
                        f"<br>Size Vector: [{', '.join([f'{x:.1f}' for x in get_mapped_size_choice(idx)])}]"
                    ) if idx in nodes_for_src else f'Node {idx}'
                    for idx in nodes_for_src
                ],
                hoverinfo='text'
                ))
            
            for dst in dsts:
                path = all_steiner_paths[(src, dst)]
                path_length = len(path)
                path_x = [x_coords[node] for node in path]
                path_y = [y_coords[node] for node in path]
                for j in range(path_length - 1):
                    x1, y1 = path_x[j], path_y[j]
                    x2, y2 = path_x[j + 1], path_y[j + 1]
                    fig.add_trace(go.Scatter(
                    x=[x1, x2],
                    y=[y1, y2],
                    mode='lines',
                    line=dict(color='gray', width=2),
                    hoverinfo='skip'
            ))

            fig.update_layout(
                title=f'{benchname} Snetout Group {src} to {dsts} Visualization',
                showlegend=False
            )
            fig.write_html(f'./results/result/{benchname}_snetout_group_{l}_levels_src_{src}.html')

    print('Snetout group plots saved as HTML.')

def plot_whole_graph(sg, g, levels, num_pins, benchname):
    # Get all relevant edges and nodes
    id1 = 98
    id2 = 386
    id3 = 24
    ids = [id1, id2, id3]
    plot_nodes = []
    net_src = []
    net_dst = []
    cell_src = []
    cell_dst = []
    connected_src = []
    connected_dst = []
    for id in ids:
        id_net_src, id_net_dst = g.in_edges(id, etype='net_out')
        id_cell_src, id_cell_dst = g.out_edges(id, etype='cell_out')
        id_connected_src, id_connected_dst = g.out_edges(id_cell_dst, etype='net_out')
        
        # Convert all tensors to lists and extend the collections
        plot_nodes.extend([id])  # Add the ID itself
        plot_nodes.extend(id_net_src.tolist())
        plot_nodes.extend(id_net_dst.tolist())
        plot_nodes.extend(id_cell_src.tolist())
        plot_nodes.extend(id_cell_dst.tolist())
        plot_nodes.extend(id_connected_src.tolist())
        plot_nodes.extend(id_connected_dst.tolist())
        
        net_src.extend(id_net_src.tolist())
        net_dst.extend(id_net_dst.tolist())
        cell_src.extend(id_cell_src.tolist())
        cell_dst.extend(id_cell_dst.tolist())
        connected_src.extend(id_connected_src.tolist())
        connected_dst.extend(id_connected_dst.tolist())

    plot_nodes = list(set(plot_nodes))
    
    # Create figure
    fig = go.Figure()
    
    # Get node coordinates and convert to numpy arrays
    x_coords = g.nodes['node'].data['x'].detach().numpy()
    y_coords = g.nodes['node'].data['y'].detach().numpy()
    node_to_inst = {}
    node_to_type = {}
    basedir = './benchmarks/'
    nx_graph = utils.get_network_graph(benchname, basedir + 'network/')
    inst_names = utils.get_inst_names(nx_graph, g)
    virtual_node_ids = g.nodes('virtual_node')
    _, def_types = parse_def_type(def_file_path)
    for inst_name, v_node in zip(inst_names, virtual_node_ids):
        cell_type = def_types.get(inst_name, 'unknown')
        _, connected_nodes = g.out_edges(v_node, etype='net_shared')
        for node in connected_nodes:
            node_to_inst[node.item()] = inst_name
            node_to_type[node.item()] = cell_type
    node_to_vnode = {}
    for v_node in range(g.num_nodes('virtual_node')):
        _, connected_nodes = g.out_edges(v_node, etype='net_shared')
        for node in connected_nodes:
            node_to_vnode[node.item()] = v_node
            


    def get_mapped_size_choice(idx):
        if idx < num_pins:
            # If node has a virtual node mapping, use that
            if idx in node_to_vnode:
                vnode = node_to_vnode[idx]
                return g.nodes['virtual_node'].data['size_choice'][vnode]
        # For Steiner nodes or unmapped nodes, return zero vector
        return torch.zeros(7, dtype=torch.float32)
    
    # Create node colors mapping
    node_colors = {}
    # id1 is blue
    node_colors[id1] = 'blue'
    # id1_cell nodes are blue
    for node in cell_dst:
        node_colors[node] = 'blue'
    # id1_net source nodes are yellow
    for node in net_src:
        node_colors[node] = 'yellow'
    # id1_connected destination nodes are purple
    for node in connected_dst:
        node_colors[node] = 'purple'
    # Plot nodes with explicit color assignment
    fig.add_trace(go.Scatter(
        x=x_coords[plot_nodes],
        y=y_coords[plot_nodes],
        mode='markers',
        marker=dict(
            color=[node_colors.get(node, 'red') for node in plot_nodes],
            size=[1 if node >= num_pins else 10 for node in plot_nodes],
            opacity=0.8
        ),
        text=[
            "Node {}\n"
            "n_ats: {:.4f}\n"
            "n_slew: {:.4f}\n"
            "Type: {}\n"
            "Size Vector: [{}]".format(
                node,
                sg.nodes['node'].data['n_ats'][node].detach().numpy() if node >= num_pins else g.nodes['node'].data['n_ats'][node].detach().numpy(),
                sg.nodes['node'].data['n_slews'][node].detach().numpy() if node >= num_pins else g.nodes['node'].data['n_slews'][node].detach().numpy(),
                'Steiner' if node >= num_pins else node_to_type.get(node, 'Unknown'),
                ', '.join(['{:.1f}'.format(x) for x in get_mapped_size_choice(node)])
            )
            for node in plot_nodes
        ],
        hoverinfo='text'
    ))
    
    # Plot net_out edges (blue)
    for src, dst in zip(net_src, net_dst):
        fig.add_trace(go.Scatter(
            x=[x_coords[src], x_coords[dst]],
            y=[y_coords[src], y_coords[dst]],
            mode='lines',
            line=dict(color='blue', width=2),
            hoverinfo='skip'
        ))
    
    # Plot cell_out edges (red)
    for src, dst in zip(cell_src, cell_dst):
        fig.add_trace(go.Scatter(
            x=[x_coords[src], x_coords[dst]],
            y=[y_coords[src], y_coords[dst]],
            mode='lines',
            line=dict(color='red', width=2),
            hoverinfo='skip'
        ))
    
    # Plot connected edges (blue for net_out)
    for src, dst in zip(connected_src, connected_dst):
        fig.add_trace(go.Scatter(
            x=[x_coords[src], x_coords[dst]],
            y=[y_coords[src], y_coords[dst]],
            mode='lines',
            line=dict(color='blue', width=2),
            hoverinfo='skip'
        ))
    
    # Update layout
    fig.update_layout(
        title=f'{benchname} Graph Visualization for Node {id1} and Connected Components',
        showlegend=False,
        hovermode='closest'
    )
    
    # Save the plot
    fig.write_html(f'./results/circuit_graph/{benchname}_node_{id1}_components.html')
    print(f'Graph plot saved as {benchname}_node_{id1}_components.html')

plot_whole_graph(new_g, g, levels, num_pins, benchname)
process_and_plot_graph(new_g, g, levels, num_pins, optimized_coords, benchname)