import torch
import networkx as nx
import dgl.function as fn
import pickle as pk
import numpy as np
import lib
from gradient_collector import GradientCollector
import torch.nn.functional as F
parser = lib.LibertyParser()
parser.parse_file('./data/sky130_fd_sc_hd__tt_025C_1v80.lib')
wire_load_models = parser.parse_cap_res()['small']
unit_cap = wire_load_models[0]
unit_res = wire_load_models[1]
delay_slew = ['n_ats', 'n_slews']

techdatafile = './data/libdump_bilin.bin'
def load_lib(filename):
    # Use a breakpoint in the code line below to debug your script.
    f1 = open(filename, 'rb')
    dmp = pk.load(f1)
    f1.close()
    return dmp
inf = 1000000
def preprocess_arcs(a):
    shape = list(a.shape)
    b = a.reshape(-1, shape[-1] >> 3)
    inf_vector = np.full((b.shape[0], 1), inf)
    b = np.hstack((-inf_vector, b[:, 0:5], inf_vector, -inf_vector, b[:, 5:10], inf_vector, b[:, 10:]))
    shape[-1] = -1
    b = b.reshape(shape)
    return b
def preprocess_arcs(a):
    shape = list(a.shape)
    b = a.reshape(-1, shape[-1] >> 3)
    inf_vector = np.full((b.shape[0], 1), inf)
    b = np.hstack((-inf_vector, b[:, 0:5], inf_vector, -inf_vector, b[:, 5:10], inf_vector, b[:, 10:]))
    shape[-1] = -1
    b = b.reshape(shape)
    return b

libdump = load_lib(techdatafile)
arc_emb = torch.tensor(preprocess_arcs(libdump['arc_emb']), dtype=torch.float32)
cap_emb = torch.tensor(libdump['cap_emb'], dtype=torch.float32)

# Define message and reduction functions
def build_arcs_delay(eb):
    if len(eb) == 0:
        return
    else:
        eb.data['delay'] = eb.src['delay'] + eb.data['res'] * eb.dst['caps'][:,2]
        return {'delay': eb.data['delay']}

def build_arcs_a(eb):
    """Calculate a(u) = Cap(u) * Delay(u) + Σ a(v)"""
    if len(eb) == 0:
        return
    else:
        # Forward message: current node's contribution to a
        a = eb.dst['caps'][:, 2] * eb.dst['n_ats']
        eb.data['a'] = a
        eb.data['child_a'] = eb.src['a']
        return {'a': eb.data['a'], 'child_a': eb.data['child_a']}

def reduce_a(nodes):
    """Reduce function for a: sum of all child a values plus node's contribution"""
    return {
        'a': nodes.mailbox['child_a'].sum(dim=1) + nodes.mailbox['a'].mean(dim=1)
    }

def build_arcs_b(eb):
    """Calculate b(u) = b(fa(u)) + Res(fa(u) → u) * a(u)"""
    if len(eb) == 0:
        return
    else:
        # b calculation using resistance and a
        b = eb.data['res'] * eb.dst['a']
        pred_b = eb.src['b']
        eb.data['b'] = b
        return {'b': eb.data['b'], 'pred_b': pred_b}

def reduce_b(nodes):
    """Reduce function for b: maximum of incoming b values"""
    return {
        'b': nodes.mailbox['pred_b'].sum(dim=1) + nodes.data['b']
    }

def build_arcs_slew(eb):
    if len(eb) == 0:
        return   
    else:
        eb.data['s'] = torch.sqrt(torch.pow(eb.src['n_slews'], 2) + torch.pow(eb.dst['imp'], 2))
        eb.data['s'].requires_grad_()
        return {'s': eb.data['s']}     

# Define message and reduction functions as lambda functions
reduce_cap = lambda nodes: {'caps': nodes.mailbox['m'].sum(dim=1) + nodes.data['caps']}
reduce_delay = lambda nodes: {'delay': torch.logsumexp(nodes.mailbox['delay'], dim=1)}
reduce_a = lambda nodes: {'a': torch.logsumexp(nodes.mailbox['a'], dim=1)}
reduce_b = lambda nodes: {'b': torch.logsumexp(nodes.mailbox['b'], dim=1)}
reduce_slew = lambda nodes: {'s': nodes.mailbox['s'].sum(dim=1)}

def interp(slew, cap, delay_rise, slew_rise):
    # Ensure x and y are scalar values
    x = slew.item() if torch.is_tensor(slew) else slew
    y = cap.item() if torch.is_tensor(cap) else cap
    slew_vals_delay = delay_rise['index_1']
    cap_vals_delay = delay_rise['index_2']
    slew_vals_slew = slew_rise['index_1']
    cap_vals_slew = slew_rise['index_2']
    slew_lut = slew_rise['values'][0]
    delay_lut = delay_rise['values'][0]

    # Initialize flags for extrapolation
    x_extrapolate = False
    y_extrapolate = False

    x1_idxs = [0,0]
    x2_idxs = [1,1]
    y1_idxs = [0,0]
    y2_idxs = [1,1]
    slew_vals = [slew_vals_delay, slew_vals_slew]
    cap_vals = [cap_vals_delay, cap_vals_slew]
    # Check x bounds and set indices
    for slew_delay in range(2):
        x1_idx = x1_idxs[slew_delay]
        x2_idx = x2_idxs[slew_delay]
        slew_val = slew_vals[slew_delay]
        cap_val = cap_vals[slew_delay]
        if x < slew_val[0]:
            x1_idx = 0
            x2_idx = 1
            x_extrapolate = True
            x_extrap_dir = -1  # extrapolating below minimum
        elif x >= slew_val[-1]:
            x1_idx = len(slew_val) - 2
            x2_idx = len(slew_val) - 1
            x_extrapolate = True
            x_extrap_dir = 1   # extrapolating above maximum
        else:
            for i in range(len(slew_val) - 1):
                if slew_val[i] <= x < slew_val[i + 1]:
                    x1_idx = i
                    x2_idx = i + 1
                    break
        x1_idxs[slew_delay] = x1_idx
        x2_idxs[slew_delay] = x2_idx
        y1_idx = y1_idxs[slew_delay]
        y2_idx = y2_idxs[slew_delay]
        # Check y bounds and set indices
        if y < cap_val[0]:
            y1_idx = 0
            y2_idx = 1
            y_extrapolate = True
            y_extrap_dir = -1  # extrapolating below minimum
        elif y >= cap_val[-1]:
            y1_idx = len(cap_val) - 2
            y2_idx = len(cap_val) - 1
            y_extrapolate = True
            y_extrap_dir = 1   # extrapolating above maximum
        else:
            for i in range(len(cap_val) - 1):
                if cap_val[i] <= y < cap_val[i + 1]:
                    y1_idx = i
                    y2_idx = i + 1
                    break
        y1_idxs[slew_delay] = y1_idx
        y2_idxs[slew_delay] = y2_idx
    
    # Get reference points
    for slew_delay in range(2):
        x1_idx = x1_idxs[slew_delay]
        x2_idx = x2_idxs[slew_delay]
        y1_idx = y1_idxs[slew_delay]
        y2_idx = y2_idxs[slew_delay]
        x1, x2 = slew_vals[slew_delay][x1_idx], slew_vals[slew_delay][x2_idx]
        y1, y2 = cap_vals[slew_delay][y1_idx], cap_vals[slew_delay][y2_idx]

        # Extract LUT values
        v21_s = torch.tensor(slew_lut[x1_idx][y1_idx], requires_grad=True)
        v22_s = torch.tensor(slew_lut[x1_idx][y2_idx], requires_grad=True)
        v31_s = torch.tensor(slew_lut[x2_idx][y1_idx], requires_grad=True)
        v32_s = torch.tensor(slew_lut[x2_idx][y2_idx], requires_grad=True)

        v21_c = torch.tensor(delay_lut[x1_idx][y1_idx], requires_grad=True)
        v22_c = torch.tensor(delay_lut[x1_idx][y2_idx], requires_grad=True)
        v31_c = torch.tensor(delay_lut[x2_idx][y1_idx], requires_grad=True)
        v32_c = torch.tensor(delay_lut[x2_idx][y2_idx], requires_grad=True)
        # Interpolate/Extrapolate in y direction
        if y_extrapolate:
            # Use 0.5 slope for y extrapolation
            v2y_s = v21_s + y_extrap_dir * 0.5 * abs(y - y1)
            v3y_s = v31_s + y_extrap_dir * 0.5 * abs(y - y1)
            v2y_c = v21_c + y_extrap_dir * 0.5 * abs(y - y1)
            v3y_c = v31_c + y_extrap_dir * 0.5 * abs(y - y1)
            
        else:
            # Normal interpolation in y direction
            v2y_s = (y2 - y) / (y2 - y1) * v21_s + (y - y1) / (y2 - y1) * v22_s
            v3y_s = (y2 - y) / (y2 - y1) * v31_s + (y - y1) / (y2 - y1) * v32_s
            v2y_c = (y2 - y) / (y2 - y1) * v21_c + (y - y1) / (y2 - y1) * v22_c
            v3y_c = (y2 - y) / (y2 - y1) * v31_c + (y - y1) / (y2 - y1) * v32_c
            
        # Interpolate/Extrapolate in x direction
        if x_extrapolate:
            # Use 0.5 slope for x extrapolation
            v_xy_s = v2y_s + x_extrap_dir * 0.5 * abs(x - x1)
            v_xy_c = v2y_c + x_extrap_dir * 0.5 * abs(x - x1)
            
        else:
            # Normal interpolation in x direction
            x_diff = x2 - x1
            v_xy_s = (x2 - x) / x_diff * v2y_s + (x - x1) / x_diff * v3y_s
            v_xy_c = (x2 - x) / x_diff * v2y_c + (x - x1) / x_diff * v3y_c

    # Set constant gradients for extrapolation regions
    grad_val_xs = 0.5 if x_extrapolate else (v3y_s - v2y_s) / (x2 - x1)
    grad_val_ys = 0.5 if y_extrapolate else (v22_s - v21_s) / (y2 - y1)
    grad_val_xc = 0.5 if x_extrapolate else (v3y_c - v2y_c) / (x2 - x1)
    grad_val_yc = 0.5 if y_extrapolate else (v22_c - v21_c) / (y2 - y1)
    
    # Ensure grad_val_xs is a tensor
    grad_val_xs = torch.tensor(grad_val_xs, dtype=torch.float32) if isinstance(grad_val_xs, float) else grad_val_xs
    grad_val_ys = torch.tensor(grad_val_ys, dtype=torch.float32) if isinstance(grad_val_ys, float) else grad_val_ys
    grad_val_xc = torch.tensor(grad_val_xc, dtype=torch.float32) if isinstance(grad_val_xc, float) else grad_val_xc
    grad_val_yc = torch.tensor(grad_val_yc, dtype=torch.float32) if isinstance(grad_val_yc, float) else grad_val_yc

    grad_xs = grad_val_xs.clone().detach().requires_grad_(True)
    grad_ys = grad_val_ys.clone().detach().requires_grad_(True)
    grad_xc = grad_val_xc.clone().detach().requires_grad_(True)
    grad_yc = grad_val_yc.clone().detach().requires_grad_(True)

    return v_xy_s, v_xy_c, grad_xs, grad_ys, grad_xc, grad_yc



class CellPropFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, slew, cap, level_nodes):
        # Save inputs for backward pass
        ctx.g = g
        ctx.save_for_backward(slew.clone(), cap.clone())
        ctx.level_nodes = level_nodes

        # Initialize collector and save to context
        ctx.collector = GradientCollector()
        
        # Register hooks for tensors we want to track
        slew.register_hook(ctx.collector.hook_factory('slews'))
        cap.register_hook(ctx.collector.hook_factory('caps'))

        # Calculate delay and slew using LUT interpolation
        delay_max = []
        slew_max = []
        grad_xs_values = []
        grad_ys_values = []
        grad_xc_values = []
        grad_yc_values = []
        whole_delay = []
        whole_slew = []
        
        for node in level_nodes:
            delay_values = []
            slew_values = []
            predecessors, _ = g.in_edges(node, etype='cell_out')
            timing_info = parser.lookup_timing(g, node)
            delay_rise = timing_info['cell_rise']
            slew_rise = timing_info['rise_transition']
            eids = g.in_edges(node, etype='cell_out', form='eid')
            
            if len(predecessors.tolist()) > 0:
                grad_xs_each_pred = []
                grad_ys_each_pred = []
                grad_xc_each_pred = []
                grad_yc_each_pred = []
                
                for pred, eid in zip(predecessors, eids):
                    arc_type = g.edges['cell_out'].data['feat1'][eid]
                    pred_delay = g.nodes['node'].data['n_ats'][pred]
                    
                    s_r, d_r, grad_xs_r, grad_ys_r, grad_xc_r, grad_yc_r = interp(
                        slew[pred], cap[pred,2], delay_rise, slew_rise
                    )
                    d_r += pred_delay  # Add predecessor's delay
                    
                    grad_xs_each_pred.append(grad_xs_r)
                    grad_ys_each_pred.append(grad_ys_r)
                    grad_xc_each_pred.append(grad_xc_r)
                    grad_yc_each_pred.append(grad_yc_r)
                    
                    delay_values.append(d_r)
                    slew_values.append(s_r)
                
                # Convert lists to tensors
                delay_values = torch.stack(delay_values)
                slew_values = torch.stack(slew_values)
                whole_delay.append(delay_values)
                whole_slew.append(slew_values)
                
                # Compute logsumexp
                delay_interp = torch.logsumexp(delay_values, dim=0)
                slew_interp = torch.logsumexp(slew_values, dim=0)
                
                delay_max.append(delay_interp)
                slew_max.append(slew_interp)
                
                grad_xs_values.append(grad_xs_each_pred)
                grad_ys_values.append(grad_ys_each_pred)
                grad_xc_values.append(grad_xc_each_pred)
                grad_yc_values.append(grad_yc_each_pred)
            
            else:
                delay_max.append(0)
                slew_max.append(0)
                grad_xs_values.append(0)
                grad_ys_values.append(0)
                grad_xc_values.append(0)
                grad_yc_values.append(0)
        
        ctx.grad_xs_values = grad_xs_values
        ctx.grad_ys_values = grad_ys_values
        ctx.grad_xc_values = grad_xc_values
        ctx.grad_yc_values = grad_yc_values
        ctx.whole_delay = whole_delay
        ctx.whole_slew = whole_slew
        
        g.nodes['node'].data['n_ats'][level_nodes] = torch.tensor(delay_max, dtype=torch.float32)
        g.nodes['node'].data['n_slews'][level_nodes] = torch.tensor(slew_max, dtype=torch.float32)

        # Ensure gradients are retained for non-leaf tensors
        g.nodes['node'].data['n_slews'].retain_grad()
        
        return g.nodes['node'].data['n_slews'], g.nodes['node'].data['n_ats']

    @staticmethod
    def backward(ctx, grad_output_slew, grad_output_delay):
        level_nodes = ctx.level_nodes
        g = ctx.g
        
        # Initialize gradients with collected values from hooks
        grad_slew = grad_output_slew
        grad_cap = ctx.collector.gradients.get('caps', torch.zeros_like(g.nodes['node'].data['caps']))
        
        # Get saved gradient values
        grad_xs_values = ctx.grad_xs_values
        grad_ys_values = ctx.grad_ys_values
        grad_xc_values = ctx.grad_xc_values
        grad_yc_values = ctx.grad_yc_values
        whole_delay = ctx.whole_delay
        whole_slew = ctx.whole_slew
        
        for i, node in enumerate(level_nodes):
            predecessors, _ = g.in_edges(node, etype='cell_out')
            
            for j, pred in enumerate(predecessors):
                # Calculate LSE gradient
                saved_delay = whole_delay[i][j]
                saved_slew = whole_slew[i][j]
                
                exp_delay = torch.exp(saved_delay + g.nodes['node'].data['n_ats'][pred])
                sum_exp = torch.sum(torch.exp(saved_delay + g.nodes['node'].data['n_ats'][predecessors]))
                
                if exp_delay == np.inf:
                    LSE_delay = 1
                elif sum_exp == np.inf:
                    LSE_delay = 0
                else:
                    LSE_delay = exp_delay / (sum_exp + 1e-8)
                
                exp_slew = torch.exp(saved_slew + g.nodes['node'].data['n_slews'][pred])
                sum_exp_slew = torch.sum(torch.exp(saved_slew + g.nodes['node'].data['n_slews'][predecessors]))
                if exp_slew == np.inf:
                    LSE_slew = 1
                elif sum_exp_slew == np.inf:
                    LSE_slew = 0
                else:
                    LSE_slew = exp_slew / (sum_exp_slew + 1e-8)

                # Get gradients for this predecessor
                grad_xs = grad_xs_values[i][j]
                grad_ys = grad_ys_values[i][j]
                grad_xc = grad_xc_values[i][j]
                grad_yc = grad_yc_values[i][j]

                # Calculate final gradients
                grad_slew_val = LSE_delay * grad_xc + LSE_slew * grad_xs
                grad_cap_val = LSE_delay * grad_yc + LSE_slew * grad_ys

                g.nodes['node'].data['slew_grad'][pred] = grad_slew_val
                g.nodes['node'].data['caps_grad'][node][2] = grad_cap_val

        return None, g.nodes['node'].data['slew_grad'], g.nodes['node'].data['caps_grad'], None

def reduce_grad_a(nodes):
    return {
        'a_grad': nodes.mailbox['a_grad'].sum(dim=1)
    }

def build_arcs_grad_a(eb):
    """
    Message function for gradient of a
    Computes gradients
    """
    eb.dst['a_grad'] = eb.data['res'] * eb.dst['b_grad'] + eb.src['a_grad']
    return {'a_grad': eb.dst['a_grad']}

def build_arcs_load_grad(eb):
    eb.dst['load_grad'] = eb.data['res'] * eb.dst['delay_grad'] + eb.src['delay_grad']
    return {'load_grad': eb.dst['load_grad']}

def build_arcs_slew_grad(eb):
    eb.dst['slew_grad'] = eb.dst['n_slews'] / (eb.src['n_slews']+1e-4) * eb.src['slew_grad']
    return {'slew_grad': eb.dst['slew_grad']}

def reduce_load_grad(nodes):
    return {'load_grad': nodes.mailbox['load_grad'].sum(dim=1)}

def reduce_slew_grad(nodes):
    return {'slew_grad': nodes.mailbox['slew_grad'].sum(dim=1)}

class NetPropFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, res, caps, slews, sg, g, steiner_paths, net_groups, nx_g, level_nodes, all_no_src_steiner_paths, all_no_dst_steiner_paths):
        sg.nodes['node'].data['x'].requires_grad_(True)
        sg.nodes['node'].data['y'].requires_grad_(True)
        ctx.save_for_backward(res, caps, slews)
        ctx.x_coords = sg.nodes['node'].data['x']
        ctx.y_coords = sg.nodes['node'].data['y']
        ctx.g = g
        ctx.sg = sg
        ctx.steiner_paths = steiner_paths
        ctx.all_no_src_steiner_paths = all_no_src_steiner_paths
        ctx.all_no_dst_steiner_paths = all_no_dst_steiner_paths
        ctx.net_groups = net_groups
        # 2. Delay - forward delay propagation
        passed = []
        for node in steiner_paths:
            if node not in passed:
                sg['snet_out'].pull(node, message_func=build_arcs_delay, reduce_func=reduce_delay)
                passed.append(node)
        
        passed = []
        for node in reversed(steiner_paths):
            if node not in passed:
                sg['snet_in'].pull(node, message_func=build_arcs_a, reduce_func=reduce_a)
                passed.append(node)
        
        passed = []
        for node in steiner_paths:
            if node not in passed:
                sg['snet_out'].pull(steiner_paths, message_func=build_arcs_b, reduce_func=reduce_b)
                passed.append(node)

        # Calculate impulse and slew
        imp_values = torch.sqrt(torch.clamp(2 * sg.nodes['node'].data['b'][steiner_paths] - 
                                          torch.pow(sg.nodes['node'].data['delay'][steiner_paths], 2), min=1e-6))
        new_imp = sg.nodes['node'].data['imp'].clone()
        new_imp[steiner_paths] = imp_values
        sg.nodes['node'].data['imp'] = new_imp
        sg['snet_out'].pull(steiner_paths, message_func=build_arcs_slew, reduce_func=reduce_slew)

        # Copy values
        delay = sg.nodes['node'].data['delay'].clone()
        slew = sg.nodes['node'].data['n_slews'].clone()

        # steiner_paths에 있는 노드들에 대해서만 업데이트
        for path in steiner_paths:
            # Update only nodes in the current path
            delay[path] = sg.nodes['node'].data['n_ats'][path]
            slew[path] = sg.nodes['node'].data['n_slews'][path]

        ctx.collector = GradientCollector()  # Add collector to context
        
        # Register hooks for important tensors
        res.register_hook(ctx.collector.hook_factory('res'))
        caps.register_hook(ctx.collector.hook_factory('caps'))
        slews.register_hook(ctx.collector.hook_factory('slews'))
        
        return delay, slew


    @staticmethod
    def backward(ctx, grad_output_delay, grad_output_slew):
        sg = ctx.sg
        steiner_paths = ctx.steiner_paths
        all_no_src_steiner_paths = ctx.all_no_src_steiner_paths
        all_no_dst_steiner_paths = ctx.all_no_dst_steiner_paths
        # 입입력에 대한 gradient 반환
        grad_res = ctx.collector.gradients.get('res', torch.zeros_like(sg.edges['snet_out'].data['res']))
        grad_caps = ctx.collector.gradients.get('caps', torch.zeros_like(sg.nodes['node'].data['caps']))
        grad_slews = grad_output_slew if grad_output_slew is not None else torch.zeros_like(sg.nodes['node'].data['n_slews'])

        sg.nodes['node'].data['slew_grad'] = grad_slews
        sg.nodes['node'].data['delay_grad'] = grad_output_delay
        for node in all_no_dst_steiner_paths:
            sg['snet_in'].pull(node, build_arcs_slew_grad, reduce_slew_grad)
        epsilon = 1e-8  # 매우 작은 값
        sg.nodes['node'].data['imp2_grad'][steiner_paths] = sg.nodes['node'].data['slew_grad'][steiner_paths] / (2 * sg.nodes['node'].data['n_slews'][steiner_paths] + epsilon)
        
        



        #1. gradient prop.
        for node in all_no_dst_steiner_paths:
            sg['snet_in'].pull(node, fn.copy_u('b_grad', 'b_grad'), fn.sum('b_grad', 'b_grad')) 
            sg.nodes['node'].data['b_grad'][node] = sg.nodes['node'].data['b_grad'][node] + 2* sg.nodes['node'].data['imp2_grad'][node]

        for node in all_no_src_steiner_paths:
            sg['snet_out'].pull(node, build_arcs_grad_a, reduce_grad_a)

        for node in all_no_dst_steiner_paths:
            sg['snet_in'].pull(node, fn.copy_u('delay_grad', 'delay_grad'), fn.sum('delay_grad', 'delay_grad'))
            sg.nodes['node'].data['delay_grad'][node] = sg.nodes['node'].data['delay_grad'][node] + sg.nodes['node'].data['caps'][node][2] * sg.nodes['node'].data['a_grad'][node] + 2* sg.nodes['node'].data['delay'][node] * sg.nodes['node'].data['imp2_grad'][node]
        for node in all_no_src_steiner_paths:
            sg['snet_out'].pull(node, build_arcs_load_grad, reduce_load_grad)
        
        _, all_v = sg.in_edges(steiner_paths, etype='snet_out')
        grad_res[all_v] = sg.nodes['node'].data['caps'][all_v][:,2] * sg.nodes['node'].data['delay_grad'][all_v] + sg.nodes['node'].data['b'][all_v] * sg.nodes['node'].data['a_grad'][all_v]
        print("sg.nodes['node'].data['caps'][all_v][:,2]", sg.nodes['node'].data['caps'][all_v][:,2])
        print("sg.nodes['node'].data['delay_grad'][all_v]", sg.nodes['node'].data['delay_grad'][all_v])
        print("sg.nodes['node'].data['b'][all_v]", sg.nodes['node'].data['b'][all_v])
        print("sg.nodes['node'].data['a_grad'][all_v]", sg.nodes['node'].data['a_grad'][all_v])
        grad_caps[all_v][:,2] = sg.nodes['node'].data['caps'][all_v][:,2] * sg.nodes['node'].data['delay_grad'][all_v] + sg.nodes['node'].data['b_grad'][all_v] * sg.nodes['node'].data['a_grad'][all_v]
        grad_slews = sg.nodes['node'].data['slew_grad']
        print("grad_res", grad_res)
        return grad_res, grad_caps, grad_slews, None, None, None, None, None, None, None, None, None


