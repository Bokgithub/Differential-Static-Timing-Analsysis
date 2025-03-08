import dgl
import torch
import lib
import networkx as nx


def_content = open('./benchmarks/s44.def', 'r').read()
parser = lib.LibertyParser()
tracks_info = parser.parse_tracks(def_content)
x_origin = tracks_info['li1']['X'][0]
x_num = tracks_info['li1']['X'][1]
x_step = tracks_info['li1']['X'][2]
y_origin = tracks_info['li1']['Y'][0]
y_num = tracks_info['li1']['Y'][1]
y_step = tracks_info['li1']['Y'][2]
src_to_dst_path ={}
def steiner_tree(g):
    # src and dst nodes connected by net_out
    src_nodes, dst_nodes = g.edges(etype='net_out')
    def map_to_nearest_track(coord, origin, step):
        # Calculate how many steps away from origin
        steps = round((coord - origin) / step)
        # Return the nearest valid track coordinate
        return float(origin + (steps * step))
    x_data = g.nodes['node'].data['x']
    mapped_x = torch.tensor([map_to_nearest_track(x.item(), x_origin, x_step) for x in x_data], 
                          dtype=torch.float32, 
                          requires_grad=True)
    g.nodes['node'].data['x'] = mapped_x

    # y coordinates
    y_data = g.nodes['node'].data['y']
    mapped_y = torch.tensor([map_to_nearest_track(y.item(), y_origin, y_step) for y in y_data], 
                          dtype=torch.float32,
                          requires_grad=True)
    g.nodes['node'].data['y'] = mapped_y
    # group for each net
    nets = {}

    for src, dst in zip(src_nodes, dst_nodes):
        src_id = src.item()
        if src_id not in nets:
            nets[src_id] = set()
        nets[src_id].add(dst.item())

    all_steiner_points = []
    all_edges = []
    src_to_dst_coord_path = {}  # Dictionary to store coordinate paths
    src_to_dst_node_path = {}   # Dictionary to store node ID paths

    # create steiner tree for each net
    for src_id, dst_set in nets.items():
        # source pin's coordinates
        src_x = g.nodes['node'].data['x'][src_id].item()
        src_y = g.nodes['node'].data['y'][src_id].item()
        P = [(src_x, src_y)] # all pin coordinates of current net (source + destinations)
        
        
        for dst_id in dst_set:
            dst_x = g.nodes['node'].data['x'][dst_id].item()
            dst_y = g.nodes['node'].data['y'][dst_id].item()
            P.append((dst_x, dst_y))
        P_prime = P.copy()

        net_steiner_points = []
        net_edges = []
        
        # find the closest pair of pins
        def closest_pair(points):
            min_dist = float('inf')
            closest = None
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    dist = abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])
                    if dist < min_dist:
                        min_dist = dist
                        closest = (points[i], points[j])
            return closest
        
        # MBB(Minimum Bounding Box) 
        def create_mbb(p1, p2):
            x_min, x_max = min(p1[0], p2[0]), max(p1[0], p2[0])
            y_min, y_max = min(p1[1], p2[1]), max(p1[1], p2[1])
            
            # Generate all x and y points with the given step within the range
            x_points = [x for x in range(int(x_min), int(x_max) + 1, x_step)]
            y_points = [y for y in range(int(y_min), int(y_max) + 1, y_step)]
            # Create all grid points
            return [(x, y) for x in x_points for y in y_points]
        

        # Create mapping between coordinates and node IDs
        coord_to_id = {(src_x, src_y): src_id}  # Initialize with source
        for dst_id in dst_set:
            dst_x = g.nodes['node'].data['x'][dst_id].item()
            dst_y = g.nodes['node'].data['y'][dst_id].item()
            coord_to_id[(dst_x, dst_y)] = dst_id

        steiner_id_counter = g.num_nodes('node') + len(all_steiner_points)
        
        # Create mapping between P and P_cand
        if len(P_prime) == 2:
            pA, pB = P_prime
            net_edges.append((pA, pB))
        elif len(P_prime) > 2:
            pA, pB = closest_pair(P)
            P_prime.remove(pA)
            P_prime.remove(pB)
            pMBB, pC = pA, pB
            while P_prime:
                curr_mbb = create_mbb(pA, pB)
                min_dist = float('inf')
                closest = None
                for mbb_point in curr_mbb:
                    for p in P_prime:
                        dist = abs(mbb_point[0] - p[0]) + abs(mbb_point[1] - p[1])
                        if dist < min_dist:
                            min_dist = dist
                            closest = (mbb_point, p)
                
                pMBB, pC = closest

                # Add Steiner point to mappings
                if pMBB not in coord_to_id:
                    coord_to_id[pMBB] = steiner_id_counter
                    steiner_id_counter += 1
                    net_steiner_points.append(pMBB)
                    all_steiner_points.append(pMBB)
                elif pMBB in P_prime:
                    net_steiner_points.append(pMBB)
                    all_steiner_points.append(pMBB)

                edges_to_add = [(pA, pMBB), (pB, pMBB), (pMBB, pC)]
                for edge in edges_to_add:
                    if edge not in net_edges:
                        net_edges.append(edge)
                if pC in P_prime:
                    P_prime.remove(pC)
                pA, pB = pMBB, pC

        # Add edges using the coordinate mapping
        for edge in net_edges:
            if edge[0] not in coord_to_id or edge[1] not in coord_to_id:
                # Convert floating point coordinates to match exactly
                edge0_rounded = (float(edge[0][0]), float(edge[0][1]))
                edge1_rounded = (float(edge[1][0]), float(edge[1][1]))
                
                if edge0_rounded not in coord_to_id or edge1_rounded not in coord_to_id:
                    print(f"Missing coordinate mapping for edge: {edge}")
                    continue
                    
                start_id = coord_to_id[edge0_rounded]
                end_id = coord_to_id[edge1_rounded]
            else:
                start_id = coord_to_id[edge[0]]
                end_id = coord_to_id[edge[1]]
                
            all_edges.append((start_id, end_id))
            all_edges.append((end_id, start_id))  # Add reverse direction for bidirectional graph

        # Sort paths to ensure they're in correct order
        for (src, dst) in src_to_dst_coord_path.keys():
            src_coord = (g.nodes['node'].data['x'][src].item(), 
                        g.nodes['node'].data['y'][src].item())
            dst_coord = (g.nodes['node'].data['x'][dst].item(), 
                        g.nodes['node'].data['y'][dst].item())
            
            # Ensure path starts with source and ends with destination
            if src_to_dst_coord_path[(src, dst)][0] != src_coord:
                src_to_dst_coord_path[(src, dst)].insert(0, src_coord)
                src_to_dst_node_path[(src, dst)].insert(0, src)
            if src_to_dst_coord_path[(src, dst)][-1] != dst_coord:
                src_to_dst_coord_path[(src, dst)].append(dst_coord)
                src_to_dst_node_path[(src, dst)].append(dst)

        # net별 엣지 정보에 노드 ID 정보 추가
        node_to_id = {src_id: (src_x, src_y)}  # node_to_id mapping for all src and dst pins
        for dst_id in dst_set:
            dst_x = g.nodes['node'].data['x'][dst_id].item()
            dst_y = g.nodes['node'].data['y'][dst_id].item()
            node_to_id[dst_id] = (dst_x, dst_y)

        # assign ID to Steiner point
        steiner_id_offset = g.num_nodes('node') + len(all_steiner_points) - len(net_steiner_points)
        for i, spoint in enumerate(net_steiner_points):
            node_to_id[steiner_id_offset + i] = spoint

        # 현재 net의 모든 좌표에 대한 역매핑 생성
        coord_to_id = {coord: node_id for node_id, coord in node_to_id.items()}

        # 바로 그래프에 연결 관계 추가
        for edge in net_edges:
            if edge[0] not in coord_to_id or edge[1] not in coord_to_id:
                print(f"Missing coordinate mapping for edge: {edge}")
                continue
            start_id = coord_to_id[edge[0]]
            end_id = coord_to_id[edge[1]]
            all_edges.append((start_id, end_id))
            all_edges.append((end_id, start_id))  # Add reverse direction for bidirectional graph
    
    # 새로운 그래프 생성 (simplified edge types)
    new_g = dgl.heterograph({
        ('node', 'snet_out', 'node'): ([], []),
        ('node', 'snet_in', 'node'): ([], [])
    })
    
    # add all nodes (original + steiner) as 'node' type
    total_nodes = g.num_nodes('node') + len(all_steiner_points)
    new_g.add_nodes(total_nodes, ntype='node')
    
    # copy original node features 
    for key in g.nodes['node'].data.keys():
        if not key.startswith('_'):  # Skip internal keys
            orig_data = g.nodes['node'].data[key]
            if len(orig_data.shape) > 1:
                # 2D tensor case
                new_shape = (total_nodes,) + orig_data.shape[1:]
                new_data = torch.zeros(new_shape, dtype=orig_data.dtype)
                new_data[:g.num_nodes('node')] = orig_data
            else:
                # 1D tensor case
                new_data = torch.zeros(total_nodes, dtype=orig_data.dtype)
                new_data[:g.num_nodes('node')] = orig_data
            new_g.nodes['node'].data[key] = new_data

    # add steiner point coordinates
    steiner_x = torch.tensor([p[0] for p in all_steiner_points], dtype=torch.float32)
    steiner_y = torch.tensor([p[1] for p in all_steiner_points], dtype=torch.float32)
    
    
    # Assign coordinates to the correct range of indices
    start_idx = g.num_nodes('node')
    end_idx = start_idx + len(all_steiner_points)
    new_g.nodes['node'].data['x'][start_idx:end_idx] = steiner_x
    new_g.nodes['node'].data['y'][start_idx:end_idx] = steiner_y

    # Create temporary undirected graph to find paths
    G = nx.Graph()
    G.add_edges_from(all_edges)

    # Get original src and dst nodes
    src_nodes, dst_nodes = g.edges(etype='net_out')
    directed_edges = []
    
    # Find paths from src to dst and add directed edges
    for src, dst in zip(src_nodes, dst_nodes):
        src_id = src.item()
        dst_id = dst.item()
        if nx.has_path(G, src_id, dst_id):
            path = nx.shortest_path(G, src_id, dst_id)
            # Add edges along the path direction
            for i in range(len(path)-1):
                directed_edges.append((path[i], path[i+1]))

    # Add all directed edges to the new graph
    if directed_edges:
        src, dst = zip(*directed_edges)
        # Add edges in src->dst direction as snet_out
        new_g.add_edges(torch.tensor(src), torch.tensor(dst), 
                       etype=('node', 'snet_out', 'node'))
        # Add edges in dst->src direction as snet_in
        new_g.add_edges(torch.tensor(dst), torch.tensor(src), 
                       etype=('node', 'snet_in', 'node'))
    
    return new_g, all_steiner_points, all_edges, g.num_nodes('node')
