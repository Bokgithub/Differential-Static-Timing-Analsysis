benchmarks = '''
s44
'''

import pickle as pk
import dgl
import circuitgraph as cg
import circuitgraph.parsing as parser
import copy
import networkx as nx

def load_lib(filename):
    # Use a breakpoint in the code line below to debug your script.
    f1 = open(filename, 'rb')
    dmp = pk.load(f1)
    f1.close()
    return dmp

libdump = load_lib('../data/libdump_bilin.bin')
ntypes = ['node', 'virtual_node']
etypes = ['cell_out', 'net_in', 'net_out', 'net_shared']
def preprocess_verilog(netlist):
    """Remove vector declarations and empty declarations from the netlist"""
    lines = netlist.split('\n')
    processed_lines = []
    
    # Keep track of module ports
    module_ports = []
    
    for line in lines:
        line = line.strip()
        # Skip empty lines and comments
        if not line or line.startswith('/*') or line.startswith('*/'):
            processed_lines.append(line)
            continue
            
        # Handle module declaration
        if 'module' in line and '(' in line:
            # Extract port list
            start = line.find('(')
            end = line.find(')')
            if start != -1 and end != -1:
                ports = line[start+1:end].split(',')
                module_ports = [p.strip() for p in ports if p.strip()]
            processed_lines.append(line)
            # Add port declarations right after module
            for port in module_ports:
                if port in ['clk', 'rst']:  # Common input signals
                    processed_lines.append(f"input {port};")
                elif port == 'y':  # Common output signal
                    processed_lines.append(f"output {port};")
                elif port == 'x':  # x is usually input
                    processed_lines.append(f"input {port};")
                elif port == 'a':  # a is usually input
                    processed_lines.append(f"input {port};")
                else:  # Other signals
                    processed_lines.append(f"input {port};")
            continue
            
        # Process input/output/wire declarations
        if any(keyword in line for keyword in ['input', 'output', 'wire']):
            # Remove vector declarations and get all words
            words = []
            for word in line.split():
                if '[' in word or ']' in word:
                    continue
                if word in ['input', 'output', 'wire']:
                    words.append(word)
                elif ';' not in word:  # Keep identifiers, skip semicolon
                    words.append(word)
            
            # Only keep lines with actual declarations
            if len(words) > 1:  # Has both keyword and identifier
                processed_lines.append(' '.join(words) + ';')
        else:
            # Handle instance connections (e.g., .A2(a[21]) -> .A2(a_21))
            if '(' in line and '[' in line:
                # Convert indexed signals to flattened format
                while '[' in line:
                    start = line.find('[')
                    end = line.find(']', start)
                    if start != -1 and end != -1:
                        signal_name = line[line.rfind('(', 0, start)+1:start]
                        index = line[start+1:end]
                        line = line[:line.rfind('(', 0, start)+1] + f"{signal_name}_{index}" + line[end+1:]
            processed_lines.append(line)
    
    # Debug print to see the processed netlist
    print("\nProcessed module ports:", module_ports)
    result = '\n'.join(processed_lines)
    print("\nFirst 20 lines of processed netlist:")
    print('\n'.join(result.split('\n')[:20]))
    
    return result
def get_circuit_graph_and_instances(netlist, blackboxes):
    netlist = preprocess_verilog(netlist)
    circuit = parser.parse_verilog_netlist(netlist, blackboxes)
    # Clone underlying Networkx graph
    graph = copy.deepcopy(circuit.graph)
    return graph, circuit.blackboxes

def set_pin_node_edge_features(graph):
    # Add node feat1 and feat2 as zeros
    node_feat1 = {node: 0 for node in graph.nodes}
    nx.set_node_attributes(graph, node_feat1, 'feat1')
    nx.set_node_attributes(graph, node_feat1, 'feat2')

    #Circuitgraph adds a buffer node for each net
    #Remove that node and add net_out and net_in instead
    remove_node_list = []
    remove_edge_list = []
    for n in graph.nodes:
        plist = list(graph.predecessors(n))
        slist = list(graph.successors(n))
        
        # for inputs
        if graph.nodes[n]['type'] == 'input':
            for s in slist:
                graph.add_edge(s, n, en=n, _TYPE=1)
                graph.edges[(n, s)]['_TYPE'] = 2
                graph.edges[(n, s)]['en'] = n
                graph.nodes[n]['feat1'] = 1 if ('clk' in n or 'clock' in n or 'CLK' in n or 'Clk' in n) else 0
        
        # for outputs
        elif graph.nodes[n]['output']:
            graph.nodes[n]['feat1'] = 1
            # Skip if no predecessors
            if not plist:
                continue
                
            pred = plist[0]  # Get the first predecessor
            # Set edge type for predecessor -> n
            graph.edges[(pred, n)]['_TYPE'] = 2
            graph.edges[(pred, n)]['en'] = n
            
            # Handle successors
            if not slist:  # No successors
                graph.add_edge(n, pred, en=n, _TYPE=1)
            else:
                # Remove existing edges to successors
                remove_edge_list.extend((n, s) for s in slist)
                # Add new edges
                for succ in slist:
                    graph.add_edge(succ, pred, en=n, _TYPE=1)
                    graph.add_edge(pred, succ, en=n, _TYPE=2)

        # for buffers
        elif graph.nodes[n]['type'] == 'buf':
            if not plist:  # Skip if no predecessors
                continue
            pred = plist[0]
            for succ in slist:
                graph.add_edge(succ, pred, en=n, _TYPE=1)
                graph.add_edge(pred, succ, en=n, _TYPE=2)
            remove_node_list.append(n)

    # Remove nodes and edges
    graph.remove_nodes_from(remove_node_list)
    graph.remove_edges_from(remove_edge_list)
    
    # Set default node type
    for n in graph.nodes:
        graph.nodes[n]['_TYPE'] = 0

    # Fill all edge feat1 with zeros
    edge_feat1 = {edge: 0 for edge in graph.edges()}
    nx.set_edge_attributes(graph, edge_feat1, 'feat1')


def set_inst_node_and_edge_features(benchmark, graph, instances):


    for n, b in instances.items():
        cellname = b.name

        tokens = cellname.split('_')
        celltype = '_'.join(tokens[:-1])
        cellsize = int(tokens[-1])
        cellsizes = libdump['size_list'][celltype]
        sz = cellsizes.index(cellsize)
        cn = libdump['cell_types'].index(celltype)
        fanouts = b.output_set
        fanins = b.input_set
        # add cell nodes
        graph.add_node(n, _TYPE=1, feat1=cn, feat2=sz)
        edges = [(n, f'{n}.{node}') for node in fanins | fanouts]
        graph.add_edges_from(edges, _TYPE=3, feat1=0)
        # add arcs
        celldata = cells[cellname]
        cellindex = celldata[0]
        cellfo = celldata[1]
        cellfi = celldata[2]
        flipflop = 'D' in cellfi and 'Q' in cellfo

        for j, fo in enumerate(cellfo):
            for i, fi in enumerate(cellfi):
                if fo in fanouts and fi in fanins:
                    graph.nodes[f'{n}.{fi}']['feat2'] = i
                    graph.nodes[f'{n}.{fo}']['feat2'] = j
                    if not flipflop:
                        pos = abs(sum(libdump['arc_emb'][cn, i, j, 0, sz]))
                        neg = abs(sum(libdump['arc_emb'][cn, i, j, 1, sz]))
                        # add positive arc
                        if pos < 1e6:
                            graph.add_edge(f'{n}.{fi}', f'{n}.{fo}', _TYPE=0, feat1=0)
                        # add negative arc
                        if neg < 1e6:
                            graph.add_edge(f'{n}.{fi}', f'{n}.{fo}', _TYPE=0, feat1=1)
                    else: #flipflop
                        if 'D' in fi:
                            graph.nodes[f'{n}.{fi}']['feat1'] = 1
                        elif 'CLK' in fi or 'GATE' in fi:
                            pos = abs(sum(libdump['arc_emb'][cn, i, j, 0, sz]))
                            neg = abs(sum(libdump['arc_emb'][cn, i, j, 1, sz]))
                            # add positive edge
                            if pos < 1e6:
                                graph.add_edge(f'{n}.{fi}', f'{n}.{fo}', _TYPE=0, feat1=2)
                            # add negative edge
                            if neg < 1e6:
                                graph.add_edge(f'{n}.{fi}', f'{n}.{fo}', _TYPE=0, feat1=3)


def lable_graph_with_ids(graph):
    node_ids = {node: idx for idx, node in enumerate(graph.nodes)}
    edge_ids = {edge: idx for idx, edge in enumerate(graph.edges)}

    # IDs for back annotation
    nx.set_node_attributes(graph, node_ids, 'id')
    nx.set_edge_attributes(graph, edge_ids, 'id')

def make_dgl_graph(graph):
    # Initialize default attributes for all nodes and edges
    default_node_attrs = {
        '_TYPE': 0,
        'id': 0,
        'feat1': 0,
        'feat2': 0
    }
    
    default_edge_attrs = {
        '_TYPE': 0,
        'id': 0,
        'feat1': 0
    }
    
    # Set default attributes for all nodes
    for node in graph.nodes():
        for attr, value in default_node_attrs.items():
            if attr not in graph.nodes[node]:
                graph.nodes[node][attr] = value
                
    # Set default attributes for all edges
    for edge in graph.edges():
        for attr, value in default_edge_attrs.items():
            if attr not in graph.edges[edge]:
                graph.edges[edge][attr] = value
    
    # Make a homogeneous DGL graph from Networkx
    g = dgl.from_networkx(graph, 
                         node_attrs=['_TYPE', 'id', 'feat1', 'feat2'],
                         edge_attrs=['_TYPE', 'id', 'feat1'])
    
    # Convert homogeneous DGL graph to Heterogeneous DGL graph
    h = dgl.to_heterogeneous(g, ntypes, etypes)
    return h

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    benchmarks = ['s44']

    cells = libdump['cell_table']
    #List of library cells as blackboxes
    bb = [cg.BlackBox(name, values[2], values[1]) for name, values in cells.items()]

    for b in benchmarks:
        verilog_in =  f'./benchmarks/verilog/{b}.v'
        graphml = f'./benchmarks/network/{b}.graphml'
        dgl_graph = f'./benchmarks/dgl/{b}.graph.bin'
        with open(verilog_in, "r") as file:
            # Read the contents of the file
            netlist = file.read()
            print(b)


        nx_graph, inst_list = get_circuit_graph_and_instances(netlist, bb)
        set_pin_node_edge_features(nx_graph)
        set_inst_node_and_edge_features(b, nx_graph, inst_list)
        lable_graph_with_ids(nx_graph)
        g = make_dgl_graph(nx_graph)
        nx.write_graphml(nx_graph, graphml)
        dgl.save_graphs(dgl_graph, [g])





