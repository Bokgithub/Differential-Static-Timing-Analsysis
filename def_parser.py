import re
import pickle as pk

#load precomputed technology library
def load_lib(filename):
    # Use a breakpoint in the code line below to debug your script.
    f1 = open(filename, 'rb')
    dmp = pk.load(f1)
    f1.close()
    return dmp

def parse_def_file(def_file_path):
    """
    Parse the DEF file to extract instance names and their positions.
    Returns a dictionary mapping instance names to (x, y) positions.
    """
    def_positions = {}
    with open(def_file_path, 'r') as f:
        for line in f:
            # Match lines that define component placements
            # Updated regex to capture instance names like "_102_"
            match = re.match(r'\s*-\s+(\S+)\s+\S+\s+\+\s+(FIXED|PLACED)\s+\(\s*(\d+)\s+(\d+)\s*\)\s+\S+\s*;', line)
            if match:
                instance_name = match.group(1)
                x_pos = float(match.group(3))
                y_pos = float(match.group(4))
                def_positions[instance_name] = (x_pos, y_pos)
    return def_positions

def parse_def_type(def_file_path):
    """
    Parse the DEF file to extract instance names, their positions, and cell types.
    Returns two dictionaries: one mapping instance names to positions, 
    and another mapping instance names to cell types.
    """
    def_positions = {}
    def_types = {}
    with open(def_file_path, 'r') as f:
        for line in f:
            # Match lines that define component placements
            match = re.match(r'\s*-\s+(\S+)\s+(\S+)\s+\+\s+(FIXED|PLACED)\s+\(\s*(\d+)\s+(\d+)\s*\)\s+\S+\s*;', line)
            if match:
                instance_name = match.group(1)
                cell_type = match.group(2).split('sky130_fd_sc_hd__')[-1]  # Extract just the gate type
                x_pos = float(match.group(4))
                y_pos = float(match.group(5))
                def_positions[instance_name] = (x_pos, y_pos)
                def_types[instance_name] = cell_type
    return def_positions, def_types