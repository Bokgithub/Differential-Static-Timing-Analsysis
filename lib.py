# lib.py
import re
import dgl
class LibertyParser:
    def __init__(self):
        self.cells = {}  # Dictionary to store cell information
        self.cell_types = []  # cell type 리스트 저장
        self.size_lists = {}  # cell type별 size 리스트 저장
        self.liberty_content = None
        self.wire_load_models = None
        
    def parse_file(self, filepath):
        """Parse liberty file and store cell information"""
        current_cell = None
        current_timing = None
        current_table = None
        with open(filepath, 'r') as f:
            self.liberty_content = f.read()
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Parse cell
                if line.startswith('cell ('):
                    cell_name = re.search(r'cell \("(.+?)"\)', line).group(1)
                    current_cell = {'timing': {}}
                    self.cells[cell_name] = current_cell
                    
                    # Extract cell type and size
                    tokens = cell_name.split('_')
                    try:
                        # Try to convert the last token to int
                        cell_size = int(tokens[-1])
                        cell_type = '_'.join(tokens[:-1])
                    except ValueError:
                        # If last token is not a number, treat entire name as cell_type
                        cell_type = cell_name
                        cell_size = 0  # default size for special cells
                    
                    # Update cell_types list
                    if cell_type not in self.cell_types:
                        self.cell_types.append(cell_type)
                        self.size_lists[cell_type] = []
                    
                    # Update size_lists
                    if cell_size not in self.size_lists[cell_type]:
                        self.size_lists[cell_type].append(cell_size)
                        self.size_lists[cell_type].sort()  # Keep sizes in order
                    
                    continue
                
                # Parse timing block
                if line.startswith('timing ('):
                    if current_cell:
                        current_timing = {}
                        timing_id = len(current_cell['timing'])
                        current_cell['timing'][timing_id] = current_timing
                    continue
                
                # Parse timing tables
                if current_timing is not None:
                    # Parse table type (cell_fall, cell_rise, etc.)
                    table_match = re.search(r'(cell_fall|cell_rise|rise_transition|fall_transition)\s*\("([^"]+)"\)', line)
                    if table_match:
                        table_type = table_match.group(1)
                        table_name = table_match.group(2)
                        current_table = {
                            'name': table_name,
                            'index_1': None,
                            'index_2': None,
                            'values': []
                        }
                        current_timing[table_type] = current_table
                        continue
                    
                    # Parse indices
                    if current_table:
                        if line.startswith('index_1'):
                            current_table['index_1'] = self._parse_array(line)
                        elif line.startswith('index_2'):
                            current_table['index_2'] = self._parse_array(line)
                        elif line.startswith('values'):
                            # Start collecting multi-line values
                            values_str = line
                            while line.strip().endswith('\\'):
                                line = next(f).strip()
                                values_str += line
                            values = self._parse_values(values_str)
                            current_table['values'].append(values)
                            
                            # Check if both index_1 exists and values array is complete
                            if (current_table['index_1'] is not None and 
                                len(current_table['values']) == len(current_table['index_1'])):
                                current_table = None

    def _parse_array(self, line):
        """Parse array of numbers from line"""
        numbers_str = re.search(r'\((.*?)\)', line).group(1)
        # Remove quotes from each number and convert to float
        return [float(x.strip().strip('"')) for x in numbers_str.split(',')]
    
    def _parse_values(self, line):
        """Parse values line which might be split across multiple lines"""
        # Remove 'values' keyword if present
        line = line.replace('values', '').strip()
        # Remove quotes, parentheses, semicolons and backslashes
        line = re.sub(r'[()";]', '', line)
        
        # Split by backslash to get rows
        rows = [row.strip() for row in line.split('\\')]
        
        # Parse each row into numbers
        matrix = []
        for row in rows:
            if row:  # Skip empty rows
                row_values = [float(x.strip()) for x in row.split(',') if x.strip()]
                if row_values:  # Only add non-empty rows
                    matrix.append(row_values)
        
        return matrix
    
    def get_cell(self, cell_name):
        """Get cell information by name"""
        return self.cells.get(cell_name)
    
    def get_timing_table(self, cell_name, timing_id, table_type):
        """Get specific timing table for a cell"""
        cell = self.get_cell(cell_name)
        if cell and timing_id in cell['timing']:
            return cell['timing'][timing_id].get(table_type)
        return None
    
    def get_timing_by_index(self, type_idx, size_idx, timing_id, table_type):
        """Get timing information using numerical indices"""
        if 0 <= type_idx < len(self.cell_types):
            cell_type = self.cell_types[type_idx]
            if cell_type in self.size_lists:
                sizes = self.size_lists[cell_type]
                if 0 <= size_idx < len(sizes):
                    size = sizes[size_idx]
                    cell_name = f"{cell_type}_{size}"
                    return self.get_timing_table(cell_name, timing_id, table_type)
        return None
    
    def lookup_timing(self, g, node_id):
        """Look up timing information for a node in the DGL graph"""
        node_data = g.nodes['node'].data
        type_idx = node_data['feat1'][node_id].item()  # cell type index
        size_idx = node_data['feat2'][node_id].item()  # size index
        
        # Get timing tables for this cell
        cell_fall = self.get_timing_by_index(type_idx, size_idx, 0, 'cell_fall')
        cell_rise = self.get_timing_by_index(type_idx, size_idx, 0, 'cell_rise')
        rise_transition = self.get_timing_by_index(type_idx, size_idx, 0, 'rise_transition')
        fall_transition = self.get_timing_by_index(type_idx, size_idx, 0, 'fall_transition')
        
        return {
            'cell_fall': cell_fall,
            'cell_rise': cell_rise,
            'rise_transition': rise_transition,
            'fall_transition': fall_transition,
            'type': self.cell_types[type_idx] if type_idx < len(self.cell_types) else None,
            'size': self.size_lists.get(self.cell_types[type_idx], [])[size_idx] if type_idx < len(self.cell_types) else None
        }
    def parse_tracks(self, def_content):
        """Parse TRACKS information from DEF file content"""
        tracks_dict = {}
        
        # Find all TRACKS lines
        for line in def_content.split('\n'):
            if line.startswith('TRACKS'):
                # Split the line into parts
                parts = line.split()
                # Get direction (X or Y), values, and layer
                direction = parts[1]  # X or Y
                start_val = int(parts[2])
                do_val = int(parts[4])
                step_val = int(parts[6])
                layer = parts[8]
                
                # Initialize layer dict if not exists
                if layer not in tracks_dict:
                    tracks_dict[layer] = {'X': [], 'Y': []}
                
                # Add values to appropriate direction
                tracks_dict[layer][direction] = [start_val, do_val, step_val]
        
        self.tracks_info = tracks_dict
        return tracks_dict
    def parse_gcellgrid(self, def_content):
        gcellgrid_dict = {'X': [], 'Y': []}
        
        # Find all GCELLGRID lines
        for line in def_content.split('\n'):
            if line.startswith('GCELLGRID'):
                # Split the line into parts
                parts = line.split()
                # Get direction (X or Y) and values
                direction = parts[1]  # X or Y
                start_val = int(parts[2])
                do_val = int(parts[4])
                step_val = int(parts[6])
                
                # Add values to appropriate direction
                gcellgrid_dict[direction] = [start_val, do_val, step_val]
                
        return gcellgrid_dict
    
    def parse_cap_res(self):
        """Parse wire load models from liberty file content to get capacitance and resistance values"""
        if self.liberty_content is None:
            raise ValueError("No liberty file content. Call parse_file() first.")
            
        wire_load_models = {}
        
        # Find all wire_load blocks
        wire_load_pattern = r'wire_load\s*\("([^"]+)"\)\s*{([^}]+)}'
        matches = re.finditer(wire_load_pattern, self.liberty_content, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            model_name = match.group(1).lower()  # Convert to lowercase for consistency
            block_content = match.group(2)
            
            # Extract capacitance and resistance values
            cap_match = re.search(r'capacitance\s*:\s*([\d.e-]+)\s*;', block_content)
            res_match = re.search(r'resistance\s*:\s*([\d.e-]+)\s*;', block_content)
            
            if cap_match and res_match:
                cap_value = float(cap_match.group(1))
                res_value = float(res_match.group(1))
                wire_load_models[model_name] = [cap_value, res_value]
        
        self.wire_load_models = wire_load_models
        return wire_load_models
