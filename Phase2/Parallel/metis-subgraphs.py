from collections import defaultdict

def extract_subgraphs(graph_file, part_file, mapping_file, output_prefix, num_partitions=2):
    # Step 1: Read vertex mapping (1-based METIS ID to 0-based original ID)
    vertex_map = {}
    try:
        with open(mapping_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split(',')
                    old_id = int(parts[0].split(':')[1].strip())
                    new_id = int(parts[1].split(':')[1].strip())
                    vertex_map[new_id] = old_id  # Map METIS ID (1-based) to original ID (0-based)
    except FileNotFoundError:
        raise FileNotFoundError(f"Mapping file {mapping_file} not found")

    # Step 2: Read partition assignments
    partitions = {}  # Map original 0-based ID to partition (0 or 1)
    try:
        with open(part_file, 'r') as f:
            for i, line in enumerate(f, 1):  # 1-based indexing for METIS vertices
                if line.strip():
                    part = int(line.strip())
                    if part >= num_partitions:
                        raise ValueError(f"Invalid partition ID {part} for vertex {i}")
                    if i in vertex_map:
                        original_id = vertex_map[i]
                        partitions[original_id] = part
                    else:
                        print(f"Warning: Vertex {i} in {part_file} not found in mapping")
    except FileNotFoundError:
        raise FileNotFoundError(f"Partition file {part_file} not found")

    # Step 3: Read original graph and filter edges by partition
    subgraphs = [[] for _ in range(num_partitions)]  # List of edges for each partition
    nodes_per_partition = [set() for _ in range(num_partitions)]  # Track nodes in each partition
    try:
        with open(graph_file, 'r') as f:
            for line in f:
                if line.strip():
                    u, v, w = line.strip().split()
                    u, v = int(u), int(v)
                    w = float(w)
                    # Check if both vertices are in the same partition
                    if u in partitions and v in partitions and partitions[u] == partitions[v]:
                        part = partitions[u]
                        subgraphs[part].append((u, v, w))
                        nodes_per_partition[part].add(u)
                        nodes_per_partition[part].add(v)
    except FileNotFoundError:
        raise FileNotFoundError(f"Graph file {graph_file} not found")

    # Step 4: Write subgraph files
    for part in range(num_partitions):
        output_file = f"{output_prefix}_{part}.txt"
        with open(output_file, 'w') as f:
            # Write edges
            for u, v, w in subgraphs[part]:
                f.write(f"{u} {v} {w}\n")
        print(f"Wrote subgraph for partition {part} to {output_file} with "
              f"{len(nodes_per_partition[part])} nodes and {len(subgraphs[part])} edges")

try:
    graph_file = "test_data.edges"
    part_file = "graph.metis.part.2"
    mapping_file = "vertex_mapping.txt"
    output_prefix = "subgraph"
    extract_subgraphs(graph_file, part_file, mapping_file, output_prefix, num_partitions=2)
except Exception as e:
    print(f"Error: {e}")