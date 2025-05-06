import math
import argparse
from collections import defaultdict

def convert_to_metis(input_file, output_file, weight_scale=1000):
    # Step 1: Read input and collect vertices and edges
    adj_list = defaultdict(list)  # Store (neighbor, weight) for each vertex
    edges = set()  # Track unique undirected edges
    vertices = set()  # Track all vertices

    try:
        with open(input_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) != 3:
                        raise ValueError(f"Invalid line format: {line.strip()}")
                    u, v, w = parts
                    try:
                        u, v = int(u), int(v)
                        w = float(w)
                    except ValueError:
                        raise ValueError(f"Invalid data in line: {line.strip()}")
                    w_int = round(w * weight_scale)  # Scale and round weight
                    vertices.add(u)
                    vertices.add(v)
                    adj_list[u].append((v, w_int))
                    adj_list[v].append((u, w_int))  # Undirected edge
                    edge = tuple(sorted([u, v]))  # Count each edge once
                    edges.add(edge)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file {input_file} not found")

    # Step 2: Map 0-based vertex IDs to 1-based consecutive IDs
    vertex_map = {old_id: new_id + 1 for new_id, old_id in enumerate(sorted(vertices))}
    n_vertices = len(vertices)
    n_edges = len(edges)

    # Validate vertex count
    if n_vertices == 0:
        raise ValueError("No vertices found in the input file")

    # Step 3: Convert adjacency lists to 1-based indices
    new_adj_list = defaultdict(list)
    for old_u in adj_list:
        new_u = vertex_map[old_u]
        for old_v, w in adj_list[old_u]:
            new_v = vertex_map[old_v]
            new_adj_list[new_u].append((new_v, w))

    # Step 4: Write METIS file
    with open(output_file, 'w') as f:
        # Write header: n_vertices, n_edges, fmt=1 (edge weights)
        f.write(f"{n_vertices} {n_edges} 1\n")
        # Write adjacency list for each vertex (1 to n_vertices)
        for i in range(1, n_vertices + 1):
            if i in new_adj_list:
                # Sort neighbors by vertex ID for consistency
                neighbors = sorted(new_adj_list[i], key=lambda x: x[0])
                line = ' '.join(f"{v} {w}" for v, w in neighbors)
                f.write(f"{line}\n")
            else:
                f.write("\n")  # Empty line for isolated vertices

    # Save vertex mapping for reference
    mapping_file = "vertex_mapping.txt"
    with open(mapping_file, "w") as f:
        for old_id, new_id in vertex_map.items():
            f.write(f"Old ID: {old_id}, New ID: {new_id}\n")

    print(f"Converted {input_file} to {output_file} with {n_vertices} vertices and {n_edges} edges.")
    print(f"Vertex mapping saved to {mapping_file}")

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Convert a graph edge list to METIS format.")
    parser.add_argument("input_file", type=str, help="Input graph file (format: u v w per line)")
    parser.add_argument("output_file", type=str, help="Output METIS file")
    parser.add_argument("--weight-scale", type=int, default=1000, 
                        help="Scale factor for edge weights (default: 1000)")
    
    # Parse arguments
    args = parser.parse_args()

    # Run conversion
    try:
        convert_to_metis(args.input_file, args.output_file, args.weight_scale)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()