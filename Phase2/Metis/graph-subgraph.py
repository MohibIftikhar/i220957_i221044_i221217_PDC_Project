import math
import argparse
import subprocess
from collections import defaultdict

def convert_to_metis(input_file, output_file, weight_scale=1000, has_header=False):
    adj_list = defaultdict(list)
    edges = set()
    vertices = set()

    try:
        with open(input_file, 'r') as f:
            for line_number, line in enumerate(f, 1):
                if line.strip() and not line.startswith(('#', '%')):
                    if line_number == 1 and has_header:
                        print(f"Skipping header line {line_number}: {line.strip()}")
                        continue
                    parts = line.strip().split()
                    if len(parts) != 3:
                        raise ValueError(f"Invalid line format at line {line_number}: {line.strip()}")
                    try:
                        u, v, w = int(parts[0]), int(parts[1]), float(parts[2])
                    except ValueError:
                        raise ValueError(f"Invalid data at line {line_number}: {line.strip()}")
                    if u < 0 or v < 0:
                        raise ValueError(f"Negative vertex ID at line {line_number}: {line.strip()}")
                    if w < 0:
                        raise ValueError(f"Negative weight at line {line_number}: {line.strip()}")
                    w_int = round(w * weight_scale)
                    vertices.add(u)
                    vertices.add(v)
                    adj_list[u].append((v, w_int))
                    adj_list[v].append((u, w_int))
                    edge = tuple(sorted([u, v]))
                    edges.add(edge)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file {input_file} not found")

    if len(vertices) == 0:
        raise ValueError("No vertices found in the input file")

    vertex_map = {old_id: new_id + 1 for new_id, old_id in enumerate(sorted(vertices))}
    n_vertices = len(vertices)
    n_edges = len(edges)

    new_adj_list = defaultdict(list)
    for old_u in adj_list:
        new_u = vertex_map[old_u]
        for old_v, w in adj_list[old_u]:
            new_v = vertex_map[old_v]
            new_adj_list[new_u].append((new_v, w))

    with open(output_file, 'w') as f:
        f.write(f"{n_vertices} {n_edges} 1\n")
        for i in range(1, n_vertices + 1):
            if i in new_adj_list:
                neighbors = sorted(new_adj_list[i], key=lambda x: x[0])
                line = ' '.join(f"{v} {w}" for v, w in neighbors)
                f.write(f"{line}\n")
            else:
                f.write("\n")

    mapping_file = "vertex_mapping.txt"
    with open(mapping_file, "w") as f:
        for old_id, new_id in vertex_map.items():
            f.write(f"Old ID: {old_id}, New ID: {new_id}\n")

    print(f"Converted {input_file} to {output_file} with {n_vertices} vertices and {n_edges} edges.")
    print(f"Vertex mapping saved to {mapping_file}")

    return vertex_map, n_vertices, n_edges

def run_metis(metis_file, num_partitions):
    part_file = f"{metis_file}.part.{num_partitions}"
    try:
        result = subprocess.run(
            ["gpmetis", metis_file, str(num_partitions)],
            check=True, capture_output=True, text=True
        )
        print(f"Ran gpmetis to produce {part_file}")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"gpmetis failed: {e.stderr}")
    except FileNotFoundError:
        raise FileNotFoundError("gpmetis not found in PATH. Ensure METIS is installed.")
    return part_file

def extract_subgraphs(graph_file, part_file, vertex_map, output_prefix, num_partitions, has_header=False):
    vertex_map_inv = {new_id: old_id for old_id, new_id in vertex_map.items()}  # 1-based to 0-based

    partitions = {}
    partition_counts = [0] * num_partitions
    try:
        with open(part_file, 'r') as f:
            for i, line in enumerate(f, 1):  # 1-based METIS vertex IDs
                if line.strip():
                    part = int(line.strip())
                    if part >= num_partitions:
                        raise ValueError(f"Invalid partition ID {part} for vertex {i}")
                    if i in vertex_map_inv:
                        original_id = vertex_map_inv[i]
                        partitions[original_id] = part
                        partition_counts[part] += 1
                    else:
                        print(f"Warning: Vertex {i} in {part_file} not found in mapping")
    except FileNotFoundError:
        raise FileNotFoundError(f"Partition file {part_file} not found")

    print("Partition assignments:")
    for part in range(num_partitions):
        print(f"Partition {part}: {partition_counts[part]} nodes")
    print(f"Total vertices assigned: {sum(partition_counts)}")

    subgraphs = [[] for _ in range(num_partitions)]
    nodes_per_partition = [set() for _ in range(num_partitions)]
    try:
        with open(graph_file, 'r') as f:
            for line_number, line in enumerate(f, 1):
                if line.strip() and not line.startswith(('#', '%')):
                    if line_number == 1 and has_header:
                        print(f"Skipping header line {line_number}: {line.strip()}")
                        continue
                    parts = line.strip().split()
                    if len(parts) != 3:
                        print(f"Skipping invalid line {line_number}: {line.strip()}")
                        continue
                    try:
                        u, v, w = int(parts[0]), int(parts[1]), float(parts[2])
                        print(f"Processing edge: {u} {v} {w}")
                    except ValueError:
                        print(f"Skipping invalid data at line {line_number}: {line.strip()}")
                        continue
                    if u in partitions and v in partitions and partitions[u] == partitions[v]:
                        part = partitions[u]
                        subgraphs[part].append((u, v, w))
                        nodes_per_partition[part].add(u)
                        nodes_per_partition[part].add(v)
                    else:
                        print(f"Edge {u} {v} not included: u in {partitions.get(u, 'N/A')}, v in {partitions.get(v, 'N/A')}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Graph file {graph_file} not found")

    for part in range(num_partitions):
        output_file = f"{output_prefix}_{part}.txt"
        with open(output_file, 'w') as f:
            for u, v, w in sorted(subgraphs[part]):
                f.write(f"{u} {v} {w}\n")
        print(f"Wrote subgraph for partition {part} to {output_file} with "
              f"{len(nodes_per_partition[part])} nodes and {len(subgraphs[part])} edges")

        node_file = f"{output_prefix}_{part}_nodes.txt"
        with open(node_file, 'w') as f:
            for node in sorted(nodes_per_partition[part]):
                f.write(f"{node}\n")
        print(f"Wrote {node_file} with {len(nodes_per_partition[part])} nodes")

def main():
    parser = argparse.ArgumentParser(description="Convert graph to METIS format, partition with gpmetis, and extract subgraphs.")
    parser.add_argument("input_file", type=str, help="Input graph file (format: u v w per line)")
    parser.add_argument("output_metis_file", type=str, help="Output METIS file")
    parser.add_argument("subgraph_prefix", type=str, help="Prefix for subgraph output files (e.g., 'subgraph' for subgraph_0.txt)")
    parser.add_argument("--num-partitions", type=int, default=2, help="Number of partitions for METIS (default: 2)")
    parser.add_argument("--weight-scale", type=int, default=1000, help="Scale factor for edge weights (default: 1000)")
    parser.add_argument("--has-header", action="store_true", help="Skip the first line of the input file if it’s a header")

    args = parser.parse_args()

    try:
        vertex_map, n_vertices, n_edges = convert_to_metis(
            args.input_file, args.output_metis_file, args.weight_scale, args.has_header
        )
        part_file = run_metis(args.output_metis_file, args.num_partitions)
        extract_subgraphs(
            args.input_file, part_file, vertex_map, args.subgraph_prefix, args.num_partitions, args.has_header
        )
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()