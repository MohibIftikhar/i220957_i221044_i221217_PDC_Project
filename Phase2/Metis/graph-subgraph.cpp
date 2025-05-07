#include <metis.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <tuple>
#include <stdexcept>
#include <cstdlib>
#include <chrono>

void read_graph(
    const std::string& filename,
    std::vector<idx_t>& xadj,
    std::vector<idx_t>& adjncy,
    std::vector<idx_t>& adjwgt,
    std::vector<std::tuple<idx_t, idx_t, double>>& edges)
{
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open graph file: " + filename);
    }

    std::vector<std::vector<std::pair<idx_t, double>>> adj_list;
    idx_t u, v;
    double w;
    idx_t max_vertex = -1;

    while (file >> u >> v >> w) {
        if (u < 0 || v < 0) {
            throw std::runtime_error("Negative vertex ID found");
        }
        if (w < 0) {
            throw std::runtime_error("Negative weight found");
        }
        max_vertex = std::max(max_vertex, std::max(u, v));
        if (adj_list.size() <= static_cast<size_t>(max_vertex)) {
            adj_list.resize(max_vertex + 1);
        }
        adj_list[u].emplace_back(v, w);
        adj_list[v].emplace_back(u, w);
        edges.emplace_back(u, v, w);
    }

    idx_t nvtxs = adj_list.size();
    xadj.push_back(0);
    for (idx_t i = 0; i < nvtxs; ++i) {
        for (const auto& [v, w] : adj_list[i]) {
            adjncy.push_back(v);
            adjwgt.push_back(static_cast<idx_t>(w));
        }
        xadj.push_back(adjncy.size());
    }

    std::cout << "Read graph with " << nvtxs << " vertices and " << adjncy.size() / 2 << " edges\n";
}

void extract_subgraphs(const std::vector<idx_t>& part,
                       const std::vector<std::tuple<idx_t, idx_t, double>>& edges,
                       idx_t nparts) {
    std::vector<std::vector<std::tuple<idx_t, idx_t, double>>> subgraphs(nparts);
    std::vector<std::set<idx_t>> local_nodes(nparts);
    std::vector<std::set<idx_t>> ghost_nodes(nparts);

    for (const auto& [u, v, w] : edges) {
        idx_t u_part = part[u];
        idx_t v_part = part[v];
        if (u_part == v_part) {
            subgraphs[u_part].emplace_back(u, v, w);
            local_nodes[u_part].insert(u);
            local_nodes[u_part].insert(v);
        } else {
            subgraphs[u_part].emplace_back(u, v, w);
            local_nodes[u_part].insert(u);
            ghost_nodes[u_part].insert(v);
            subgraphs[v_part].emplace_back(u, v, w);
            local_nodes[v_part].insert(v);
            ghost_nodes[v_part].insert(u);
        }
    }

    for (idx_t part = 0; part < nparts; ++part) {
        std::ofstream out("../Parallel/subgraph_" + std::to_string(part) + ".txt");
        for (const auto& [u, v, w] : subgraphs[part]) {
            out << u << " " << v << " " << w << "\n";
        }
        out.close();
        std::cout << "Wrote subgraph_" << part << ".txt with " << subgraphs[part].size() << " edges\n";

        std::ofstream nodes_out("../Parallel/subgraph_" + std::to_string(part) + "_nodes.txt");
        nodes_out << "# Local nodes\n";
        for (idx_t node : local_nodes[part]) {
            nodes_out << node << "\n";
        }
        nodes_out << "# Ghost nodes\n";
        for (idx_t node : ghost_nodes[part]) {
            nodes_out << node << "\n";
        }
        nodes_out.close();
        std::cout << "Wrote subgraph_" << part << "_nodes.txt with "
                  << local_nodes[part].size() << " local nodes and "
                  << ghost_nodes[part].size() << " ghost nodes\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <graph_file> <number_of_partitions>\n";
        return 1;
    }

    std::string graph_filename = argv[1];
    char* endptr;
    idx_t nparts = std::strtol(argv[2], &endptr, 10);
    if (*endptr != '\0' || nparts <= 0) {
        std::cerr << "Error: Number of partitions must be a positive integer\n";
        return 1;
    }

    std::vector<idx_t> xadj, adjncy, adjwgt;
    std::vector<std::tuple<idx_t, idx_t, double>> edges;

    auto start_total = std::chrono::high_resolution_clock::now();

    auto start_read = std::chrono::high_resolution_clock::now();
    try {
        read_graph(graph_filename, xadj, adjncy, adjwgt, edges);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    auto end_read = std::chrono::high_resolution_clock::now();

    idx_t nvtxs = xadj.size() - 1;
    if (nparts > nvtxs) {
        std::cerr << "Error: Number of partitions (" << nparts << ") cannot exceed number of vertices (" << nvtxs << ")\n";
        return 1;
    }

    idx_t ncon = 1;
    std::vector<idx_t> part(nvtxs);
    idx_t objval;

    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
    options[METIS_OPTION_UFACTOR] = 100;
    options[METIS_OPTION_DBGLVL] = 0; // <--- Add this line to suppress output


    auto start_metis = std::chrono::high_resolution_clock::now();
    int ret = METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adjncy.data(), nullptr, nullptr, adjwgt.data(),
                                  &nparts, nullptr, nullptr, options, &objval, part.data());
    auto end_metis = std::chrono::high_resolution_clock::now();

    if (ret != METIS_OK) {
        std::cerr << "METIS partitioning failed\n";
        return 1;
    }

    auto start_subgraphs = std::chrono::high_resolution_clock::now();
    extract_subgraphs(part, edges, nparts);
    auto end_subgraphs = std::chrono::high_resolution_clock::now();

    std::ofstream vertex_map("../Parallel/vertex_to_partition.txt");
    for (idx_t i = 0; i < nvtxs; ++i) {
        vertex_map << i << " " << part[i] << "\n";
    }
    vertex_map.close();

    auto end_total = std::chrono::high_resolution_clock::now();

    std::cout << "\n=== Performance Summary ===\n";
    std::cout << "Total Vertices       : " << nvtxs << "\n";
    std::cout << "Total Edges          : " << edges.size() << "\n";
    std::cout << "Partitions Requested : " << nparts << "\n";
    std::cout << "Edge Cut (by METIS)  : " << objval << "\n";

    auto duration = [](auto start, auto end) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    };

    std::cout << "\n--- Timing (ms) ---\n";
    std::cout << "Graph Read           : " << duration(start_read, end_read) << " ms\n";
    std::cout << "METIS Partitioning   : " << duration(start_metis, end_metis) << " ms\n";
    std::cout << "Subgraph Extraction  : " << duration(start_subgraphs, end_subgraphs) << " ms\n";
    std::cout << "Total Time           : " << duration(start_total, end_total) << " ms\n";
    std::cout << "============================\n";

    return 0;
}


// g++ graph-subgraph.cpp -o ex -lmetis
// time ./ex ../Datasets/bio-CE/bio-CE-HT.edges 3
// time ./ex ../Datasets/ia-wiki/ia-wiki.edges 2

// time ./ex ../Datasets/bio-h/bio-h.edges 2

