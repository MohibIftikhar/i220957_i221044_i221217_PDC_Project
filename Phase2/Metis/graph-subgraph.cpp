#include <metis.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <stdexcept>
#include <cstdlib>

void read_graph(const std::string& filename, std::vector<idx_t>& xadj, std::vector<idx_t>& adjncy, std::vector<idx_t>& adjwgt, std::vector<std::tuple<idx_t, idx_t, idx_t>>& edges) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open graph file: " + filename);
    }

    std::vector<std::vector<std::pair<idx_t, idx_t>>> adj_list;
    idx_t u, v, w;
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
            adjwgt.push_back(w);
        }
        xadj.push_back(adjncy.size());
    }

    std::cout << "Read graph with " << nvtxs << " vertices and " << adjncy.size() / 2 << " edges\n";
}

void extract_subgraphs(const std::vector<idx_t>& part, const std::vector<std::tuple<idx_t, idx_t, idx_t>>& edges, idx_t nparts) {
    std::vector<std::vector<std::tuple<idx_t, idx_t, idx_t>>> subgraphs(nparts);
    std::vector<std::set<idx_t>> local_nodes(nparts);
    std::vector<std::set<idx_t>> ghost_nodes(nparts);

    for (const auto& [u, v, w] : edges) {
        idx_t u_part = part[u];
        idx_t v_part = part[v];
        if (u_part == v_part) {
            // Intra-partition edge
            subgraphs[u_part].emplace_back(u, v, w);
            local_nodes[u_part].insert(u);
            local_nodes[u_part].insert(v);
        } else {
            // Cross-partition edge
            subgraphs[u_part].emplace_back(u, v, w);
            local_nodes[u_part].insert(u);
            ghost_nodes[u_part].insert(v);
            subgraphs[v_part].emplace_back(u, v, w);
            local_nodes[v_part].insert(v);
            ghost_nodes[v_part].insert(u);
        }
    }

    for (idx_t part = 0; part < nparts; ++part) {
        std::ofstream out("subgraph_" + std::to_string(part) + ".txt");
        for (const auto& [u, v, w] : subgraphs[part]) {
            out << u << " " << v << " " << w << "\n";
        }
        out.close();
        std::cout << "Wrote subgraph_" << part << ".txt with " << subgraphs[part].size() << " edges\n";

        std::ofstream nodes_out("subgraph_" + std::to_string(part) + "_nodes.txt");
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
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <number_of_partitions>\n";
        return 1;
    }

    // Convert and validate number of partitions
    char* endptr;
    idx_t nparts = std::strtol(argv[1], &endptr, 10);
    if (*endptr != '\0' || nparts <= 0) {
        std::cerr << "Error: Number of partitions must be a positive integer\n";
        return 1;
    }

    // Graph data
    std::vector<idx_t> xadj, adjncy, adjwgt;
    std::vector<std::tuple<idx_t, idx_t, idx_t>> edges;
    try {
        read_graph("graph.txt", xadj, adjncy, adjwgt, edges);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    idx_t nvtxs = xadj.size() - 1;
    if (nparts > nvtxs) {
        std::cerr << "Error: Number of partitions (" << nparts << ") cannot exceed number of vertices (" << nvtxs << ")\n";
        return 1;
    }

    idx_t ncon = 1; // Number of constraints
    std::vector<idx_t> part(nvtxs);
    idx_t objval;

    // METIS options
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT; // Minimize edge cut
    options[METIS_OPTION_UFACTOR] = 100; // 10% imbalance

    // Partition graph
    int ret = METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adjncy.data(), nullptr, nullptr, adjwgt.data(),
                                  &nparts, nullptr, nullptr, options, &objval, part.data());
    if (ret != METIS_OK) {
        std::cerr << "METIS partitioning failed\n";
        return 1;
    }

    // Output partitions
    std::cout << "Edge cut: " << objval << "\n";
    for (idx_t i = 0; i < nvtxs; ++i) {
        std::cout << "Vertex " << i << " -> Partition " << part[i] << "\n";
    }

    // Extract subgraphs with ghost nodes
    extract_subgraphs(part, edges, nparts);

    return 0;
}