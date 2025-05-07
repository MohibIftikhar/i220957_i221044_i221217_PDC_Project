#include <iostream>
#include <vector>
#include <queue>
#include <set>
#include <map>
#include <omp.h>
#include <limits>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <chrono>
#include <mpi.h>

using namespace std;

// Structure to represent an edge
struct Edge {
    int u, v;
    double weight;
    Edge() : u(0), v(0), weight(0.0) {}
    Edge(int _u, int _v, double _w = 0.0) : u(_u), v(_v), weight(_w) {}
};

// Structure to represent the SSSP tree
struct SSSPTree {
    vector<int> parent;
    vector<double> dist;
    vector<bool> affected;
    vector<bool> affected_del;
    SSSPTree(int n) : parent(n, -1),
                      dist(n, numeric_limits<double>::infinity()),
                      affected(n, false),
                      affected_del(n, false) {}
};

// Load subgraph data
void LoadSubgraph(int rank, vector<vector<pair<int, double>>>& G, set<int>& local_nodes, set<int>& ghost_nodes, map<int, int>& ghost_to_owner, int& n, int& edge_count) {
    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    map<int, int> global_vertex_to_owner;
    int max_vertex = -1;
    ifstream vertex_map_stream("vertex_to_partition.txt");
    if (!vertex_map_stream.is_open()) {
        cerr << "Rank " << rank << ": Error: Could not open vertex_to_partition.txt" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    string line;
    int line_number = 0;
    while (getline(vertex_map_stream, line)) {
        line_number++;
        if (line.empty() || line[0] == '#') continue;
        istringstream iss(line);
        int vertex, owner;
        if (!(iss >> vertex >> owner)) {
            cerr << "Rank " << rank << ": Error: Malformed data at line " << line_number << " in vertex_to_partition.txt: " << line << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (vertex < 0 || owner < 0 || owner >= num_procs) {
            cerr << "Rank " << rank << ": Error: Invalid data at line " << line_number << " in vertex_to_partition.txt" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        global_vertex_to_owner[vertex] = owner;
        max_vertex = max(max_vertex, vertex);
    }
    vertex_map_stream.close();
    if (max_vertex < 0) {
        cerr << "Rank " << rank << ": Error: No valid vertices found in vertex_to_partition.txt" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    n = max_vertex + 1;

    string edge_file = "subgraph_" + to_string(rank) + ".txt";
    ifstream edge_stream(edge_file);
    if (!edge_stream.is_open()) {
        cerr << "Rank " << rank << ": Error: Could not open " << edge_file << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    vector<Edge> edges;
    line_number = 0;
    while (getline(edge_stream, line)) {
        line_number++;
        if (line.empty() || line[0] == '#') continue;
        istringstream iss(line);
        int u, v;
        double w;
        if (!(iss >> u >> v >> w)) {
            cerr << "Rank " << rank << ": Error: Malformed data at line " << line_number << " in " << edge_file << ": " << line << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (u < 0 || v < 0 || w < 0) {
            cerr << "Rank " << rank << ": Error: Invalid data (u=" << u << ", v=" << v << ", w=" << w << ") at line " << line_number << " in " << edge_file << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        edges.emplace_back(u, v, w);
    }
    edge_stream.close();

    string node_file = "subgraph_" + to_string(rank) + "_nodes.txt";
    ifstream node_stream(node_file);
    if (!node_stream.is_open()) {
        cerr << "Rank " << rank << ": Error: Could not open " << node_file << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    line_number = 0;
    bool reading_local = true;
    while (getline(node_stream, line)) {
        line_number++;
        if (line == "# Local nodes") {
            reading_local = true;
            continue;
        } else if (line == "# Ghost nodes") {
            reading_local = false;
            continue;
        }
        if (line.empty() || line[0] == '#') continue;
        istringstream iss(line);
        int node;
        if (!(iss >> node)) {
            cerr << "Rank " << rank << ": Error: Malformed node data at line " << line_number << " in " << node_file << ": " << line << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (node < 0) {
            cerr << "Rank " << rank << ": Error: Invalid node " << node << " at line " << line_number << " in " << node_file << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (reading_local) {
            local_nodes.insert(node);
        } else {
            if (global_vertex_to_owner.find(node) == global_vertex_to_owner.end()) {
                cerr << "Rank " << rank << ": Error: Ghost node " << node << " at line " << line_number << " has no owner in " << node_file << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            ghost_nodes.insert(node);
            ghost_to_owner[node] = global_vertex_to_owner[node];
        }
    }
    node_stream.close();

    G.resize(n);
    edge_count = 0;
    set<pair<int, int>> added_edges;
    for (const auto& edge : edges) {
        int u = edge.u, v = edge.v;
        double w = edge.weight;
        G[u].push_back({v, w});
        G[v].push_back({u, w});
        if (added_edges.insert({min(u, v), max(u, v)}).second) {
            edge_count++;
        }
    }

    cout << "Rank " << rank << ": Loaded subgraph with " << n << " vertices, "
         << local_nodes.size() << " local nodes, " << ghost_nodes.size() << " ghost nodes, "
         << edge_count << " edges" << endl;
}

// Load changes from changes.txt
void LoadChanges(string filename, int rank, vector<Edge>& Del, vector<Edge>& Ins, int n) {
    if (rank != 0) return; // Only rank 0 reads the file
    ifstream change_stream(filename);
    if (!change_stream.is_open()) {
        cerr << "Rank " << rank << ": Error: Could not open" << filename << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    string line;
    int line_number = 0;
    while (getline(change_stream, line)) {
        line_number++;
        if (line.empty() || line[0] == '#') continue;
        istringstream iss(line);
        char type;
        int u, v;
        double w = 0.0;
        if (!(iss >> type >> u >> v)) {
            cerr << "Rank " << rank << ": Error: Malformed data at line " << line_number << " in changes.txt: " << line << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (u < 0 || v < 0 || u >= n || v >= n) {
            cerr << "Rank " << rank << ": Error: Invalid vertices (u=" << u << ", v=" << v << ") at line " << line_number << " in changes.txt" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (type == 'D' || type == 'd') {
            Del.emplace_back(u, v);
        } else if (type == 'I' || type == 'i') {
            if (!(iss >> w) || w < 0) {
                cerr << "Rank " << rank << ": Error: Invalid or missing weight for insertion at line " << line_number << " in changes.txt: " << line << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            Ins.emplace_back(u, v, w);
        } else {
            cerr << "Rank " << rank << ": Error: Invalid change type '" << type << "' at line " << line_number << " in changes.txt" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    change_stream.close();
    cout << "Rank " << rank << ": Loaded " << Del.size() << " deletions and " << Ins.size() << " insertions" << endl;
}

// Compute initial SSSP tree
void ComputeInitialSSSP(const vector<vector<pair<int, double>>>& G, SSSPTree& T, int source, const set<int>& local_nodes, const set<int>& ghost_nodes, int rank, int num_procs, int n) {
    if (local_nodes.count(source)) {
        T.dist[source] = 0;
        T.parent[source] = -1;
    }
    priority_queue<pair<double, int>, vector<pair<double, int>>, greater<>> pq;
    if (local_nodes.count(source)) {
        pq.push({0, source});
    }

    while (!pq.empty()) {
        double d = pq.top().first;
        int u = pq.top().second;
        pq.pop();

        if (d > T.dist[u]) continue;
        if (!local_nodes.count(u)) continue;

        for (const auto& edge : G[u]) {
            int v = edge.first;
            double w = edge.second;
            if (local_nodes.count(v) && T.dist[v] > T.dist[u] + w) {
                T.dist[v] = T.dist[u] + w;
                T.parent[v] = u;
                pq.push({T.dist[v], v});
            }
        }
    }

    vector<double> global_dist(T.dist.size());
    MPI_Allreduce(T.dist.data(), global_dist.data(), T.dist.size(), MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    T.dist = global_dist;

    bool changed;
    for (int iter = 0; iter < num_procs; ++iter) {
        changed = false;
        for (int v : local_nodes) {
            for (const auto& edge : G[v]) {
                int u = edge.first;
                double w = edge.second;
                if (T.dist[v] > T.dist[u] + w) {
                    T.dist[v] = T.dist[u] + w;
                    T.parent[v] = u;
                    changed = true;
                }
                if (local_nodes.count(u) && T.dist[u] > T.dist[v] + w) {
                    T.dist[u] = T.dist[v] + w;
                    T.parent[u] = v;
                    changed = true;
                }
            }
        }
        MPI_Allreduce(T.dist.data(), global_dist.data(), T.dist.size(), MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        T.dist = global_dist;

        int global_changed;
        MPI_Allreduce(&changed, &global_changed, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        if (!global_changed) break;
    }

    vector<double> local_dist(T.dist.begin(), T.dist.end());
    vector<int> local_parent(T.parent.begin(), T.parent.end());
    for (int v = 0; v < n; ++v) {
        if (!local_nodes.count(v)) {
            local_dist[v] = numeric_limits<double>::infinity();
            local_parent[v] = -1;
        }
    }

    vector<int> global_parent(n);
    MPI_Allreduce(local_dist.data(), global_dist.data(), n, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(local_parent.data(), global_parent.data(), n, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    for (int v = 0; v < n; ++v) {
        T.dist[v] = global_dist[v];
        if (T.dist[v] < numeric_limits<double>::infinity()) {
            T.parent[v] = global_parent[v];
        } else {
            T.parent[v] = -1;
        }
    }
    T.parent[source] = -1;
}

// Add an edge to the graph
void AddEdge(vector<vector<pair<int, double>>>& G, int u, int v, double weight) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #pragma omp critical
    {
        G[u].push_back({v, weight});
        G[v].push_back({u, weight});
    }
}

// Remove an edge from the graph
void RemoveEdge(vector<vector<pair<int, double>>>& G, int u, int v) {
    G[u].erase(
        remove_if(G[u].begin(), G[u].end(),
            [v](const pair<int, double>& e) {
                return e.first == v;
            }),
        G[u].end()
    );
    G[v].erase(
        remove_if(G[v].begin(), G[v].end(),
            [u](const pair<int, double>& e) {
                return e.first == u;
            }),
        G[v].end()
    );
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
}

// Process edge changes
void ProcessCE(vector<vector<pair<int, double>>>& G, SSSPTree& T, const vector<Edge>& Del, const vector<Edge>& Ins, int source, const set<int>& local_nodes, const set<int>& ghost_nodes, map<int, pair<double, int>>& ghost_updates) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < Del.size(); ++i) {
        int u = Del[i].u, v = Del[i].v;
        #pragma omp critical
        {
            RemoveEdge(G, u, v);
        }
        if (local_nodes.count(v)) {
            T.dist[v] = numeric_limits<double>::infinity();
            T.parent[v] = -1;
            T.affected_del[v] = true;
            T.affected[v] = true;
        } else if (local_nodes.count(u)) {
            T.dist[u] = numeric_limits<double>::infinity();
            T.parent[u] = -1;
            T.affected_del[u] = true;
            T.affected[u] = true;
        }
        if (ghost_nodes.count(v)) {
            #pragma omp critical
            {
                ghost_updates[v] = {numeric_limits<double>::infinity(), -1};
            }
        } else if (ghost_nodes.count(u)) {
            #pragma omp critical
            {
                ghost_updates[u] = {numeric_limits<double>::infinity(), -1};
            }
        }
    }

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < Ins.size(); ++i) {
        int u = Ins[i].u, v = Ins[i].v;
        double w = Ins[i].weight;
        AddEdge(G, u, v, w);
        int x = (T.dist[u] < T.dist[v]) ? u : v;
        int y = (x == u) ? v : u;
        double new_dist = T.dist[x] + w;
        if (local_nodes.count(y)) {
            #pragma omp critical
            {
                if (new_dist < T.dist[y] && T.dist[x] < numeric_limits<double>::infinity()) {
                    T.dist[y] = new_dist;
                    T.parent[y] = x;
                    T.affected[y] = true;
                }
            }
        } else if (ghost_nodes.count(y)) {
            #pragma omp critical
            {
                if (new_dist < T.dist[y] && T.dist[x] < numeric_limits<double>::infinity()) {
                    ghost_updates[y] = {new_dist, x};
                }
            }
        }
    }
}

// Communicate ghost node updates
void CommunicateGhostUpdates(SSSPTree& T, const set<int>& local_nodes, const set<int>& ghost_nodes, const map<int, int>& ghost_to_owner, map<int, pair<double, int>>& ghost_updates, int rank, int num_procs) {
    vector<vector<pair<int, pair<double, int>>>> send_buffers(num_procs);
    for (const auto& [v, update] : ghost_updates) {
        int owner = ghost_to_owner.at(v);
        send_buffers[owner].push_back({v, update});
    }

    vector<MPI_Request> send_requests;
    for (int p = 0; p < num_procs; ++p) {
        if (p == rank || send_buffers[p].empty()) continue;
        int count = send_buffers[p].size();
        MPI_Request req;
        MPI_Isend(send_buffers[p].data(), count * sizeof(pair<int, pair<double, int>>), MPI_BYTE, p, 0, MPI_COMM_WORLD, &req);
        send_requests.push_back(req);
    }

    vector<pair<int, pair<double, int>>> recv_buffer;
    MPI_Status status;
    int flag;
    do {
        MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
        if (flag) {
            int source = status.MPI_SOURCE;
            int count;
            MPI_Get_count(&status, MPI_BYTE, &count);
            recv_buffer.resize(count / sizeof(pair<int, pair<double, int>>));
            MPI_Recv(recv_buffer.data(), count, MPI_BYTE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (const auto& [v, update] : recv_buffer) {
                if (local_nodes.count(v) && update.first < T.dist[v]) {
                    T.dist[v] = update.first;
                    T.parent[v] = update.second;
                    T.affected[v] = true;
                    cout << "Rank " << rank << ": Received update for vertex " << v << ": dist = " << update.first << ", parent = " << update.second << endl;
                }
            }
        }
    } while (flag);

    MPI_Waitall(send_requests.size(), send_requests.data(), MPI_STATUSES_IGNORE);
    ghost_updates.clear();
}

// Asynchronous SSSP update
void AsynchronousUpdating(vector<vector<pair<int, double>>>& G, SSSPTree& T, const vector<Edge>& Del, const vector<Edge>& Ins, int source, int A, const set<int>& local_nodes, const set<int>& ghost_nodes, const map<int, int>& ghost_to_owner, int rank, int num_procs) {
    map<int, pair<double, int>> ghost_updates;
    ProcessCE(G, T, Del, Ins, source, local_nodes, ghost_nodes, ghost_updates);
    CommunicateGhostUpdates(T, local_nodes, ghost_nodes, ghost_to_owner, ghost_updates, rank, num_procs);

    // Protect source vertex
    if (local_nodes.count(source)) {
        T.dist[source] = 0;
        T.parent[source] = -1;
        T.affected_del[source] = false;
        T.affected[source] = true;
    }

    // Reset distances for vertices directly affected by deletions
    for (int v = 0; v < G.size(); ++v) {
        if (!local_nodes.count(v) || !T.affected_del[v] || v == source) continue;
        queue<int> Q;
        Q.push(v);
        set<int> visited;
        visited.insert(v);
        T.dist[v] = numeric_limits<double>::infinity();
        T.parent[v] = -1;
        T.affected[v] = true;

        // Only reset vertices that were dependent on the deleted edge
        int level = 0;
        while (!Q.empty() && level <= A) {
            int x = Q.front();
            Q.pop();
            for (const auto& [c, w] : G[x]) {
                if (local_nodes.count(c) && visited.find(c) == visited.end() && c != source) {
                    bool dependent = false;
                    for (const auto& del_edge : Del) {
                        if ((del_edge.u == x && del_edge.v == c) || (del_edge.u == c && del_edge.v == x)) {
                            dependent = true;
                            break;
                        }
                    }
                    if (dependent) {
                        T.dist[c] = numeric_limits<double>::infinity();
                        T.parent[c] = -1;
                        T.affected[c] = true;
                        T.affected_del[c] = true;
                        Q.push(c);
                        visited.insert(c);
                    }
                }
            }
            level++;
        }
        T.affected_del[v] = false;
    }

    // Update distances for affected vertices
    bool global_change;
    do {
        bool local_change = false;
        vector<bool> to_process(T.affected);

        for (int v = 0; v < G.size(); ++v) {
            if (!local_nodes.count(v) || !to_process[v] || v == source) continue;
            T.affected[v] = false;

            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < G[v].size(); ++i) {
                int n = G[v][i].first;
                double w = G[v][i].second;
                bool updated = false;

                #pragma omp critical
                {
                    if (T.dist[n] < numeric_limits<double>::infinity() && T.dist[v] > T.dist[n] + w) {
                        T.dist[v] = T.dist[n] + w;
                        T.parent[v] = n;
                        T.affected[v] = true;
                        updated = true;
                    }
                    if (T.dist[v] < numeric_limits<double>::infinity() && T.dist[n] > T.dist[v] + w) {
                        if (local_nodes.count(n)) {
                            T.dist[n] = T.dist[v] + w;
                            T.parent[n] = v;
                            T.affected[n] = true;
                            updated = true;
                        } else if (ghost_nodes.count(n)) {
                            ghost_updates[n] = {T.dist[v] + w, v};
                            updated = true;
                        }
                    }
                }
                if (updated) {
                    local_change = true;
                }
            }
        }

        CommunicateGhostUpdates(T, local_nodes, ghost_nodes, ghost_to_owner, ghost_updates, rank, num_procs);

        int local_flag = local_change ? 1 : 0;
        MPI_Allreduce(&local_flag, &global_change, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    } while (global_change);
}

// Main function
#include <iomanip> // For std::setw, std::left, etc.


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double start_total = MPI_Wtime();
    double start, end;
    map<string, double> timings;

    string changes_file;
    if (argc >= 2) {
        changes_file = argv[1];
    }

    if (rank == 0) {
        cout << "\n----------------- PARALLEL SSSP (MPI + OpenMP) -----------------\n";
        cout << "Number of processes (subgraphs): " << num_procs << "\n";
    }

    vector<vector<pair<int, double>>> G;
    set<int> local_nodes, ghost_nodes;
    map<int, int> ghost_to_owner;
    int n, edge_count;

    start = MPI_Wtime();
    LoadSubgraph(rank, G, local_nodes, ghost_nodes, ghost_to_owner, n, edge_count);
    end = MPI_Wtime();
    timings["LoadSubgraph"] = (end - start) * 1000.0;

    int total_edges = 0;
    MPI_Reduce(&edge_count, &total_edges, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    int source = 0;
    int A = max(50, n / 100);

    SSSPTree T(n);
    start = MPI_Wtime();
    ComputeInitialSSSP(G, T, source, local_nodes, ghost_nodes, rank, num_procs, n);
    end = MPI_Wtime();
    timings["ComputeInitialSSSP"] = (end - start) * 1000.0;

    vector<Edge> Del, Ins;
    start = MPI_Wtime();
    if (!changes_file.empty()) {
        LoadChanges(changes_file, rank, Del, Ins, n);

        int del_size = Del.size();
        MPI_Bcast(&del_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) Del.reserve(del_size);
        for (int i = 0; i < del_size; ++i) {
            int data[2];
            if (rank == 0) {
                data[0] = Del[i].u;
                data[1] = Del[i].v;
            }
            MPI_Bcast(data, 2, MPI_INT, 0, MPI_COMM_WORLD);
            if (rank != 0) Del.emplace_back(data[0], data[1]);
        }

        int ins_size = Ins.size();
        MPI_Bcast(&ins_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) Ins.reserve(ins_size);
        for (int i = 0; i < ins_size; ++i) {
            int data[3];
            if (rank == 0) {
                data[0] = Ins[i].u;
                data[1] = Ins[i].v;
                data[2] = static_cast<int>(Ins[i].weight * 1000);
            }
            MPI_Bcast(data, 3, MPI_INT, 0, MPI_COMM_WORLD);
            if (rank != 0) Ins.emplace_back(data[0], data[1], data[2] / 1000.0);
        }
    }
    end = MPI_Wtime();
    timings["LoadAndBroadcastChanges"] = (end - start) * 1000.0;

    start = MPI_Wtime();
    AsynchronousUpdating(G, T, Del, Ins, source, A, local_nodes, ghost_nodes, ghost_to_owner, rank, num_procs);
    end = MPI_Wtime();
    timings["AsynchronousUpdating"] = (end - start) * 1000.0;

    start = MPI_Wtime();
    vector<double> local_ghost_dist(n, numeric_limits<double>::infinity());
    vector<int> local_ghost_parent(n, -1);
    for (int v : local_nodes) {
        local_ghost_dist[v] = T.dist[v];
        local_ghost_parent[v] = T.parent[v];
    }

    vector<double> global_ghost_dist(n);
    vector<int> global_ghost_parent(n);
    MPI_Allreduce(local_ghost_dist.data(), global_ghost_dist.data(), n, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(local_ghost_parent.data(), global_ghost_parent.data(), n, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    for (int v = 0; v < n; ++v) {
        T.dist[v] = global_ghost_dist[v];
        if (T.dist[v] < numeric_limits<double>::infinity()) {
            T.parent[v] = global_ghost_parent[v];
        } else {
            T.parent[v] = -1;
        }
    }
    T.parent[source] = -1;
    end = MPI_Wtime();
    timings["GhostReconciliation"] = (end - start) * 1000.0;

    start = MPI_Wtime();
    vector<pair<int, pair<double, int>>> local_output;
    for (int v : local_nodes) {
        local_output.emplace_back(v, make_pair(T.dist[v], T.parent[v]));
    }
    sort(local_output.begin(), local_output.end());

    stringstream ss;
    ss << fixed << setprecision(1);
    for (const auto& [v, data] : local_output) {
        if (data.first >= numeric_limits<double>::infinity()) {
            ss << v << " inf " << data.second << "\n";
        } else {
            ss << v << " " << data.first << " " << data.second << "\n";
        }
    }
    string local_output_str = ss.str();
    int local_size = local_output_str.size();

    vector<int> output_sizes(num_procs);
    MPI_Gather(&local_size, 1, MPI_INT, output_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> displacements(num_procs, 0);
    int total_size = 0;
    if (rank == 0) {
        total_size = output_sizes[0];
        for (int i = 1; i < num_procs; ++i) {
            displacements[i] = total_size;
            total_size += output_sizes[i];
        }
    }

    vector<char> all_output(total_size);
    MPI_Gatherv(local_output_str.data(), local_size, MPI_CHAR, all_output.data(), output_sizes.data(), displacements.data(), MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        string all_output_str(all_output.begin(), all_output.end());
        ofstream out("output_mpi_openmp.txt");
        if (!out.is_open()) {
            cerr << "Error: Could not open output file." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        out << "Vertex Distance Parent\n";
        vector<pair<int, pair<double, int>>> final_output;
        stringstream all_ss(all_output_str);
        string line;
        while (getline(all_ss, line)) {
            istringstream iss(line);
            int v, parent;
            string dist_str;
            if (!(iss >> v >> dist_str >> parent)) continue;
            double dist = (dist_str == "inf") ? numeric_limits<double>::infinity() : stod(dist_str);
            final_output.emplace_back(v, make_pair(dist, parent));
        }
        sort(final_output.begin(), final_output.end());
        for (const auto& [v, data] : final_output) {
            out << v << " " << (data.first >= numeric_limits<double>::infinity() ? "inf" : to_string(data.first)) << " " << data.second << "\n";
        }
        out.close();
    }
    end = MPI_Wtime();
    timings["OutputGathering"] = (end - start) * 1000.0;

    double end_total = MPI_Wtime();
    timings["TotalExecution"] = (end_total - start_total) * 1000.0;

    if (rank == 0) {
        cout << "\n----------------- PERFORMANCE SUMMARY (ms) -----------------\n";
        cout << left << setw(35) << "Component" << right << setw(12) << "Time (ms)" << "\n";
        cout << string(47, '-') << "\n";
        for (const auto& [label, time_ms] : timings) {
            cout << left << setw(35) << label << right << setw(12) << fixed << setprecision(2) << time_ms << "\n";
        }
        cout << "---------------------------------------------------------------\n";
        cout << "Output written to output_mpi_openmp.txt\n";
    }

    MPI_Finalize();
    return 0;
}

// mpicxx -o e_mpi_omp parallel_mpi_openmp.cpp -fopenmp
// mpirun -np 2 ./e_mpi_omp
// time mpirun -np 2 ./e_mpi_omp ../Datasets/bio-CE/bio-CE-HT_updates_500.edges