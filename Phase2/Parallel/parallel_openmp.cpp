#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
#include <limits>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <set>
#include <tuple>

using namespace std;

struct Edge {
    int u, v;
    double weight;
    Edge(int _u, int _v, double _w = -1.0) : u(_u), v(_v), weight(_w) {}
};

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

void ComputeInitialSSSP(const vector<vector<pair<int, double>>> &G, SSSPTree &T, int source) {
    T.dist[source] = 0;
    T.parent[source] = -1;
    priority_queue<pair<double, int>, vector<pair<double, int>>, greater<>> pq;
    pq.push({0, source});

    while (!pq.empty()) {
        double d = pq.top().first;
        int u = pq.top().second;
        pq.pop();

        if (d > T.dist[u]) continue;

        for (const auto &edge : G[u]) {
            int v = edge.first;
            double w = edge.second;
            if (T.dist[v] > T.dist[u] + w) {
                T.dist[v] = T.dist[u] + w;
                T.parent[v] = u;
                pq.push({T.dist[v], v});
            }
        }
    }
}

void RemoveEdge(vector<vector<pair<int, double>>> &G, int u, int v) {
    auto remove_from_list = [](vector<pair<int, double>> &adj, int node) {
        adj.erase(remove_if(adj.begin(), adj.end(),
                            [node](const pair<int, double>& e) {
                                return e.first == node;
                            }),
                  adj.end());
    };

    remove_from_list(G[u], v);  // Remove v from u's adjacency
    remove_from_list(G[v], u);  // Remove u from v's adjacency
}

void ProcessCE(vector<vector<pair<int, double>>> &G, SSSPTree &T, const vector<Edge> &Del, const vector<Edge> &Ins, int source) {
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < Del.size(); ++i) {
        int u = Del[i].u, v = Del[i].v;
#pragma omp critical
        {
            RemoveEdge(G, u, v);
        }
        if (T.parent[v] == u) {
            T.dist[v] = numeric_limits<double>::infinity();
            T.parent[v] = -1;
            T.affected_del[v] = true;
            T.affected[v] = true;
        } else if (T.parent[u] == v) {
            T.dist[u] = numeric_limits<double>::infinity();
            T.parent[u] = -1;
            T.affected_del[u] = true;
            T.affected[u] = true;
        }
    }

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < Ins.size(); ++i) {
        int u = Ins[i].u, v = Ins[i].v;
        double w = Ins[i].weight;
#pragma omp critical
        {
            G[u].push_back({v, w});
            G[v].push_back({u, w});
        }

        int x = (T.dist[u] < T.dist[v]) ? u : v;
        int y = (x == u) ? v : u;
        double new_dist = T.dist[x] + w;
#pragma omp critical
        {
            if (new_dist < T.dist[y]) {
                T.dist[y] = new_dist;
                T.parent[y] = x;
                T.affected[y] = true;
            }
        }
    }
}

void AsynchronousUpdating(vector<vector<pair<int, double>>> &G, SSSPTree &T, const vector<Edge> &Del, const vector<Edge> &Ins, int source, int A) {
    ProcessCE(G, T, Del, Ins, source);

    for (int v = 0; v < G.size(); ++v) {
        if (T.affected_del[v]) {
            queue<int> Q;
            Q.push(v);
            int level = 0;
            while (!Q.empty() && level <= A) {
                int x = Q.front();
                Q.pop();
                for (int c = 0; c < G.size(); ++c) {
                    if (T.parent[c] == x) {
                        T.dist[c] = numeric_limits<double>::infinity();
                        T.parent[c] = -1;
                        T.affected[c] = true;
                        T.affected_del[c] = true;
                        if (level < A) {
                            Q.push(c);
                        }
                    }
                }
                level++;
            }
            T.affected_del[v] = false;
        }
    }

    bool change = true;
    while (change) {
        change = false;
        vector<bool> to_process(T.affected);
        for (int v = 0; v < G.size(); ++v) {
            if (to_process[v]) {
                T.affected[v] = false;
#pragma omp parallel for schedule(dynamic)
                for (size_t i = 0; i < G[v].size(); ++i) {
                    int n = G[v][i].first;
                    double w = G[v][i].second;
                    bool updated = false;
#pragma omp critical
                    {
                        if (T.dist[v] > T.dist[n] + w) {
                            T.dist[v] = T.dist[n] + w;
                            T.parent[v] = n;
                            T.affected[v] = true;
                            updated = true;
                        }
                        if (T.dist[n] > T.dist[v] + w) {
                            T.dist[n] = T.dist[v] + w;
                            T.parent[n] = v;
                            T.affected[n] = true;
                            updated = true;
                        }
                    }
                    if (updated) {
                        change = true;
                    }
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    cout << "\n----------------- PARALLEL SSSP : OpenMP Version -----------------\n\n";

    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <graph_file> [changes_file]\n";
        return 1;
    }

    string graph_filename = argv[1];
    string changes_filename = (argc >= 3) ? argv[2] : "";

    double t_total_start = omp_get_wtime();

    // Read graph file
    double t_read_start = omp_get_wtime();
    ifstream graph_file(graph_filename);
    if (!graph_file.is_open()) {
        cerr << "Error: Could not open " << graph_filename << "\n";
        return 1;
    }

    vector<tuple<int, int, double>> edges;
    int u, v;
    double w;
    int max_node = -1;

    while (graph_file >> u >> v >> w) {
        if (u < 0 || v < 0 || w < 0) {
            cerr << "Error: Invalid edge values in " << graph_filename << ": " << u << " " << v << " " << w << "\n";
            return 1;
        }
        max_node = max({max_node, u, v});
        edges.emplace_back(u, v, w);
    }
    graph_file.close();

    vector<vector<pair<int, double>>> G(max_node + 1);
    for (const auto &[u, v, w] : edges) {
        G[u].push_back({v, w});
        G[v].push_back({u, w});
    }

    double t_read_end = omp_get_wtime();

    // Read changes file (optional)
    vector<Edge> Del, Ins;
    double t_changes_start = 0.0, t_changes_end = 0.0;
    if (!changes_filename.empty()) {
        t_changes_start = omp_get_wtime();
        ifstream changes_file(changes_filename);
        if (!changes_file.is_open()) {
            cerr << "Error: Could not open " << changes_filename << "\n";
            return 1;
        }

        string line;
        while (getline(changes_file, line)) {
            if (line.empty()) continue;
            istringstream iss(line);
            char type;
            int u, v;
            double w;

            if (!(iss >> type >> u >> v)) {
                cerr << "Error: Invalid line in " << changes_filename << ": " << line << "\n";
                continue;
            }

            if (type == 'D' || type == 'd') {
                Del.emplace_back(u, v);
            } else if ((type == 'I' || type == 'i') && (iss >> w)) {
                Ins.emplace_back(u, v, w);
            } else {
                cerr << "Error: Invalid format or unknown change type: " << line << "\n";
            }
        }
        changes_file.close();
        t_changes_end = omp_get_wtime();
    }

    // Initial SSSP computation
    double t_sssp_start = omp_get_wtime();
    int source = 0;
    int A = (G.size() > 10000) ? G.size() / 100 : 50;
    SSSPTree T(G.size());
    ComputeInitialSSSP(G, T, source);
    double t_sssp_end = omp_get_wtime();

    // Asynchronous updates
    double t_async_start = 0.0, t_async_end = 0.0;
    if (!changes_filename.empty()) {
        t_async_start = omp_get_wtime();
        AsynchronousUpdating(G, T, Del, Ins, source, A);
        t_async_end = omp_get_wtime();
    }

    // Write output
    double t_write_start = omp_get_wtime();
    ofstream out("output_openmp.txt");
    if (!out.is_open()) {
        cerr << "Error: Could not open output_openmp.txt\n";
        return 1;
    }

    out << fixed << setprecision(1);
    out << "Vertex Distance Parent\n";
    for (int i = 0; i < G.size(); ++i) {
        out << i << " " << T.dist[i] << " " << T.parent[i] << "\n";
    }
    out.close();
    double t_write_end = omp_get_wtime();

    double t_total_end = omp_get_wtime();

    // Convert to milliseconds
    auto ms = [](double start, double end) {
        return static_cast<int>((end - start) * 1000.0);
    };

    // Output performance summary
    cout << "\n=== Performance Summary ===\n";
    cout << "Total Vertices       : " << G.size() << "\n";
    cout << "Source Vertex        : " << source << "\n";
    cout << "\n--- Timing (ms) ---\n";
    cout << "Graph Read           : " << ms(t_read_start, t_read_end) << " ms\n";
    if (!changes_filename.empty()) {
        cout << "Read Changes         : " << ms(t_changes_start, t_changes_end) << " ms\n";
    } else {
        cout << "Read Changes         : skipped\n";
    }
    cout << "Initial SSSP         : " << ms(t_sssp_start, t_sssp_end) << " ms\n";
    if (!changes_filename.empty()) {
        cout << "Asynchronous Update  : " << ms(t_async_start, t_async_end) << " ms\n";
    } else {
        cout << "Asynchronous Update  : skipped\n";
    }
    cout << "Write Output         : " << ms(t_write_start, t_write_end) << " ms\n";
    cout << "Total Time           : " << ms(t_total_start, t_total_end) << " ms\n";
    cout << "============================\n";
    cout << "Output written to output_openmp.txt\n\n";

    return 0;
}

// g++ -o e_omp parallel_openmp.cpp -fopenmp
// time ./e_omp ../Datasets/bio-CE/bio-CE-HT.edges 
// time ./e_omp ../Datasets/bio-CE/bio-CE-HT.edges ../Datasets/bio-CE/bio-CE-HT_updates_500.edges

// time OMP_NUM_THREADS=2 ./e_omp ../Datasets/bio-h/bio-h.edges
// time OMP_NUM_THREADS=4 ./e_omp ../Datasets/bio-h/bio-h.edges
// time OMP_NUM_THREADS=6 ./e_omp ../Datasets/bio-h/bio-h.edges
// time OMP_NUM_THREADS=8 ./e_omp ../Datasets/bio-h/bio-h.edges

// time OMP_NUM_THREADS=2 ./e_omp ../Datasets/bio-h/bio-h.edges ../Datasets/bio-h/bio-h_updates_10000.edges 
// time OMP_NUM_THREADS=4 ./e_omp ../Datasets/bio-h/bio-h.edges ../Datasets/bio-h/bio-h_updates_10000.edges 
// time OMP_NUM_THREADS=6 ./e_omp ../Datasets/bio-h/bio-h.edges ../Datasets/bio-h/bio-h_updates_10000.edges 
// time OMP_NUM_THREADS=8 ./e_omp ../Datasets/bio-h/bio-h.edges ../Datasets/bio-h/bio-h_updates_10000.edges 

// time OMP_NUM_THREADS=2 ./e_omp ../Datasets/bio-h/bio-h.edges ../Datasets/bio-h/bio-h_updates_12500.edges 
// time OMP_NUM_THREADS=4 ./e_omp ../Datasets/bio-h/bio-h.edges ../Datasets/bio-h/bio-h_updates_12500.edges 
// time OMP_NUM_THREADS=6 ./e_omp ../Datasets/bio-h/bio-h.edges ../Datasets/bio-h/bio-h_updates_12500.edges 
// time OMP_NUM_THREADS=8 ./e_omp ../Datasets/bio-h/bio-h.edges ../Datasets/bio-h/bio-h_updates_12500.edges 

// time OMP_NUM_THREADS=2 ./e_omp ../Datasets/bio-h/bio-h.edges ../Datasets/bio-h/bio-h_updates_15000.edges 
// time OMP_NUM_THREADS=4 ./e_omp ../Datasets/bio-h/bio-h.edges ../Datasets/bio-h/bio-h_updates_15000.edges 
// time OMP_NUM_THREADS=6 ./e_omp ../Datasets/bio-h/bio-h.edges ../Datasets/bio-h/bio-h_updates_15000.edges 
// time OMP_NUM_THREADS=8 ./e_omp ../Datasets/bio-h/bio-h.edges ../Datasets/bio-h/bio-h_updates_15000.edges 

// time ./e_omp ../Datasets/bio-CX/bio-CE-CX.edges
// time ./e_omp ../Datasets/bio-CX/bio-CE-CX.edges ../Datasets/bio-CX/bio-CE-CX_updates_10000.edges