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

using namespace std;

struct Edge {
    int u, v;
    double weight;
    Edge(int _u, int _v, double _w) : u(_u), v(_v), weight(_w) {}
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

void RemoveEdge(vector<vector<pair<int, double>>> &G, int u, int v, double weight) {
    G[u].erase(remove_if(G[u].begin(), G[u].end(),
                         [v, weight](const pair<int, double> &e) {
                             return e.first == v && e.second == weight;
                         }),
               G[u].end());
    G[v].erase(remove_if(G[v].begin(), G[v].end(),
                         [u, weight](const pair<int, double> &e) {
                             return e.first == u && e.second == weight;
                         }),
               G[v].end());
}

void ProcessCE(vector<vector<pair<int, double>>> &G, SSSPTree &T, const vector<Edge> &Del, const vector<Edge> &Ins, int source) {
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < Del.size(); ++i) {
        int u = Del[i].u, v = Del[i].v;
#pragma omp critical
        {
            RemoveEdge(G, u, v, Del[i].weight);
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

int main() {
    double start_time, end_time;

    // Read graph file
    start_time = omp_get_wtime();
    ifstream graph_file("graph.txt");
    if (!graph_file.is_open()) {
        cerr << "Error: Could not open graph.txt\n";
        return 1;
    }

    vector<vector<pair<int, double>>> G;
    set<int> all_nodes;
    int u, v;
    double w;
    
    while (graph_file >> u >> v >> w) {
        if (u < 0 || v < 0 || w < 0) {
            cerr << "Error: Invalid edge values in graph.txt: " << u << " " << v << " " << w << "\n";
            return 1;
        }
        all_nodes.insert(u);
        all_nodes.insert(v);
        G.resize(all_nodes.size());
        G[u].push_back({v, w});
        G[v].push_back({u, w});
    }
    graph_file.close();
    end_time = omp_get_wtime();
    cout << "Time to read graph file: " << end_time - start_time << " seconds\n";

    // Read changes file
    start_time = omp_get_wtime();
    ifstream changes_file("changes.txt");
    if (!changes_file.is_open()) {
        cerr << "Error: Could not open changes.txt\n";
        return 1;
    }

    vector<Edge> Del, Ins;
    string line;
    while (getline(changes_file, line)) {
        if (line.empty()) continue;
        istringstream iss(line);
        char type;
        int u, v;
        double w;
        if (!(iss >> type >> u >> v >> w)) {
            cerr << "Error: Invalid line in changes.txt: " << line << "\n";
            continue;
        }

        if (u < 0 || v < 0 || w < 0) {
            cerr << "Error: Invalid edge values in changes.txt: " << line << "\n";
            continue;
        }

        if (type == 'D' || type == 'd') {
            Del.emplace_back(u, v, w);
        } else if (type == 'I' || type == 'i') {
            Ins.emplace_back(u, v, w);
        } else {
            cerr << "Error: Unknown change type in line: " << line << "\n";
        }
    }
    changes_file.close();
    end_time = omp_get_wtime();
    cout << "Time to read changes file: " << end_time - start_time << " seconds\n";

    // Compute initial SSSP
    start_time = omp_get_wtime();
    int source = 0;
    int A = (G.size() > 10000) ? G.size() / 100 : 50;

    SSSPTree T(G.size());
    ComputeInitialSSSP(G, T, source);
    end_time = omp_get_wtime();
    cout << "Time to compute initial SSSP: " << end_time - start_time << " seconds\n";

    // Perform asynchronous updates
    start_time = omp_get_wtime();
    AsynchronousUpdating(G, T, Del, Ins, source, A);
    end_time = omp_get_wtime();
    cout << "Time for asynchronous updates: " << end_time - start_time << " seconds\n";

    // Write output to file
    start_time = omp_get_wtime();
    ofstream out("output_openmp.txt");
    if (!out.is_open()) {
        cerr << "Error: Could not open output.txt\n";
        return 1;
    }

    out << fixed << setprecision(1);
    out << "Vertex Distance Parent\n";
    for (int i = 0; i < G.size(); ++i) {
        out << i << " " << T.dist[i] << " " << T.parent[i] << "\n";
    }
    out.close();
    end_time = omp_get_wtime();
    cout << "Time to write output file: " << end_time - start_time << " seconds\n" << endl;

    return 0;
}

//  g++ -o e_omp parallel_openmp.cpp -fopenmp
// time ./e_omp
