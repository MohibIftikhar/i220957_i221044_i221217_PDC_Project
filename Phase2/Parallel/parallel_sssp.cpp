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

using namespace std;

// Structure to represent an edge
struct Edge
{
    int u, v;
    double weight;
    Edge(int _u, int _v, double _w) : u(_u), v(_v), weight(_w) {}
};

// Structure to represent the SSSP tree
struct SSSPTree
{
    vector<int> parent;
    vector<double> dist;
    vector<bool> affected;
    vector<bool> affected_del;
    SSSPTree(int n) : parent(n, -1),
                      dist(n, numeric_limits<double>::infinity()),
                      affected(n, false),
                      affected_del(n, false) {}
};

// Compute initial SSSP tree using Dijkstra's algorithm
void ComputeInitialSSSP(const vector<vector<pair<int, double>>> &G, SSSPTree &T, int source)
{
    T.dist[source] = 0;
    T.parent[source] = source;
    priority_queue<pair<double, int>, vector<pair<double, int>>, greater<>> pq;
    pq.push({0, source});

    while (!pq.empty())
    {
        double d = pq.top().first;
        int u = pq.top().second;
        pq.pop();

        if (d > T.dist[u])
            continue;

        for (const auto &edge : G[u])
        {
            int v = edge.first;
            double w = edge.second;
            if (T.dist[v] > T.dist[u] + w)
            {
                T.dist[v] = T.dist[u] + w;
                T.parent[v] = u;
                pq.push({T.dist[v], v});
            }
        }
    }
}

// Function to remove an edge from the graph
void RemoveEdge(vector<vector<pair<int, double>>>& G, int u, int v, double weight) {
    G[u].erase(
        remove_if(G[u].begin(), G[u].end(),
            [v, weight](const pair<int, double>& e) { 
                return e.first == v && e.second == weight; 
            }),
        G[u].end()
    );
    G[v].erase(
        remove_if(G[v].begin(), G[v].end(),
            [u, weight](const pair<int, double>& e) { 
                return e.first == u && e.second == weight; 
            }),
        G[v].end()
    );
}

// Function to process edge changes (Step 1: Identify affected vertices)
void ProcessCE(vector<vector<pair<int, double>>> &G, SSSPTree &T, const vector<Edge> &Del, const vector<Edge> &Ins, int source)
{
    // Process deletions
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < Del.size(); ++i)
    {
        int u = Del[i].u, v = Del[i].v;
#pragma omp critical
        {
            RemoveEdge(G, u, v, Del[i].weight);
        }
        if (T.parent[v] == u)
        {
            T.dist[v] = numeric_limits<double>::infinity();
            T.parent[v] = -1;
            T.affected_del[v] = true;
            T.affected[v] = true;
        }
        else if (T.parent[u] == v)
        {
            T.dist[u] = numeric_limits<double>::infinity();
            T.parent[u] = -1;
            T.affected_del[u] = true;
            T.affected[u] = true;
        }
    }

    // Process insertions
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < Ins.size(); ++i)
    {
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
            if (new_dist < T.dist[y])
            {
                T.dist[y] = new_dist;
                T.parent[y] = x;
                T.affected[y] = true;
            }
        }
    }
}

// Function to perform asynchronous SSSP update (Algorithm 4)
void AsynchronousUpdating(vector<vector<pair<int, double>>> &G, SSSPTree &T, const vector<Edge> &Del, const vector<Edge> &Ins, int source, int A)
{
    ProcessCE(G, T, Del, Ins, source);

    for (int v = 0; v < G.size(); ++v)
    {
        if (T.affected_del[v])
        {
            queue<int> Q;
            Q.push(v);
            int level = 0;
            while (!Q.empty() && level <= A)
            {
                int x = Q.front();
                Q.pop();
                for (int c = 0; c < G.size(); ++c)
                {
                    if (T.parent[c] == x)
                    {
                        T.dist[c] = numeric_limits<double>::infinity();
                        T.parent[c] = -1;
                        T.affected[c] = true;
                        T.affected_del[c] = true;
                        if (level < A)
                        {
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
    while (change)
    {
        change = false;
        vector<bool> to_process(T.affected);
        for (int v = 0; v < G.size(); ++v)
        {
            if (to_process[v])
            {
                T.affected[v] = false;
#pragma omp parallel for schedule(dynamic)
                for (size_t i = 0; i < G[v].size(); ++i)
                {
                    int n = G[v][i].first;
                    double w = G[v][i].second;
                    bool updated = false;
#pragma omp critical
                    {
                        if (T.dist[v] > T.dist[n] + w)
                        {
                            T.dist[v] = T.dist[n] + w;
                            T.parent[v] = n;
                            T.affected[v] = true;
                            updated = true;
                        }
                        if (T.dist[n] > T.dist[v] + w)
                        {
                            T.dist[n] = T.dist[v] + w;
                            T.parent[n] = v;
                            T.affected[n] = true;
                            updated = true;
                        }
                    }
                    if (updated)
                    {
                        change = true;
                    }
                }
            }
        }
    }
}

// Main function to read graph and run the algorithm
int main()
{
    // Open graph file
    ifstream graph_file("test_data.txt");
    if (!graph_file.is_open())
    {
        cerr << "Error: Could not open bio-CE-HT.edges\n";
        return 1;
    }

    // Collect unique vertices and edges
    set<int> vertices;
    vector<Edge> edges;
    string line;
    int line_number = 0;
    bool has_header = false; // Set to true if file has a header line

    while (getline(graph_file, line))
    {
        line_number++;
        if (line.empty() || line[0] == '#' || line[0] == '%')
        {
            cout << "Skipping line " << line_number << ": " << line << "\n";
            continue;
        }
        if (line_number == 1 && has_header)
        {
            cout << "Skipping header line: " << line << "\n";
            continue;
        }

        stringstream ss(line);
        int u, v;
        double w;
        if (!(ss >> u >> v >> w))
        {
            cerr << "Error: Invalid format at line " << line_number << ": " << line << "\n";
            continue;
        }
        if (u < 0 || v < 0)
        {
            cerr << "Error: Negative vertex ID at line " << line_number << ": " << line << "\n";
            continue;
        }
        if (w < 0)
        {
            cerr << "Error: Negative weight at line " << line_number << ": " << line << "\n";
            continue;
        }

        vertices.insert(u);
        vertices.insert(v);
        edges.emplace_back(u, v, w);
    }
    graph_file.close();

    if (vertices.empty())
    {
        cerr << "Error: No vertices found in bio-CE-HT.edges\n";
        return 1;
    }
    if (edges.empty())
    {
        cerr << "Error: No edges found in bio-CE-HT.edges\n";
        return 1;
    }

    // Map vertex IDs to consecutive 0-based indices
    map<int, int> vertex_map; // Original ID -> New ID (0 to vertices.size()-1)
    int new_id = 0;
    for (int v : vertices)
    {
        vertex_map[v] = new_id++;
    }
    int n = vertices.size(); // Number of unique vertices
    cout << "Number of vertices: " << n << "\n";
    cout << "Number of edges: " << edges.size() << "\n";

    // Initialize adjacency list with remapped IDs
    vector<vector<pair<int, double>>> G(n);
    for (const auto& edge : edges)
    {
        int u = vertex_map[edge.u];
        int v = vertex_map[edge.v];
        double w = edge.weight;
        G[u].push_back({v, w});
        G[v].push_back({u, w}); // Undirected edge
    }

    // Algorithm parameters
    int source = vertex_map[0]; // Remap source vertex (original ID 0)
    int A = 50;
    if (n > 10000)
    {
        A = n / 100;
    }

    SSSPTree T(n);
    ComputeInitialSSSP(G, T, source);

    // Print results with original vertex IDs
    ofstream out("output.txt");
    if (!out.is_open())
    {
        cerr << "Error: Could not open output.txt for writing\n";
        return 1;
    }

    // Create reverse mapping for output (new ID -> original ID)
    vector<int> reverse_map(n);
    for (const auto& [orig_id, new_id] : vertex_map)
    {
        reverse_map[new_id] = orig_id;
    }

    out << fixed << setprecision(1);
    out << "Vertex Distance Parent\n";
    for (int i = 0; i < n; ++i)
    {
        int parent_orig = T.parent[i] == -1 ? -1 : reverse_map[T.parent[i]];
        out << reverse_map[i] << " " << T.dist[i] << " " << parent_orig << "\n";
    }
    out.close();
    cout << "Output written to output.txt\n";

    return 0;
}