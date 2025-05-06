#include <iostream>
#include <vector>
#include <list>
#include <queue>
#include <climits>
#include <fstream>
#include <sstream>
#include <functional>

using namespace std;

// Structure for an edge
struct Edge {
    int to;
    long long weight;
    Edge(int t, long long w) : to(t), weight(w) {}
};

// Graph class using adjacency list representation
class Graph {
public:
    int V;
    vector<list<Edge>> vertices;
    vector<bool> active;

    Graph(int vertices) : V(vertices) {
        this->vertices.resize(V);
        this->active.assign(V, true);
    }

    void addEdge(int from, int to, long long weight) {
        if (from >= V || to >= V || !active[from] || !active[to]) return;
        vertices[from].emplace_back(to, weight);
        vertices[to].emplace_back(from, weight);
    }

    bool deleteEdge(int from, int to) {
        if (from >= V || to >= V || !active[from] || !active[to]) return false;
        bool deleted = false;

        for (auto it = vertices[from].begin(); it != vertices[from].end(); ++it) {
            if (it->to == to) {
                vertices[from].erase(it);
                deleted = true;
                break;
            }
        }

        for (auto it = vertices[to].begin(); it != vertices[to].end(); ++it) {
            if (it->to == from) {
                vertices[to].erase(it);
                deleted = true;
                break;
            }
        }

        return deleted;
    }

    void deleteNode(int node) {
        if (node >= V || !active[node]) return;
        active[node] = false;
        vertices[node].clear();

        for (int u = 0; u < V; u++) {
            if (!active[u]) continue;
            for (auto it = vertices[u].begin(); it != vertices[u].end();) {
                if (it->to == node) {
                    it = vertices[u].erase(it);
                } else {
                    ++it;
                }
            }
        }
    }
};

// Function to read the initial graph from a file
Graph* readGraphFromFile(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return nullptr;
    }

    int maxVertex = -1;
    string line;
    vector<tuple<int, int, double>> edges;

    while (getline(file, line)) {
        stringstream ss(line);
        int from, to;
        double distance;
        ss >> from >> to >> distance;
        maxVertex = max(maxVertex, max(from, to));
        edges.emplace_back(from, to, distance);
    }

    file.close();

    int V = maxVertex + 1;
    Graph* graph = new Graph(V);

    for (const auto& edge : edges) {
        int from = get<0>(edge);
        int to = get<1>(edge);
        long long weight = static_cast<long long>(get<2>(edge));
        graph->addEdge(from, to, weight);
    }

    return graph;
}

// Dijkstra's algorithm
void dijkstra(Graph& graph, int source, vector<long long>& Dist, vector<int>& Parent) {
    using PQNode = pair<long long, int>;
    priority_queue<PQNode, vector<PQNode>, greater<PQNode>> pq;

    Dist.assign(graph.V, LLONG_MAX);
    Parent.assign(graph.V, -1);
    if (!graph.active[source]) return;

    Dist[source] = 0;
    pq.emplace(0, source);

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();

        if (d != Dist[u] || !graph.active[u]) continue;

        for (const Edge& edge : graph.vertices[u]) {
            int v = edge.to;
            long long weight = edge.weight;
            if (!graph.active[v]) continue;

            if (Dist[v] > Dist[u] + weight) {
                Dist[v] = Dist[u] + weight;
                Parent[v] = u;
                pq.emplace(Dist[v], v);
            }
        }
    }
}

// Recompute SSSP from scratch on every update
void UpdateSingleChange(Graph& graph, int source, vector<long long>& Dist, vector<int>& Parent,
                        int u, int v, long long W_uv, bool isDeletion = false) {
    dijkstra(graph, source, Dist, Parent);
}

// Apply changes from a file
void readChangesFromFile(const string& filename, Graph* graph, vector<long long>& Dist, vector<int>& Parent, int source) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open changes file " << filename << endl;
        return;
    }

    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;

        stringstream ss(line);
        char type;
        int u, v;
        double w;
        ss >> type >> u >> v >> w;
        long long weight = static_cast<long long>(w);

        if (type == 'D' || type == 'd') {
            cout << "\nDeleting edge e(" << u << "," << v << ") with weight " << w << ":\n";
            if (graph->deleteEdge(u, v)) {
                UpdateSingleChange(*graph, source, Dist, Parent, u, v, LLONG_MAX, true);
            } else {
                cout << "Edge not found.\n";
            }
        } else if (type == 'I' || type == 'i') {
            cout << "\nAdding edge e(" << u << "," << v << ") with weight " << w << ":\n";
            graph->addEdge(u, v, weight);
            UpdateSingleChange(*graph, source, Dist, Parent, u, v, weight, false);
        } else {
            cerr << "Unknown change type: " << type << "\n";
            continue;
        }

        // Print updated results
        for (int i = 0; i < graph->V; i++) {
            cout << "Vertex " << i << ": Dist = " << (Dist[i] == LLONG_MAX ? "INF" : to_string(Dist[i]))
                 << ", Parent = " << Parent[i] << "\n";
        }
    }

    file.close();
}

// Main
int main() {
    Graph* graph = readGraphFromFile("graph.txt");
    if (!graph) return 1;

    vector<long long> Dist;
    vector<int> Parent;

    dijkstra(*graph, 0, Dist, Parent);

    cout << "Initial SSSP Distances:\n";
    for (int i = 0; i < graph->V; i++) {
        cout << "Vertex " << i << ": Dist = " << (Dist[i] == LLONG_MAX ? "INF" : to_string(Dist[i])) 
             << ", Parent = " << Parent[i] << "\n";
    }

    readChangesFromFile("changes.txt", graph, Dist, Parent, 0);

    delete graph;
    return 0;
}
