#include <iostream>
#include <vector>
#include <list>
#include <queue>
#include <climits>
#include <functional>
#include <fstream>
#include <sstream>
#include <cmath>

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
    int V; // Number of vertices
    vector<list<Edge>> vertices; // Adjacency list
    vector<bool> active; // Track active vertices (for node deletion)

    Graph(int vertices) : V(vertices) {
        this->vertices.resize(V);
        this->active.assign(V, true); // All vertices are initially active
    }

    void addEdge(int from, int to, long long weight) {
        if (from >= V || to >= V || !active[from] || !active[to]) return; // Ignore if vertices are invalid or inactive
        vertices[from].emplace_back(to, weight);
        vertices[to].emplace_back(from, weight); // Add reverse edge for undirected graph
    }

    // Delete an edge from the graph (delete both directions for undirected behavior)
    bool deleteEdge(int from, int to) {
        if (from >= V || to >= V || !active[from] || !active[to]) return false;
        bool deleted = false;
        // Delete (from, to)
        for (auto it = vertices[from].begin(); it != vertices[from].end(); ++it) {
            if (it->to == to) {
                vertices[from].erase(it);
                deleted = true;
                break;
            }
        }
        // Delete (to, from)
        for (auto it = vertices[to].begin(); it != vertices[to].end(); ++it) {
            if (it->to == from) {
                vertices[to].erase(it);
                deleted = true;
                break;
            }
        }
        return deleted;
    }

    // Delete a node and all its associated edges
    void deleteNode(int node) {
        if (node >= V || !active[node]) return;

        // Mark the node as inactive
        active[node] = false;

        // Clear all outgoing edges from this node
        vertices[node].clear();

        // Remove all incoming edges to this node
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

    // Update the weight of an existing edge (update both directions)
    bool updateEdgeWeight(int from, int to, long long newWeight) {
        if (from >= V || to >= V || !active[from] || !active[to]) return false;
        bool updated = false;
        for (auto& edge : vertices[from]) {
            if (edge.to == to) {
                edge.weight = newWeight;
                updated = true;
                break;
            }
        }
        for (auto& edge : vertices[to]) {
            if (edge.to == from) {
                edge.weight = newWeight;
                updated = true;
                break;
            }
        }
        return updated;
    }
};

// Function to read the initial graph from a file (no scaling, undirected)
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
        double distance = get<2>(edge);
        long long weight = static_cast<long long>(distance); // No scaling
        graph->addEdge(from, to, weight); // Adds both directions
    }

    return graph;
}

// Compute initial SSSP using Dijkstra's algorithm
void dijkstra(Graph& graph, int source, vector<long long>& Dist, vector<int>& Parent) {
    using PQNode = pair<long long, int>;
    priority_queue<PQNode, vector<PQNode>, greater<PQNode>> pq;

    Dist.assign(graph.V, LLONG_MAX);
    Parent.assign(graph.V, -1);
    if (!graph.active[source]) return; // Source is inactive
    Dist[source] = 0;
    pq.emplace(0, source);

    while (!pq.empty()) {
        long long d = pq.top().first;
        int u = pq.top().second;
        pq.pop();

        if (d != Dist[u]) continue;
        if (!graph.active[u]) continue; // Skip inactive vertices

        for (const Edge& edge : graph.vertices[u]) {
            int v = edge.to;
            if (!graph.active[v]) continue; // Skip inactive vertices
            long long weight = edge.weight;

            if (Dist[u] != LLONG_MAX && Dist[v] > Dist[u] + weight) {
                Dist[v] = Dist[u] + weight;
                Parent[v] = u;
                pq.emplace(Dist[v], v);
            }
        }
    }
}

// Function to find descendants of a vertex in the SSSP tree
void findDescendants(const Graph& graph, const vector<int>& Parent, int v, vector<bool>& affected) {
    for (int u = 0; u < graph.V; u++) {
        if (!graph.active[u]) continue;
        if (Parent[u] == v && !affected[u]) {
            affected[u] = true;
            findDescendants(graph, Parent, u, affected);
        }
    }
}

// Simplified UpdateSingleChange for edge deletion (handle both directions)
void UpdateSingleChange(Graph& graph, int source, vector<long long>& Dist, vector<int>& Parent, 
                        int u, int v, long long W_uv, bool isDeletion = false) {
    if (!graph.active[u] || !graph.active[v]) return;

    if (isDeletion) {
        // Check if either (u,v) or (v,u) is in the SSSP tree
        vector<bool> affected(graph.V, false);
        int affectedVertex = -1;

        // Check (u,v)
        if (Parent[v] == u) {
            affectedVertex = v;
            affected[v] = true;
            Dist[v] = LLONG_MAX;
            Parent[v] = -1;
        }
        // Check (v,u)
        else if (Parent[u] == v) {
            affectedVertex = u;
            affected[u] = true;
            Dist[u] = LLONG_MAX;
            Parent[u] = -1;
        }

        if (affectedVertex != -1) {
            // Mark all descendants of the affected vertex as affected
            findDescendants(graph, Parent, affectedVertex, affected);
            for (int i = 0; i < graph.V; i++) {
                if (affected[i]) {
                    Dist[i] = LLONG_MAX;
                    Parent[i] = -1;
                }
            }
        }

        // Recompute shortest paths for all vertices using Dijkstra's
        using PQNode = pair<long long, int>;
        priority_queue<PQNode, vector<PQNode>, greater<PQNode>> pq;
        vector<bool> visited(graph.V, false);

        // Initialize the priority queue with the source
        pq.emplace(0, source);
        Dist[source] = 0;

        while (!pq.empty()) {
            long long d = pq.top().first;
            int w = pq.top().second;
            pq.pop();

            if (visited[w]) continue;
            visited[w] = true;
            if (!graph.active[w]) continue;

            for (const Edge& edge : graph.vertices[w]) {
                int n = edge.to;
                if (!graph.active[n]) continue;
                if (Dist[w] != LLONG_MAX && Dist[n] > Dist[w] + edge.weight) {
                    Dist[n] = Dist[w] + edge.weight;
                    Parent[n] = w;
                    pq.emplace(Dist[n], n);
                }
            }
        }
    } else {
        // For edge insertion or weight update, use a simplified approach
        using PQNode = pair<long long, int>;
        priority_queue<PQNode, vector<PQNode>, greater<PQNode>> pq;

        // Temporarily try the new edge (consider both directions)
        long long newDistV = (Dist[u] != LLONG_MAX) ? Dist[u] + W_uv : LLONG_MAX;
        if (newDistV < Dist[v]) {
            Dist[v] = newDistV;
            Parent[v] = u;
        }
        long long newDistU = (Dist[v] != LLONG_MAX) ? Dist[v] + W_uv : LLONG_MAX;
        if (newDistU < Dist[u]) {
            Dist[u] = newDistU;
            Parent[u] = v;
        }

        // Recompute shortest paths starting from all vertices
        vector<bool> visited(graph.V, false);
        Dist[source] = 0;
        pq.emplace(0, source);

        while (!pq.empty()) {
            long long d = pq.top().first;
            int w = pq.top().second;
            pq.pop();

            if (visited[w]) continue;
            visited[w] = true;
            if (!graph.active[w]) continue;

            for (const Edge& edge : graph.vertices[w]) {
                int n = edge.to;
                if (!graph.active[n]) continue;
                if (Dist[w] != LLONG_MAX && Dist[n] > Dist[w] + edge.weight) {
                    Dist[n] = Dist[w] + edge.weight;
                    Parent[n] = w;
                    pq.emplace(Dist[n], n);
                }
            }
        }
    }
}

// Function to read and apply changes from a file (no scaling)
void readChangesFromFile(const string& filename, Graph* graph, vector<long long>& Dist, vector<int>& Parent, int source) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open changes file " << filename << endl;
        return;
    }

    string line;
    // Read number of deletions
    getline(file, line);
    int numDeletions = stoi(line);

    // Process deletions
    for (int i = 0; i < numDeletions; i++) {
        getline(file, line);
        stringstream ss(line);
        int u, v;
        double w;
        ss >> u >> v >> w;
        long long weight = static_cast<long long>(w); // No scaling

        cout << "\nDeleting edge e(" << u << "," << v << ") with weight " << w << ":\n";
        if (graph->deleteEdge(u, v)) {
            UpdateSingleChange(*graph, source, Dist, Parent, u, v, LLONG_MAX, true);
        } else {
            cout << "Edge e(" << u << "," << v << ") not found in either direction.\n";
        }
        for (int j = 0; j < graph->V; j++) {
            cout << "Vertex " << j << ": Dist = " << (Dist[j] == LLONG_MAX ? "INF" : to_string(Dist[j])) 
                 << ", Parent = " << Parent[j] << "\n";
        }
    }

    // Read number of additions
    getline(file, line);
    int numAdditions = stoi(line);

    // Process additions
    for (int i = 0; i < numAdditions; i++) {
        getline(file, line);
        stringstream ss(line);
        int u, v;
        double w;
        ss >> u >> v >> w;
        long long weight = static_cast<long long>(w); // No scaling

        cout << "\nAdding edge e(" << u << "," << v << ") with weight " << w << ":\n";
        graph->addEdge(u, v, weight);
        UpdateSingleChange(*graph, source, Dist, Parent, u, v, weight, false);
        for (int j = 0; j < graph->V; j++) {
            cout << "Vertex " << j << ": Dist = " << (Dist[j] == LLONG_MAX ? "INF" : to_string(Dist[j])) 
                 << ", Parent = " << Parent[j] << "\n";
        }
    }

    file.close();
}

// Main function to test the implementation with file reading
int main() {
    Graph* graph = readGraphFromFile("graph.txt");
    if (!graph) {
        return 1; // Exit if file reading failed
    }

    // Arrays for Dist and Parent
    vector<long long> Dist;
    vector<int> Parent;

    // Compute initial SSSP from source vertex 0
    dijkstra(*graph, 0, Dist, Parent);

    cout << "Initial SSSP Distances:\n";
    for (int i = 0; i < graph->V; i++) {
        cout << "Vertex " << i << ": Dist = " << (Dist[i] == LLONG_MAX ? "INF" : to_string(Dist[i])) 
             << ", Parent = " << Parent[i] << "\n";
    }

    // Read and apply changes from changes.txt
    readChangesFromFile("changes.txt", graph, Dist, Parent, 0);

    // Cleanup
    delete graph;

    return 0;
}