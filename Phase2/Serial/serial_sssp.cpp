#include <iostream>
#include <vector>
#include <list>
#include <queue>
#include <limits>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <omp.h>

using namespace std;

// Structure for an edge
struct Edge {
    int to;
    double weight;
    Edge(int t, double w) : to(t), weight(w) {}
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

    void addEdge(int from, int to, double weight) {
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
        if (!(ss >> from >> to >> distance)) {
            cerr << "Error: Invalid edge format in " << filename << endl;
            file.close();
            return nullptr;
        }
        if (distance < 0) {
            cerr << "Error: Negative weight detected in " << filename << endl;
            file.close();
            return nullptr;
        }
        maxVertex = max(maxVertex, max(from, to));
        edges.emplace_back(from, to, distance);
    }

    file.close();

    int V = maxVertex + 1;
    Graph* graph = new Graph(V);

    for (const auto& edge : edges) {
        int from = get<0>(edge);
        int to = get<1>(edge);
        double weight = get<2>(edge);
        graph->addEdge(from, to, weight);
    }

    return graph;
}

// Dijkstra's algorithm
void dijkstra(Graph& graph, int source, vector<double>& Dist, vector<int>& Parent) {
    using PQNode = pair<double, int>;
    priority_queue<PQNode, vector<PQNode>, greater<PQNode>> pq;

    Dist.assign(graph.V, numeric_limits<double>::infinity());
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
            double weight = edge.weight;
            if (!graph.active[v]) continue;

            if (Dist[v] > Dist[u] + weight) {
                Dist[v] = Dist[u] + weight;
                Parent[v] = u;
                pq.emplace(Dist[v], v);
            }
        }
    }
}

// Apply changes from a file
void readChangesFromFile(const string& filename, Graph* graph, vector<double>& Dist, vector<int>& Parent, int source) {
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
        if (!(ss >> type >> u >> v >> w)) {
            cerr << "Error: Invalid change entry in " << filename << endl;
            //continue;
        }
        if (u < 0 || u >= graph->V || v < 0 || v >= graph->V) {
            cerr << "Error: Invalid vertex index in " << filename << endl;
            //continue;
        }
        if (w < 0) {
            cerr << "Error: Negative weight detected in " << filename << endl;
            //continue;
        }

        if (type == 'D' || type == 'd') {
            graph->deleteEdge(u, v);
            dijkstra(*graph, source, Dist, Parent);
        } else if (type == 'I' || type == 'i') {
            graph->addEdge(u, v, w);
            dijkstra(*graph, source, Dist, Parent);
        } else {
            cerr << "Error: Unknown change type '" << type << "' in " << filename << endl;
        }
    }

    file.close();
}

// Main
int main() {

    cout << endl;
    cout << "----------------- SERIAL SSSP -----------------\n" << endl;
    double start_time, end_time;

    // Step 1: Read graph from file
    start_time = omp_get_wtime();
    Graph* graph = readGraphFromFile("graph.txt");
    if (!graph) {
        cerr << "Error: Could not read graph from file\n";
        return 1;
    }
    end_time = omp_get_wtime();
    cout << "Time to read graph from file: " << end_time - start_time << " seconds\n";

    // Step 2: Initialize vectors for distance and parent
    vector<double> Dist(graph->V, numeric_limits<double>::infinity());
    vector<int> Parent(graph->V, -1);
    int source = 0;

    // Step 3: Run Dijkstra's algorithm
    start_time = omp_get_wtime();
    dijkstra(*graph, source, Dist, Parent);
    end_time = omp_get_wtime();
    cout << "Time to run Dijkstra's algorithm: " << end_time - start_time << " seconds\n";

    // Step 4: Read changes from file
    start_time = omp_get_wtime();
    readChangesFromFile("changes.txt", graph, Dist, Parent, source);
    end_time = omp_get_wtime();
    cout << "Time to read changes from file: " << end_time - start_time << " seconds\n";

    // Step 5: Write output to file
    start_time = omp_get_wtime();
    ofstream out("output2.txt");
    if (!out.is_open()) {
        cerr << "Error: Could not open output.txt for writing\n";
        delete graph;
        return 1;
    }

    out << fixed << setprecision(1);
    out << "Vertex Distance Parent\n";
    for (int i = 0; i < graph->V; ++i) {
        out << i << " " << Dist[i] << " " << Parent[i] << "\n";
    }
    out.close();
    end_time = omp_get_wtime();
    cout << "Time to write output to file: " << end_time - start_time << " seconds\n";

    cout << "Output written to output2.txt\n";

    // Step 6: Clean up and delete graph
    delete graph;

    return 0;
}

// g++ serial_sssp.cpp -o s1
// ./s1