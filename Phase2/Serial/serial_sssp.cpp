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

        ss >> type;

        if (type == 'D' || type == 'd') {
            if (!(ss >> u >> v)) {
                cerr << "Error: Invalid delete entry in " << filename << endl;
                continue;
            }
            if (u < 0 || u >= graph->V || v < 0 || v >= graph->V) {
                cerr << "Error: Invalid vertex index in delete entry\n";
                continue;
            }
            graph->deleteEdge(u, v);
            dijkstra(*graph, source, Dist, Parent);
        } else if (type == 'I' || type == 'i') {
            if (!(ss >> u >> v >> w)) {
                cerr << "Error: Invalid insert entry in " << filename << endl;
                continue;
            }
            if (u < 0 || u >= graph->V || v < 0 || v >= graph->V) {
                cerr << "Error: Invalid vertex index in insert entry\n";
                continue;
            }
            if (w < 0) {
                cerr << "Error: Negative weight detected in insert entry\n";
                continue;
            }
            graph->addEdge(u, v, w);
            dijkstra(*graph, source, Dist, Parent);
        } else {
            cerr << "Error: Unknown change type '" << type << "' in " << filename << endl;
        }
    }

    file.close();
}

// Main
int main(int argc, char* argv[]) {
    cout << "\n----------------- SERIAL SSSP -----------------\n\n";

    if (argc < 2 || argc > 3) {
        cerr << "Usage: " << argv[0] << " <graph_file> [changes_file]\n";
        return 1;
    }

    string graph_filename = argv[1];
    string changes_filename = (argc == 3) ? argv[2] : "";

    double t_total_start = omp_get_wtime();

    double t_read_start = omp_get_wtime();
    Graph* graph = readGraphFromFile(graph_filename);
    if (!graph) {
        cerr << "Error: Could not read graph from file: " << graph_filename << "\n";
        return 1;
    }
    double t_read_end = omp_get_wtime();

    int source = 0;
    vector<double> Dist(graph->V, numeric_limits<double>::infinity());
    vector<int> Parent(graph->V, -1);

    double t_dijkstra_start = omp_get_wtime();
    dijkstra(*graph, source, Dist, Parent);
    double t_dijkstra_end = omp_get_wtime();

    double t_changes_start = 0.0, t_changes_end = 0.0;
    if (!changes_filename.empty()) {
        t_changes_start = omp_get_wtime();
        readChangesFromFile(changes_filename, graph, Dist, Parent, source);
        t_changes_end = omp_get_wtime();
    }

    double t_write_start = omp_get_wtime();
    ofstream out("output_serial.txt");
    if (!out.is_open()) {
        cerr << "Error: Could not open output_serial.txt for writing\n";
        delete graph;
        return 1;
    }

    out << fixed << setprecision(1);
    out << "Vertex Distance Parent\n";
    for (int i = 0; i < graph->V; ++i) {
        out << i << " " << Dist[i] << " " << Parent[i] << "\n";
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
    cout << "Total Vertices       : " << graph->V << "\n";
    cout << "Source Vertex        : " << source << "\n";
    cout << "\n--- Timing (ms) ---\n";
    cout << "Graph Read           : " << ms(t_read_start, t_read_end) << " ms\n";
    cout << "Dijkstra's Algorithm : " << ms(t_dijkstra_start, t_dijkstra_end) << " ms\n";
    if (!changes_filename.empty()) {
        cout << "Apply Changes        : " << ms(t_changes_start, t_changes_end) << " ms\n";
    } else {
        cout << "Apply Changes        : skipped\n";
    }
    cout << "Write Output         : " << ms(t_write_start, t_write_end) << " ms\n";
    cout << "Total Time           : " << ms(t_total_start, t_total_end) << " ms\n";
    cout << "============================\n";
    cout << "Output written to output_serial.txt\n\n";

    delete graph;
    return 0;
}

// g++ serial_sssp.cpp -o s1 -fopenmp
// time ./s1 ../Datasets/bio-CE/bio-CE-HT.edges
// time ./s1 ../Datasets/bio-CE/bio-CE-HT.edges ../Datasets/bio-CE/bio-CE-HT_updates_500.edges

// time ./s1 ../Datasets/bio-h/bio-h.edges