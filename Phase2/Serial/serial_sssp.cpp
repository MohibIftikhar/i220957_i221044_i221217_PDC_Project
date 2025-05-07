#include <iostream>
#include <vector>
#include <list>
#include <queue>
#include <climits>
#include <limits>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <functional>
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

// Batch update structures for incremental SSSP
enum class ChangeType { INSERT, DELETE };
using Weight = double;
struct EdgeChange { ChangeType type; int u, v; Weight weight; };
struct SSSPResult { std::vector<Weight> dist; std::vector<int> parent; };

void SingleChange(const EdgeChange& change, Graph& G, SSSPResult& T) {
    int u = change.u; int v = change.v; Weight w = change.weight;
    Weight old_du = (u>=0 && u<G.V) ? T.dist[u] : std::numeric_limits<Weight>::infinity();
    Weight old_dv = (v>=0 && v<G.V) ? T.dist[v] : std::numeric_limits<Weight>::infinity();

    // Check existing edge
    bool existed = false; 
    Weight existing_w = std::numeric_limits<Weight>::infinity();
    if (u>=0 && u<G.V) 
        for (auto& e: G.vertices[u]) 
            if (e.to==v) { 
                existed=true; 
                existing_w=e.weight; 
                break; 
            }

    // Apply change
    if (change.type==ChangeType::INSERT) 
        G.addEdge(u,v,w);
    else { 
        if (existed) G.deleteEdge(u,v); 
        else return; 
    }
    // SPFA-like update
    std::queue<int> q; 
    std::vector<bool> inq(G.V, false);
    auto enqueue=[&] (int x) { 
        if(x >= 0 && x < G.V && !inq[x]) {
            q.push(x);
            inq[x]=true;
        } 
    };
    if (change.type==ChangeType::INSERT) {
        // Update the shortest path if the old distance + new weight is less than already calculated distance
        if (old_du < std::numeric_limits<Weight>::infinity() && old_du + w < T.dist[v]) { 
            T.dist[v]=old_du+w; 
            T.parent[v]=u; 
            enqueue(v);
        } 
        if (old_dv< std::numeric_limits<Weight>::infinity() && old_dv+w < T.dist[u]) { 
            T.dist[u]=old_dv+w; 
            T.parent[u]=v; 
            enqueue(u);
        } 
    } else {
        bool invalid = false; 
        std::queue<int> ch;
        if ( T.parent[v] == u ) { 
            T.dist[v]=std::numeric_limits<Weight>::infinity(); 
            T.parent[v]=-1; 
            enqueue(v); 
            invalid=true; 
            ch.push(v);
        } 
        else if ( T.parent[u] == v ) { 
            T.dist[u]=std::numeric_limits<Weight>::infinity(); 
            T.parent[u]=-1; 
            enqueue(u); 
            invalid=true; 
            ch.push(u);
        } 
        while(!ch.empty()) {
            int cur=ch.front(); 
            ch.pop(); 
            for(int i = 0 ; i < G.V ; ++i) 
                if(T.parent[i] == cur) {
                    T.dist[i] = std::numeric_limits<Weight>::infinity(); 
                    T.parent[i] = -1; 
                    enqueue(i); 
                    ch.push(i);
                } 
            }
        if (!invalid) { 
            enqueue(u); 
            enqueue(v); 
        }
    }
    while(!q.empty()) {
        int z=q.front(); q.pop(); inq[z]=false;
        // if unreachable, try reconnect
        if (T.dist[z] == std::numeric_limits<Weight>::infinity()){
            Weight best = std::numeric_limits<Weight>::infinity(); 
            int bp = -1;
            for(auto& e: G.vertices[z]) { 
                int nb = e.to; 
                if (T.dist[nb]< std::numeric_limits<Weight>::infinity() && T.dist[nb] + e.weight < best) { 
                    best = T.dist[nb] + e.weight;
                    bp = nb; 
                }
            }
            if (best < T.dist[z]) { 
                T.dist[z]=best; 
                T.parent[z]=bp; 
                enqueue(z); 
                continue; }
        }
        // propagate
        if (T.dist[z]< std::numeric_limits<Weight>::infinity()){
            for(auto& e: G.vertices[z]){ int nb=e.to; Weight wt=e.weight;
                if (T.dist[z] + wt < T.dist[nb] ) { 
                    T.dist[nb] = T.dist[z] + wt; 
                    T.parent[nb] = z; 
                    enqueue(nb); 
                } 
            }
        }
    }
}

void process_batch_sequential(Graph& G, SSSPResult& T, const std::vector<EdgeChange>& batch) {
    for(const auto& ch: batch) SingleChange(ch, G, T);
}

std::vector<EdgeChange> readBatchChanges(const std::string& fname, Graph& G) {
    std::vector<EdgeChange> batch; std::ifstream f(fname);
    if(!f.is_open()){ std::cerr<<"Error: Could not open batch file "<<fname<<std::endl; return batch; }
    std::string line;
    while(std::getline(f,line)){
        if(line.empty()) continue;
        std::stringstream ss(line);
        char t; int u,v; double w=0;
        ss>>t>>u>>v; if(t=='I'||t=='i'){ ss>>w; batch.push_back({ChangeType::INSERT,u,v,w}); }
        else if(t=='D'||t=='d'){ batch.push_back({ChangeType::DELETE,u,v,0}); }
    }
    return batch;
}

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

void UpdateSingleChange(Graph& graph, int source, vector<double>& Dist, vector<int>& Parent,
                        int u, int v, long long W_uv, bool isDeletion = false) {
    dijkstra(graph, source, Dist, Parent);
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
            if (graph->deleteEdge(u, v)) {
                UpdateSingleChange(*graph, source, Dist, Parent, u, v, LLONG_MAX, true);
            } else {
                cout << "Edge not found.\n";
            }
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
            UpdateSingleChange(*graph, source, Dist, Parent, u, v, w, false);
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

        auto batch = readBatchChanges(changes_filename, *graph);
        SSSPResult batchResult{Dist, Parent};
        process_batch_sequential(*graph, batchResult, batch);
        Dist = std::move(batchResult.dist);
        Parent = std::move(batchResult.parent);
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
// time ./s1 ../Datasets/bio-h/bio-h.edges ../Datasets/bio-h/bio-h_updates_10000.edges
// time ./s1 ../Datasets/bio-h/bio-h.edges ../Datasets/bio-h/bio-h_updates_12500.edges
// time ./s1 ../Datasets/bio-h/bio-h.edges ../Datasets/bio-h/bio-h_updates_15000.edges


// time ./s1 ../Datasets/bio-CX/bio-CE-CX.edges
// time ./s1 ../Datasets/bio-CX/bio-CE-CX.edges ../Datasets/bio-CX/bio-CE-CX_updates_10000.edges