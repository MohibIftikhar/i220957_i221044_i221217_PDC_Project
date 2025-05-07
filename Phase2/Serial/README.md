## Serial SSSP Algorithm Implementation

### serial_sssp.cpp

This C++ program implements a serial version of Dijkstra's algorithm to compute the Single Source Shortest Path (SSSP) in an undirected weighted graph

For compilation and execution run the following commands on your WSL or Ubuntu :

Usage: <graph_file> [changes_file]

    g++ serial_sssp.cpp -o s1 -fopenmp
    time ./s1 ../Datasets/bio-CE/bio-CE-HT.edges 

For applying changes to the graph execute it as follows:

    time ./s1 ../Datasets/bio-CE/bio-CE-HT.edges ../Datasets/bio-CE/bio-CE-HT_updates_500.edges

Update the file path arguments as needed before executing the program

The final results will be written to:   

    output_serial.txt

