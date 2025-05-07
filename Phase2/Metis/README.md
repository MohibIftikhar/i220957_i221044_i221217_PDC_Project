## **graph-subgraph.cpp:**

   -> Makes use of the metis.h header.
   
   -> Internally processes the .part file to extract the subgraphs.
   
   -> Includes the ghost nodes in the partitioned subgraphs.

   The graph-subgraph.cpp file takes a txt file as an input of the format:
   "n m w" where an edge exists between n and m of weight w.
    
   Run the following command on your WSL or Ubuntu : (Provide the number of partitions as command line input)

        g++ graph-subgraph.cpp -o ex -lmetis
        ./ex 3

   or alternatively:

        g++ graph-subgraph.cpp -o ex -I/usr/include/metis -L/usr/lib -lmetis
        ./ex 3

   After running the program you should get:
   
   1) "*subgraph-prefix* _ *partiton* _ nodes.txt" + "*subgraph-prefix*_*partition*.txt" for the number of partitions you specified. (Default partitions = 2)

   2) "vertex_to_partion.txt" that contains the mapping of the nodes to their respective subgraphs.

   -> You shall find these files in the *Parallel* directory where they will be used for further processing.
    
   -> The node files shall include the ghost nodes under the "#Ghost nodes" comment.
        
   -> In the subgraph file the ghost node edges shall be included as well.



## **graph-subgraph.py:**

   -> Does not use the metis.h header.
   
   -> Runs the command gpmetis for graph partitioning.
   
   -> Does not handle ghost nodes(only provides the divided subgraphs).

   The graph-subgraph.py file takes a txt file as an input of the format:
   "n m w" where an edge exists between n and m of weight w.

   Run the following command on your WSL or Ubuntu :

   [python3] [python file] [input file] [output file] [subgraph prefix] [--num-partitions NUM_PARTITIONS] [--weight-scale WEIGHT_SCALE] [--has-header]

        python3 graph-subgraph.py graph.txt graph.metis subgraph --num-partitions 2 

   *The subgraph prefix is the name used for the out files of the subgraphs after partitioning

   After running the program you should get:
   
   1) The output file with your specified file name that is formatted according to the metis requirement.
    
   2) A "vertex_mapping.txt" that shows the old node ids mapped to new ones. (needed in case previous node indexes start from zero)
    
   3) A "*output-filename*.part.*partitions*" file with subgraph assignment for each node. (Each line represents a node and the value represents the subgraph assigned to that node)
    
   4) "*subgraph-prefix* _ *partiton* _ nodes.txt" + "*subgraph-prefix*_*partition*.txt" for the number of partitions you specified. (Default partitions = 2)
