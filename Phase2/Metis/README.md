The graph-subgraph.py file takes a txt file as an input of the format:
"n m w" where an edge exists between n and m of weight w.

Run the following command on your WSL or Ubuntu :

[python3] [python file] [input file] [output file] [subgraph prefix] [--num-partitions NUM_PARTITIONS] [--weight-scale WEIGHT_SCALE] [--has-header]
python3 graph-subgraph.py graph.txt graph.metis subgraph --num-partitions 2 

*The subgraph prefix is the name used for the out files of the subgraphs after partitioning

You should get:
1) The output file with your specified file name that is formatted according to the metis requirement.
2) A "vertex_mapping.txt" that shows the old node ids mapped to new ones. (needed in case previous node indexes start from zero)
3) A "*output-filename*.part.*partitions*" file with subgraph assignment for each node. (Each line represents a node and the value represents the subgraph assigned to that node)
4) "*subgraph-prefix* _ *partiton* _ nodes.txt" + "*subgraph-prefix*_*partition*.txt" for the number of partitions you specified. (Default partitions = 2)