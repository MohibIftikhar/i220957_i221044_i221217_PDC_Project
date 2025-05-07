## SSSP Algorithms

### PARALLEL

The *Parallel* directory contains three implementations of the parallel sssp algorithm
1) OpenMP
2) MPI
3) MPI + OpenMP

### SERIAL

The *Serial* directory contains the serial implementation of the sssp algorithm


### METIS

The *Metis* directory contains 2 versions of subgraph partitioning. 
* Use the cpp file implementation for sssp computation in this repository. (Consult the README.md in the *Metis* directory for further clarification)
* The py file was used for testing and visioning of what was happening.

### DATASETS

The *Datasets* directory contains somewhat large datasets for testing.
* It is to be noted that these are not real-world test datas and were merely generated for testing.

### EXAMPLES

The *ExampleDataSets* directory contains very small example graphs that were used for testing the accuracy of the algorithms i.e. whether the paths it calculates are correct or not.