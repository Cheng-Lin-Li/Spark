## This is an implementation of Alternating Least Squares (ALS) algorithm in Spark 2.1.1 with Python 2.7

## Algorithm: Alternating Least Squares (ALS) Algorithm

## Task:
The task is to modify the parallel implementation of ALS (alternating least squares) algorithm in Spark, so that it takes a utility matrix as the input, and output the root-mean-square deviation (RMSE) into standard output or a file after each iteration. The code for the algorithm is als.py under the <spark-2.1.0 installation directory>/examples/src/main/python.

#### Usage: bin/spark-submit ALS.py input-matrix n m f k p [output-file]
  1. n is the number of rows (users) of the matrix
  
  2. m is the number of columns (products).
   
  3. f is the number of dimensions/factors in the factor model. That is, U is n-by-f matrix, while V is f-by-m matrix.
  
  4. k is the number of iterations.

  5. p, which is the number of partitions for the input-matrix

  6. output-file, which is the path to the output file. This parameter is optional.

#### Input: Take a utility matrix (mat.dat) as the input

#### Output: Output root-mean-square deviation (RMSE) into standard output or a file after each iteration
After each iteration, output RMSE with 4 floating points.
The "%.4f" % RMSE is adapted to format the RMSE value, and save into file as follows. 

1.0019 

0.9794 

0.8464 

...

