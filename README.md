# Spark
There are Python 2.7 codes and learning notes for Spark 2.1.1

The repository provides demo programs for implementations of basic algorithms on Spark 2.1.1 and hadoop with Python 2.7. 
I hope these programs will help people understand the power of distributed parallel computing via map-reduce on Spark platform.

I will enrich those implementations and descriptions from time to time. If you include any of my work into your website or project; please add a link to this repository and send me an email to let me know.

Your comments are welcome.
Thanks,

|Algorithm|Description|Link|
|------|------|--------|
|A-priori SON| Finding Frequent Itemsets: SON Algorithm by A-Priori algorithm in stage 1. The implementation include Savasere, Omiecinski, and Navathe (SON) algorithm as a class and an A-Priori algorithm in python class encapsulates all functions which implement by static functions to support Spark RDD to call. |[Source Code](https://github.com/Cheng-Lin-Li/Spark/blob/master/A-Priori_SON/A-Priori_SON.py)|
|ALS with UV Decomposition|An implementation of UV Decomposition in Alternating Least Squares (ALS) Algorithm by Spark. The task is to modify the parallel implementation of ALS (alternating least squares) algorithm in Spark, so that it takes a utility matrix as the input and process by UV decomposition, and output the root-mean-square deviation (RMSE) into standard output or a file after each iteration. The code for the algorithm is als.py under the <spark-2.1.0 installation directory>/examples/src/main/python.|[Source Code](https://github.com/Cheng-Lin-Li/Spark/blob/master/ALS/ALS.py)|
|K-Means by TF-IDF |A similarity algorithm implementation of TF-IDF algorithm with cosin similarity implementation on spark platform as the measure of K-Means. The implementation of k-means is provided by Spark in examples/src/main/python/ml/kmeans_example.py. |[Source Code](https://github.com/Cheng-Lin-Li/Spark/blob/master/KMeans/kmeans.py)|
|Matrix Multiplication by Two Phases approach|Matrix Multiplication: Two Phases approach to deal with huge matrix multiplication on spark platform|[Source Code](https://github.com/Cheng-Lin-Li/Spark/blob/master/Matrix_Multiplication/TwoPhase_Matrix_Multiplication.py)|
|Minhash and Locality-Sensitive Hash (LSH)|An implementation of MinHash and LSH to find similar set/users from their items/movies preference data. The implementation is finding similar sets/users by minhash and LSH in Spark platform to speed up the calculation - calculating the similarity by Jaccard similarity (or Jaccard coefficient). LSH: The implementation of Locality-Sensitive Hash in Spark. Based on Minhash functions to get the signature for each set/users and split these minhash functions by band. Each band will contain R minhash functions results|[Source Code]https://github.com/Cheng-Lin-Li/Spark/blob/master/MinHash_LSH/lshrec.py)|
|UV decomposition| An implementation of UV decomposition algorithm. The implementation goal is to decompose user/product ratings matrix M into lower-rank matrices U and V such that the difference between M and UV is minimized. Root-mean-square error (RMSE) is adopted to measure the quality of decomposition| [Source Code](https://github.com/Cheng-Lin-Li/Spark/blob/master/UV_decomposition/UV.py)|


## Reference:
* Foundations and Appliations of Data Mining - INF553 at University of Southern California
* Spark Example code - k-means, : The K-means algorithm written from scratch against PySpark. In practice,one may prefer to use the KMeans algorithm in ML, as shown in <spark-2.1.0 installation directory>/examples/src/main/python/ml/kmeans_example.py.
* Spark Example code - ALS, : The code for the algorithm is als.py under the <spark-2.1.0 installation directory>/examples/src/main/python


Cheng-Lin Li@University of Southern California
chenglil@usc.edu or 
clark.cl.li@gmail.com

