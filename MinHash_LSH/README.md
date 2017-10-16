## This is an implementation of MinHash and Locality-Sensitive Hash (LSH) algorithm in Spark 2.1.1 with Python 2.7
An implementation of MinHash and LSH to find similar set/users from their items/movies preference data. The implementation is finding similar sets/users by minhash and LSH in Spark platform to speed up the calculation - calculating the similarity by Jaccard similarity (or Jaccard coefficient). LSH: The implementation of Locality-Sensitive Hash in Spark. Based on Minhash functions to get the signature for each set/users and split these minhash functions by band. Each band will contain R minhash functions results.

## Algorithm: MinHash and Locality-Sensitive Hash (LSH) algorithm

## Task:
Given a set of vectors to present a document as input to cluster those documents via MinHash and Locality-Sensitive Hash (LSH) algorithm.

#### Usage: bin/spark-submit input_file.txt output_file.txt


#### Input: Takes input file from folder as the input

		
#### Output: Save all results into one text file. 

