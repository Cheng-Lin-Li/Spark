## This is an implementation of TF-IDF algorithm with cosin similarity algorithm in Spark 2.1.1 with Python 2.7
A similarity algorithm implementation of TF-IDF algorithm with cosin similarity implementation on spark platform as the measure of K-Means. The implementation of k-means is provided by Spark in examples/src/main/python/ml/kmeans_example.py.

## Algorithm: TF-IDF algorithm with cosin similarity

## Task:
The task is to implement TF-IDF algorithm with cosin similarity in Apache Spark using Python. 
Given a set of vectors to present a document as input, calculating the TF-IDF with cosin similarity to cluster those documents via similarity.

#### Usage: bin/spark-submit kmeans.py <file> <k> <convergeDist> [outputfile.txt]
	k - the number of clusters
	convergDist - The converge distance/similarity to stop program iterations.
	
	example: 	bin\spark-submit .\kmeans.py .\docword.enron_s.txt 10 0.00001 kmeans_output.txt

#### Input: Takes input file from folder as the input

		
#### Output: Save all results into one text file. 

kmeans_output.txt