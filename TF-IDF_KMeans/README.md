## This is an implementation of TF-IDF algorithm with cosin similarity algorithm in Spark 2.1.1 with Python 2.7
A similarity algorithm implementation of TF-IDF algorithm with cosin similarity implementation on spark platform as the measure of K-Means. The implementation of k-means is provided by Spark in examples/src/main/python/ml/kmeans_example.py.

## Algorithm: TF-IDF algorithm with cosin similarity

## Task:
The task is to implement TF-IDF algorithm with cosin similarity in Apache Spark using Python. 
Given a set of vectors to present a document as input, calculating the TF-IDF with cosin similarity to cluster those documents via similarity.

#### Usage: bin/spark-submit kmeans.py file k convergeDist [outputfile.txt]
	k - the number of clusters
	convergDist - The converge distance/similarity to stop program iterations.
	
	example: 	bin\spark-submit .\kmeans.py .\docword.enron_s.txt 10 0.00001 kmeans_output.txt

#### Input: Takes input file from folder as the input

The input file which has the following format:

	39861 
	28102 
	3710420 
	1 118 1 
	1 285 1 
	1 1229 1 
	1 1688 1 
	1 2068 1 
	…
The first line is the number of documents in the collection (39861). The second line is the number of words in the vocabulary (28102). Note that the vocabulary only contains the words that appear in at least 10 documents. The third line (3710420) is the number of words that appear in at least one document.

Starting from the fourth line, the content is [document id] [word id] [tf]. 

For example, document #1 has word #118 (i.e., the line number in the vocabulary file) that occurs once.

		
#### Output: Save all results into one text file. 

kmeans_output.txt

For each final center, output the number of its nonzero values as following:

	87 
	60 
	50 
	56 
	…

It means total 4 clusters, the #0 cluster center is a sparse vector that has 87 nonzero values. The order doesn’t matter
