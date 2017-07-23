## This is an implementation of Savasere, Omiecinski, and Navathe (SON) algorithm in Spark 2.1.1 with Python 2.7
Finding Frequent Itemsets: SON Algorithm by A-Priori algorithm in stage 1

## Algorithm: Savasere, Omiecinski, and Navathe (SON) Algorithm, A-Priori algorithm

## Task:
The task is to implement SON algorithm in Apache Spark using Python. 
Given a set of baskets, SON algorithm divides them into chunks/partitions and then proceed in two stages. 
First, local frequent itemsets are collected, which form candidates; 
next, it makes second pass through data to determine which candidates are globally frequent.

#### Usage: bin/spark-submit A-Priori_SON.py <baskets.txt> <.3> <output.txt>
 
    1. baskets.txt is a text file which contains a basket (a list of comma-separated item numbers) per line. 
    For example
      1,2,3 
      1,2,5 
      1,3,4 
      2,3,4 
      1,2,3,4 
      2,3,5 
      1,2,4 
      1,2
      1,2,3 
      1,2,3,4,5
  
    2. <.3> = minimum support ratio (that is, for an itemset to be frequent, it should appear in at least 30% of the baskets).
   
    3. output.txt is the output result file.
  

#### Input: Take a baskets (baskets.txt) as the input

#### Output: Save all frequent itemsets into one text file. 
Each line of the file contains one itemset (a list of comma-separated item numbers). The order doesn’t matter. 

    For example,
    4
    1,3,4 
    1,2,3 2
    1,3 2,4 
    2,3 1
    2,3,4 1,4 
    3
    3,4
    1,2,4 
    2,5 
    1,2 5
