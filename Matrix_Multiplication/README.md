## This is an implementation of Two Phases Matrix Multiplication algorithm in Spark 2.1.1 with Python 2.7
Matrix Multiplication: Two Phases approach to deal with huge matrix multiplication on spark platform

## Algorithm: Matrix Multiplication: Two Phases approach

## Task:
The task is to implement MapReduce prgram in Apache Spark using Python. 
The program computes the multiplication of two given matrices using the two phase approach.
The implementation does not use join (and any outer join variations) in code, but use other transformations and actions to implement the MapReduce function in Spark.

#### Usage: bin/spark-submit TwoPhase_Matrix_Multiplication.py <mat-A/values.txt> <mat-B/values.txt> <output.txt>
 

#### Input: Takes two folders with mat-A/values.txt or mat-B/values.txt as the input

#### Output: Save all results into one text file. 
