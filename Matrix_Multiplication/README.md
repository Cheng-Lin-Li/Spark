## This is an implementation of Two Phases Matrix Multiplication algorithm in Spark 2.1.1 with Python 2.7
Matrix Multiplication: Two Phases approach to deal with huge matrix multiplication on spark platform

## Algorithm: Matrix Multiplication: Two Phases approach

## Task:
The task is to implement SON algorithm in Apache Spark using Python. 
Given a set of baskets, SON algorithm divides them into chunks/partitions and then proceed in two stages. 
First, local frequent itemsets are collected, which form candidates; 
next, it makes second pass through data to determine which candidates are globally frequent.

#### Usage: bin/spark-submit TwoPhase_Matrix_Multiplication.py <mat-A/values.txt> <mat-B/values.txt> <output.txt>
 

#### Input: Takes two folders with mat-A/values.txt or mat-B/values.txt as the input

#### Output: Save all results into one text file. 
