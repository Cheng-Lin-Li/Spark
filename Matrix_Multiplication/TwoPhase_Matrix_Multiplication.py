#!/usr/bin/env python
# encoding: utf-8
'''
Matrix Multiplication: TwoPhase approach


@author: Cheng-Lin Li a.k.a. Clark Li

@copyright:    2017 Cheng-Lin Li@University of Southern California. All rights reserved.

@license:    Licensed under the GNU v3.0. https://www.gnu.org/licenses/gpl.html

@contact:    chenglil@usc.edu
@version:    1.0

@create:    February, 10, 2017
@updated:   February, 10, 2017
'''
from __future__ import print_function 
# Import the necessary Spark library classes, as well as sys
from pyspark import SparkConf, SparkContext, StorageLevel
import sys
from operator import add

APP_NAME = "TwoPhaseMatrixMultiplication"
DEBUG = True

# Get input and output parameters
if len(sys.argv) != 4:
    print('Usage: ' + sys.argv[0] + ' <mat-A/values.txt> <mat-B/values.txt> <output.txt>')
    sys.exit(1)

# Assign the input and output variables
matrix_a = sys.argv[1]
matrix_b = sys.argv[2]
output = sys.argv[3]

# Generate multiply elements in two matrixes 
def mul(elements): #elements = (key1, [('A',i, value)...('B',j, value)...)])

    _ma = list()
    _mb = list()
    _result = list()
    for _lst in elements[1]: # element[1]=[('A',i, value)...('B',j, value)...)] 
        if str(_lst[0]) == "A":
            _ma.append([str(_lst[1]), str(_lst[2])])
        else:
            _mb.append([str(_lst[1]), str(_lst[2])])

    for _elA in _ma: #[[idx1, v1],[idx2, v2]]
        for _elB in _mb:
            _result.append ( ( (str(_elA[0]),str(_elB[0])), int(_elA[1])*int(_elB[1]) ) )
    if DEBUG: print('_result=%s'%(str(_result)))
    return _result #((i,j), multiplication)

# Create a configuration for this job
conf = SparkConf().setAppName(APP_NAME)

# Create a context for the job.
sc = SparkContext(conf=conf)

#creating RDD from external file for Matrix A and B.
rddALines = sc.textFile(matrix_a) # ["0,0,A[0,0]", ..., "i, j, A[i,k]"]
rddBLines = sc.textFile(matrix_b) # ["0,0,B[0,0]", ..., "k, j, B[k,j]"]

# Phase 1 Map Task
#Create an RDD with: The columns of A, the rows of B
# [i, k, A[i,k]] => (k, ('A', i, A[i, k]))
# [k, j, B[k,j]] => (k, ('B', j, B[k, j]))

rddPhaseOneMapperA = rddALines.map(lambda x:x.split(',')).map(lambda data: (data[1],['A', data[0],data[2]]))
rddPhaseOneMapperB = rddBLines.map(lambda x:x.split(',')).map(lambda data: (data[0],['B', data[1],data[2]]))
rddPhaseOneMapperResult = rddPhaseOneMapperA.union(rddPhaseOneMapperB).groupByKey().map(lambda x:(x[0], list(x[1])))
if DEBUG: print("==========>Phase 1 Map finished")

# Phase 1 Reduce Task
# (key1, [('A',i, value)...('B',j, value)...], ...key n, [....]) 
rddPhaseOneReducer = rddPhaseOneMapperResult.flatMap(lambda e: mul(e)).persist(StorageLevel.MEMORY_ONLY_SER)
if DEBUG: print("==========>Phase 1 Reduce collect() start")
#PhaseOneReducerResult = rddPhaseOneReducer.collect()
if DEBUG: print("==========>Phase 1 Reduce finished")
# Phase 2 Map Task
#rddP2Input = sc.parallelize(PhaseOneReducerResult)
#rddPhaseTwoMapper = rddP2Input.map(lambda x: x)
rddPhaseTwoMapper = rddPhaseOneReducer.map(lambda x: x)
if DEBUG: print("==========>Phase 2 Map finished")
# Phase 2 Reduce Task
rddPhaseTwoReducer= rddPhaseTwoMapper.reduceByKey(lambda x, y: x+y)
rddPhaseTwoReducerResult = rddPhaseTwoReducer.collect()
if DEBUG: print("==========>Phase 2 Reduce finished")
#print the results
try:
    if DEBUG != True :
        orig_stdout = sys.stdout
        f = file(output, 'w')
        sys.stdout = f
    else:
        pass
        
    for _x in rddPhaseTwoReducerResult:
        print("%s,%s\t%d"%(_x[0][0], _x[0][1], _x[1]))

    sys.stdout.flush()       
    if DEBUG != True :
        sys.stdout = orig_stdout                   
        f.close()
    else:
        pass        
except IOError as _err:
    if (DEBUG == True): 
        print ('File error: ' + str (_err))
    else :
        pass