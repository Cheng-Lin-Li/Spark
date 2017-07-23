#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
This is an example implementation of ALS for learning how to use Spark. Please refer to
pyspark.ml.recommendation.ALS for more conventional use.

This example requires numpy (http://www.numpy.org/)
"""
from __future__ import print_function

import sys

import numpy as np
from numpy.random import rand
from numpy import matrix
from pyspark.sql import SparkSession

LAMBDA = 0.01   # regularization
np.random.seed(42)

SPLITTER = ',' #Data separate symbol
DATA_INDEX_FROM = 1 #Data index from 0 or 1. example M(0,0)=2 or M(1,1)=2. The system default index is 0
M_BLANK_VALUE = 0 # Blank value which will fill into original matrix to replace with no data
U_INITIAL_VALUE = 1.0 #Initial value of U matrix has to be float
V_INITIAL_VALUE = 1.0 #Initial value of V matrix has to be float

ORIG_STDOUT = None
INPUT_FILE = 'mat.dat' #Default input file name
#OUTPUT_FILE = 'output.txt' # OUTPUT_FILE COULD BE 'OUTPUT_FILE = None' for console or file name (e.g. 'OUTPUT_FILE = 'output.txt') for file.'
OUTPUT_FILE = None # OUTPUT_FILE COULD BE 'OUTPUT_FILE = None' for console or file name (e.g. 'OUTPUT_FILE = 'output.txt') for file.'
USE_UNICODE = False
DEBUG = 1 # Level:0=No log, :1=Normal, :2=Detail
PRINT_RMSE = True #Enable/disable RMSE printing in each iteration into result.

def set_std_out_2_file(filename):
    try:
        ORIG_STDOUT = sys.stdout        
        if filename != None :
            f = open(filename, 'w')
            sys.stdout = f
        else:
            pass    
    except IOError as _err:
        print ('File error: ' + str (_err))
        exit()
        
def restore_std_out():
    try:
        sys.stdout.flush()
        sys.stdout.close()
        sys.stdout = ORIG_STDOUT                         
    except IOError as _err:
        print ('File error: ' + str (_err))
        exit()
        
def getInputData(filename):
# Get data from input file. 
    _data = []

    try:
        with open(filename, 'r') as _fp:
            for _each_line in _fp:
                _row = _each_line.strip().split(SPLITTER)
                _data.append((int(_row[0])-DATA_INDEX_FROM,int(_row[1])-DATA_INDEX_FROM,int(_row[2])))
        _fp.close()
        if DEBUG: print ('getInputData = : %s'%(_data))
        return _data
    except IOError as _err:
        print ('File error: ' + str (_err))
        exit()
        
def set_data_matrix(M, data_list = None):
    for data in data_list:
        M[data[0], data[1]] = data[2]
        if DEBUG: print ('get_DataMatrix=>M=%s'%(M))
    return M

def rmse(R, ms, us):
    diff = R - ms * us.T
    return np.sqrt(np.sum(np.power(diff, 2)) / (M * U))


def update(i, mat, ratings): #i = x, mat = V if caculate U, rating is M.
    uu = mat.shape[0]
    ff = mat.shape[1]

    XtX = mat.T * mat    #projection matrix
    Xty = mat.T * ratings[i, :].T

    for j in range(ff):
        XtX[j, j] += LAMBDA * uu

    return np.linalg.solve(XtX, Xty)


if __name__ == "__main__":

    print("""WARN: This is a naive implementation of ALS and is given as an
      example. Please use pyspark.ml.recommendation.ALS for more
      conventional use.""", file=sys.stderr)

    spark = SparkSession\
        .builder\
        .appName("PythonALS")\
        .getOrCreate()

    sc = spark.sparkContext
    output_file = ''
    if len(sys.argv) < 5 : 
        print('Usage of UV_Decomposition: %s, matrix.dat n m f k p [output.txt]'%(sys.argv[0]))
        print('    n is the number of rows (users) of the matrix.')
        print('    m is the number of columns (products).')
        print('    f is the number of dimensions/factors in the factor model.')
        print('     - That is, U is n-by-f matrix, while V is f-by-m matrix.')
        print('    p is the number of dimensions/factors in the factor model.')
        print('    output.txt is the output file. It is an option parameter')
        spark.stop()
        exit()
    else:
        input_file = sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        U = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        F = int(sys.argv[4]) if len(sys.argv) > 4 else 0       
        ITERATIONS = int(sys.argv[5]) if len(sys.argv) > 5 else 0
        partitions = int(sys.argv[6]) if len(sys.argv) > 6 else 2
        output_file = sys.argv[7] if len(sys.argv) > 7 else OUTPUT_FILE
    
    if output_file != None: set_std_out_2_file(output_file)
    
    _data_list = getInputData(input_file)

#     M = int(sys.argv[1]) if len(sys.argv) > 1 else 100
#     U = int(sys.argv[2]) if len(sys.argv) > 2 else 500
#     F = int(sys.argv[3]) if len(sys.argv) > 3 else 10
#     ITERATIONS = int(sys.argv[4]) if len(sys.argv) > 4 else 5
#     partitions = int(sys.argv[5]) if len(sys.argv) > 5 else 2

#     print("Running ALS with M=%d, U=%d, F=%d, iters=%d, partitions=%d\n" %
#           (M, U, F, ITERATIONS, partitions))

    R = np.full((M, U), M_BLANK_VALUE, dtype=np.float64)
    R = matrix(set_data_matrix(R, _data_list)) #M
    ms = matrix(np.full((M, F), float(U_INITIAL_VALUE))) #U
    us = matrix(np.full((U, F), float(V_INITIAL_VALUE)))    #V

    Rb = sc.broadcast(R)
    msb = sc.broadcast(ms)
    usb = sc.broadcast(us)

    for i in range(ITERATIONS):
        ms = sc.parallelize(range(M), partitions) \
               .map(lambda x: update(x, usb.value, Rb.value)) \
               .collect() #U, x= row number
        # collect() returns a list, so array ends up being
        # a 3-d array, we take the first 2 dims for the matrix
        ms = matrix(np.array(ms)[:, :, 0]) #U
        msb = sc.broadcast(ms)

        us = sc.parallelize(range(U), partitions) \
               .map(lambda x: update(x, msb.value, Rb.value.T)) \
               .collect() #V, x = columns
        us = matrix(np.array(us)[:, :, 0])
        usb = sc.broadcast(us)

        error = rmse(R, ms, us)
        print("Iteration %d:" % i)
        print("R= %s:" % (str(R))) 
        print("ms= %s:" % (str(ms)))   
        print("us= %s:" % (str(us)))           
#         print("\nRMSE: %5.4f\n" % error)
        if PRINT_RMSE: print("RMSE=%.4f" % error)
        
    if output_file != None: restore_std_out()
    spark.stop()
