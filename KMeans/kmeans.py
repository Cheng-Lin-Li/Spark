#!/usr/bin/env python
# encoding: utf-8
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
The K-means algorithm written from scratch against PySpark. In practice,
one may prefer to use the KMeans algorithm in ML, as shown in
examples/src/main/python/ml/kmeans_example.py.

This example requires NumPy (http://www.numpy.org/).
"""
from __future__ import print_function
from __future__ import division
import sys

import numpy as np
from scipy.sparse import csc_matrix
#from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext

SPLITTER = ' ' #Data separate symbol
DATA_INDEX_FROM = 1 #Data index from 0 or 1. example M(0,0)=2 or M(1,1)=2. The system default index is 0
UNIT_VECTOR = 'tf-idf' # Unit vector of a document. Currently only support tf-idf
DISTANCE_FUNCTION = 'cosine' # Distance function. Currently only support tf-idf

USE_UNICODE = False
DEBUG = 0 # Level:0=No log, :1=Normal, :2=Detail
PRINT_TIME = False #Enable/disable time stamp printing into result. 

INPUT_FILE = 'input.txt' #Default input file name
ORIG_STDOUT = None
#OUTPUT_FILE = 'output.txt' # OUTPUT_FILE COULD BE 'OUTPUT_FILE = None' for console or file name (e.g. 'OUTPUT_FILE = 'output.txt') for file.'
OUTPUT_FILE = None # OUTPUT_FILE COULD BE 'OUTPUT_FILE = None' for console or file name (e.g. 'OUTPUT_FILE = 'output.txt') for file.'


def getInputData(filename):
# Get data from input file. 
    _row = list()
    _col = list()
    _data = list()
    _no_docs = 0
    _counter = 0
    
    try:
        with open(filename, 'r') as _fp:
            for _each_line in _fp:
                if _counter >= 3: #skip file header
                    _r = _each_line.strip().split(SPLITTER)
                    _row.append(int(_r[0])-DATA_INDEX_FROM)
                    _col.append (int(_r[1])-DATA_INDEX_FROM)
                    _data.append(float(_r[2])) #(data=tf, indices=document id, indptr=word id)
                elif _counter == 0:
                    _no_docs = int (_each_line)
                    _counter += 1
                else:
                    _counter += 1
        _fp.close()
        if DEBUG: print ('getInputData.=>no. of documents=%d'%(_no_docs))
        if DEBUG: print ('getInputData. row = : %s'%(_row))
        if DEBUG: print ('getInputData. col = : %s'%(_col))
        if DEBUG: print ('getInputData. data = : %s'%(_data))
        return _no_docs, _row, _col, _data
    except IOError as _err:
        if DEBUG: 
            print ('File error: ' + str (_err))
        else :
            pass
        exit()

def set_std_out_2_file(filename):
    try:
        ORIG_STDOUT = sys.stdout        
        if filename != None :
            f = file(filename, 'w')
            sys.stdout = f
        else:
            pass    
    except IOError as _err:
        if (DEBUG == True): 
            print ('File error: ' + str (_err))
        else :
            pass
        exit()
        
def restore_std_out():
    try:
        sys.stdout.flush()
        sys.stdout.close()
        sys.stdout = ORIG_STDOUT                         
    except IOError as _err:
        if (DEBUG == True): 
            print ('File error: ' + str (_err))
        else :
            pass
        exit()
    
def setOutputData(filename='', kPoints=list()):
# output results. 
    try:
        if filename != None :
            orig_stdout = sys.stdout
            f = file(filename, 'w')
            sys.stdout = f
        else:
            pass
##########  
# Customized output format here

        for centroid in kPoints:
            print (np.count_nonzero(centroid.todense()))
            if DEBUG: print("Final centers: " + str(kPoints))

###########
        sys.stdout.flush()       
        if filename != None :
            sys.stdout = orig_stdout                   
            f.close()
        else:
            pass        
    except IOError as _err:
        if (DEBUG == True): 
            print ('File error: ' + str (_err))
        else :
            pass
        exit()

def parseVector(line):
    return np.array([float(x) for x in line.split('')])


def closestPoint(p, centers):
    bestIndex = 0
    closest = float("+inf")
    for i in range(len(centers)):
#        tempDist = np.sum((p - centers[i]) ** 2)
        tempDist = -1*cosine_similarity (p, centers[i])        
        if tempDist < closest:
            closest = tempDist
            bestIndex = i
    return bestIndex

def cosine_similarity(vector1, vector2):
    _v1_sqrt = vector1.power(2)
    _v2_sqrt = vector2.power(2)
    _v1 = vector1.todense()
    _v2 = vector2.todense()
    if DEBUG > 1: print('cosine=>v1=%s, v2=%s, cosine()=%s'%(str(vector1), str(vector2), _v1.dot(_v2.transpose())/(np.sqrt(_v1_sqrt.sum())*np.sqrt(_v2_sqrt.sum())))) 
    return (_v1.dot(_v2.transpose())/(np.sqrt(_v1_sqrt.sum())*np.sqrt(_v2_sqrt.sum())))
  
def tf_idf(data_csc_matrix):
    _tf_idf = list()
    _pre_ipt = 0
    (_no_docs, _no_words) = data_csc_matrix.shape    
    if DEBUG: print ('_no_docs = %s'%(_no_docs))
    if DEBUG: print('tf_idf=> input data_vectors=%s'%(data_csc_matrix.todense()))
    data_csc_matrix.eliminate_zeros() #Remove zeros from the matrix
    df_array = data_csc_matrix.getnnz(axis=0)    # Number of stored values, including explicit zeros.
    if DEBUG: print('tf_idf=> df is the number of documents where the word appears., df_array=%s'%(str(df_array)))
    
    # Calculate weighting=tf*idf vector for each document.
    # This is column/word based calculation.
    _data = data_csc_matrix.data
    _indices = data_csc_matrix.indices
    _indptr = data_csc_matrix.indptr
    if DEBUG: print('tf_idf=> _data=%s'%(str(_data)))
    if DEBUG: print('tf_idf=> _indices=%s'%(str(_indices)))
    for _i in range(len(_indptr)-1): # Calculate from each column
        _pre_ipt = _indptr[_i] # previous ipt (index pointer of the CSC matrix)
        _ipt = _indptr[_i+1]
        if _ipt == _pre_ipt:
            pass # Escape the column which is no element.
        else:
            for _j in range(_pre_ipt, _ipt):
                _data[_j] = _data[_j]*np.log2((_no_docs+1)/(df_array[_i]+1))
                if DEBUG > 1: print('_j=%s, _no_docs+1=%d, df=%f'%(_j, _no_docs+1, df_array[_i]+1))
        if DEBUG > 1: print('np.log2((_no_docs+1)/(df_array[_i]+1))=%f'%(np.log2((_no_docs+1)/(df_array[_i]+1))))
        if DEBUG > 1: print('_data=%s'%(_data))
    
    data_csc_matrix.data = _data # For easy understandable purpose, actually not necessary to re-assign it again because we operate all data based on the _data pointer.
    if DEBUG: print('tf_idf=>data_csc_matrix.todense()=%s'%(data_csc_matrix.todense()))
    
    #
    # Normalization.
    #    Normalized (divided) by its Euclidean length.
    #    Euclidean distance for each document = sqrt(w1^2+w2^2+...)
    data_csc_matrix_sqrt = data_csc_matrix.power(2)
    data_csr_matrix = data_csc_matrix.tocsr() # Transfer to CSR for normalization
    _data = data_csr_matrix.data
    _indices = data_csr_matrix.indices
    _indptr = data_csr_matrix.indptr   
    if DEBUG: print('tf_idf=>data_csr_matrix.data=%s'%(data_csr_matrix.data))     
    for _i in range(len(_indptr)-1): # Calculate from each row
        _pre_ipt = _indptr[_i] # previous ipt (index pointer of the CSR matrix)
        _ipt = _indptr[_i+1]
        _ed = np.sqrt(data_csc_matrix_sqrt.getrow(_i).sum()) #Calculate Euclidean distance
        if _ipt == _pre_ipt:
            pass# Escape the Row which is no element.
        else:
            for _j in range(_pre_ipt, _ipt):
                if DEBUG > 1: print ('tf_idf=>_data[_j]=%f, _ed=%f, _data[_j]/_ed=%f'%(_data[_j], _ed, _data[_j]/_ed))
                _data[_j] = _data[_j]/_ed
    data_csr_matrix.data = _data
    data_csc_matrix = data_csr_matrix.tocsc()
    if DEBUG: print('tf_idf=>Normalization: data_csc_matrix.todense()=%s'%(data_csc_matrix.todense()))
    if DEBUG > 1: print('tf_idf=>Normalization: data_csr_matrix.data=%s'%(str(data_csr_matrix.data)))
    
    return data_csc_matrix

def set_doclist (data_csc_matrix):
    _doclist = list()
    (_no_docs, _no_words) = data_csc_matrix.shape
    for _i in range(_no_docs):
        _doclist.append((data_csc_matrix.getrow(_i)))
    return _doclist

if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("Usage: kmeans <file> <k> <convergeDist> [outputfile.txt]", file=sys.stderr)
        exit(-1)

    print("""WARN: This is a naive implementation of KMeans Clustering and is given
       as an example! Please refer to examples/src/main/python/ml/kmeans_example.py for an
       example on how to use ML's KMeans implementation.""", file=sys.stderr)

#     spark = SparkSession\
#         .builder\
#         .appName("PythonKMeans")\
#         .getOrCreate()
    
    # Create a configuration for this job
    conf = SparkConf().setAppName("PythonKMeans")

    # Create a context for the job.
    sc = SparkContext(conf=conf)        
    input_file = sys.argv[1]
    _no_of_docs, _row, _col, _data = getInputData(input_file)
#     _document_matrix = np.array(tf_idf(csc_matrix((_data, (_row, _col)), dtype=np.float64)).todense())
    _document_matrix = (tf_idf(csc_matrix((_data, (_row, _col)), dtype=np.float64)))
    _i = 0
    _document_list = set_doclist(_document_matrix)
    if DEBUG: print('_document_matrix=%s'%(str((_document_matrix))))
    if DEBUG: print('_document_list=%s'%(str((_document_list))))
    data = sc.parallelize((_document_list)).cache()
#     lines = spark.read.text(sys.argv[1]).rdd.map(lambda r: r[0])
    if DEBUG: print('data=%s'%(str(data.collect())))    
    K = int(sys.argv[2])
    convergeDist = float(sys.argv[3])
    if len(sys.argv) == 5:
        outputfile = sys.argv[4]
    else:
        outputfile = None
        
    if DEBUG: print('takeSample')
    kPoints = data.repartition(1).takeSample(False, K, 1)
    if DEBUG: print ('Initial centroid KPoints=%s'%(kPoints))
    tempDist = 1.0

    while tempDist > convergeDist:
        closest = data.map(
            lambda p: (closestPoint(p, kPoints), (p, 1)))
        if DEBUG: print('closest.collect()=%s'%(closest.collect()))
        pointStats = closest.reduceByKey(
            lambda p1_c1, p2_c2: (p1_c1[0] + p2_c2[0], p1_c1[1] + p2_c2[1]))
        if DEBUG: print('pointStats.collect()=%s'%(pointStats.collect()))
        newPoints = pointStats.map(
            lambda st: (st[0], st[1][0] / st[1][1])).collect()
        if DEBUG: print('pointStats.collect()=%s'%(pointStats.collect()))
#        tempDist = (sum(((kPoints[iK] - p).power(2)).sum()) for (iK, p) in newPoints)
        _sum = 0
        for (iK, p) in newPoints:
            _sum += ((kPoints[iK] - p).power(2)).sum()
            if DEBUG>1: print('_sum=%s'%(tempDist))
        tempDist = _sum
        
        if DEBUG: print('tempDist=%s'%(str(tempDist)))
        
        for (iK, p) in newPoints:
            kPoints[iK] = p

    setOutputData(outputfile, kPoints)

    sc.stop()
