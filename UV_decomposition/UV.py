#!/usr/bin/env python
# encoding: utf-8
'''
UV decomposition algorithm -- An implementation of UV decomposition algorithm.

The implementation goal is to decompose user/product ratings matrix M into lower-rank matrices U and V such that the difference between M and UV is minimized.
Root-mean-square error (RMSE) is adopted to measure the quality of decomposition.

It defines classes_and_methods below:

UV decomposition algorithm: UV decomposition algorithm implementation class in Python.
    Calculate the similarity by :
    
    Major Functions:
    
@author: Cheng-Lin Li a.k.a. Clark Li

@copyright:    2017 Cheng-Lin Li@University of Southern California. All rights reserved.

@license:    Licensed under the GNU v3.0. https://www.gnu.org/licenses/gpl.html

@contact:    chenglil@usc.edu
@version:    1.0

@create:    March, 13, 2017
@updated:   March, 24, 2017
'''

from __future__ import print_function 
from __future__ import division
# Import the necessary Spark library classes, as well as sys
#from pyspark import SparkConf, SparkContext, StorageLevel
#import collections
#from itertools import combinations
from datetime import datetime
import sys
import math
import numpy as np
from numpy import matrix

__all__ = []
__version__ = 1.0
__date__ = '2017-03-13'
__updated__ = '2017-03-24'

APP_NAME = 'UV_Decomposition'
SPLITTER = ',' #Data separate symbol
DATA_INDEX_FROM = 1 #Data index from 0 or 1. example M(0,0)=2 or M(1,1)=2. The system default index is 0
M_BLANK_VALUE = 0 # Blank value which will fill into original matrix to replace with no data
U_INITIAL_VALUE = 1.0 #Initial value of U matrix has to be float
V_INITIAL_VALUE = 1.0 #Initial value of V matrix has to be float

USE_UNICODE = False
DEBUG = 0 # Level:0=No log, :1=Normal, :2=Detail
PRINT_TIME = False #Enable/disable time stamp printing into result. 
PRINT_RMSE = True #Enable/disable RMSE printing in each iteration into result.
PRINT_UV = False #Enable/disable U, V matrix printing into result

INPUT_FILE = 'mat.dat' #Default input file name
ORIG_STDOUT = None
#OUTPUT_FILE = 'output.txt' # OUTPUT_FILE COULD BE 'OUTPUT_FILE = None' for console or file name (e.g. 'OUTPUT_FILE = 'output.txt') for file.'
OUTPUT_FILE = None # OUTPUT_FILE COULD BE 'OUTPUT_FILE = None' for console or file name (e.g. 'OUTPUT_FILE = 'output.txt') for file.'


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

def set_std_out_2_file(filename):
    try:
        ORIG_STDOUT = sys.stdout        
        if filename != None :
            f = open(filename, 'w')
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
    
def setOutputData(filename='', u=None, v=None):
# output results. 
    try:
        if filename != None :
            orig_stdout = sys.stdout
            f = file(filename, 'w')
            sys.stdout = f
        else:
            pass
##########  
        print('u = %s'%(u))
        print('v = %s'%(v))

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

class UV_Decomposition(object):
    '''
    This class implements UV Decomposition algorithm.
            The learning starts with learning elements in U row by row, i.e., U[1,1], U[1,2], …, U[2, 1], 
            It then moves on to learn elements in V column by column, i.e., V[1,1], V[2,1], …, V[1, 2],
            When learning an element, it uses the latest value learned for all other elements. 
            It should compute the optimal value for the element to minimize the current RMSE.
            
        – U: d-dimensional row (user) vectors 
        – V: d-dimensional column (item) vectors            
    '''   
    hash_func = None

    def __init__(self, n, m, dimensions, iterations, print_rmse=PRINT_RMSE):
        '''
        Constructor
            - n is the number of rows (users) of the matrix.
            - m is the number of columns (products).
            - dimensions is the number of dimensions/factors in the factor model. That is, U is n-by-f matrix, while V is f-by-m matrix.
            - iterations is the number of iterations.
        '''
        self.n = int(n)
        self.m = int(m)
        self.dim = int(dimensions)
        self.iter = int(iterations)
        self.U = np.full((n, dimensions), float(U_INITIAL_VALUE))
        self.V = np.full((dimensions, m), float(V_INITIAL_VALUE))
        self.M = np.full((n, m), M_BLANK_VALUE)
        self.P = np.dot(self.U, self.V)
        self.RSME = 0.0
        self.print_rmse = print_rmse        
        if DEBUG: print ('Initial=>self.U=%s,\n self.V=%s,\n self.M=%s,\n self.P=%s'%(self.U, self.V, self.M, self.P))
        
    def execution(self, data_list = None):
        _mx = self.set_data_matrix(data_list)
        
        for i in range(self.iter):
            if DEBUG > 1: print ('iteration: %d'%(i))
            self.set_adjusted_U()
            self.set_adjusted_V()
            self.get_RSME()
            if PRINT_RMSE : print ('%.4f'%(self.RSME))
        return self.U, self.V


    def set_adjusted_U(self):
        _m = 0.0
        _numerator = 0.0
        _denominator = 0.0
        _sum_uv = 0.0
        _all_zero = True
        
        for r in range(self.n): #Calculate each row
            for s in range(self.dim): # Calculate each column
                _numerator = 0.0
                _denominator = 0.0 
                _all_zero = True              
                for j in range(self.m): # For the specific row in U matrix have to compute with every column in V matrix
                    _m = self.M[r, j]
                    if _m != M_BLANK_VALUE:
                        _all_zero = False
                        _sum_uv = 0.0 
                        for k in range(self.dim): #Calculate specific row elements in U and specific column elements in V   
                            if (k!=s):
                                _sum_uv += self.U[r, k]*self.V[k, j] #U1,1*V1,1+U1,2*V2,1
                            else: pass
                            
                        _numerator += self.V[s, j]*(_m-_sum_uv)
                        _denominator += math.pow(self.V[s, j], 2)
                        if DEBUG > 1: print ('get_adjusted=>_m=%f, _sum_uv=%f, _numerator=%f'%(_m, _sum_uv, _numerator))
                    else: pass
                if _all_zero != True:
                    self.U[r, s] = _numerator/_denominator
                    if DEBUG > 1: print ('get_adjusted=> _numerator/_denominator=%.4f/%.4f=%.4f'%(_numerator, _denominator, _numerator/_denominator))
                    if DEBUG > 1: print ('get_adjusted_U=%s'%(self.U))
                else:
                    if DEBUG: print ('All elements are zero in M[%d:]'%(r))
        if DEBUG: print ('get_adjusted_U=%s'%(self.U))
        return self.U

    def set_adjusted_V(self):
        _m = 0.0
        _numerator = 0.0
        _denominator = 0.0
        _sum_uv = 0.0
        _all_zero = True
        
        for s in range(self.m):
            for r in range(self.dim):
                _numerator = 0.0
                _denominator = 0.0
                _all_zero = True                 
                for i in range(self.n):
                    _m = self.M[i, s]
                    if _m != M_BLANK_VALUE:
                        _all_zero = False 
                        _sum_uv = 0.0
                        for k in range(self.dim):
                            if (k!=r):
                                _sum_uv+=self.U[i, k]*self.V[k, s]
                            else: pass
                            
                        _numerator += self.U[i, r]*(_m-_sum_uv)
                        _denominator += math.pow(self.U[i, r], 2)
                        if DEBUG > 1: print ('get_adjusted=>_m=%f, _sum_uv=%f, _numerator=%f'%(_m, _sum_uv, _numerator))
                    else: pass
                if _all_zero != True:
                    self.V[r, s] = _numerator/_denominator
                    if DEBUG > 1: print ('get_adjusted=> _numerator/_denominator=%.4f/%.4f=%.4f'%(_numerator, _denominator, _numerator/_denominator))
                    if DEBUG > 1: print ('get_adjusted_V=%s'%(self.V))
                else:
                    if DEBUG: print ('All elements are zero in M[:%d]'%(s))
        if DEBUG: print ('get_adjusted_V=%s'%(self.V))
        return self.V

    def get_RSME(self):
        _k = 0
        _sum = 0.0
        _mp = np.dot(self.U, self.V)
        for i in range(self.n):
            for j in range(self.m):
                _m = self.M[i, j]
                if (_m != M_BLANK_VALUE):
                    _sum += math.pow(_m-_mp[i, j],2)
                    _k += 1
                else: pass
        self.RSME = math.sqrt(_sum/_k)
        
        if DEBUG: print ('get_adjusted_RSME=%.4f'%(self.RSME))
        return self.RSME    

    def set_data_matrix(self, data_list = None):
        # input data tuples list of a matrix, data format [(r1, c1, data1), (r2, c2, data2)...]
        # output 2d array results. 
        for data in data_list:
            self.M[data[0], data[1]] = data[2]
        if DEBUG: print ('get_DataMatrix=>self.M=%s'%(self.M))
        return self.M
              
if __name__ == "__main__":
    '''
        Main program.
    '''
    output_file = ''
    
    if len(sys.argv) < 5 : 
        print('Usage of UV_Decomposition: %s, matrix.dat n m f k [output.txt]'%(sys.argv[0]))
        print('    n is the number of rows (users) of the matrix.')
        print('    m is the number of columns (products).')
        print('    f is the number of dimensions/factors in the factor model.')
        print('     - That is, U is n-by-f matrix, while V is f-by-m matrix.')
        print('    k is the number of iterations.')
        print('    output.txt is the output file. It is an option parameter')
    else:
        input_file = sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        m = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        dimensions = int(sys.argv[4]) if len(sys.argv) > 4 else 0       
        iterations = int(sys.argv[5]) if len(sys.argv) > 5 else 0
        output_file = sys.argv[6] if len(sys.argv) > 6 else OUTPUT_FILE
    if PRINT_TIME : print ('UV_Decomposition=>Start=>%s'%(str(datetime.now())))   
    
    if output_file != None: set_std_out_2_file(output_file)
    _U = None
    _V = None
    _print_rmse = PRINT_RMSE
    # Initial UV Decomposition object and read input file
    _data_list = getInputData(input_file)
    
    uv = UV_Decomposition(n, m, dimensions, iterations, _print_rmse)
    _U, _V = uv.execution(_data_list)

    if PRINT_UV: setOutputData(output_file, _U, _V)
    if output_file != None: restore_std_out()
    if PRINT_TIME : print ('UV_Decomposition=>Finish=>%s'%(str(datetime.now())))   
    
