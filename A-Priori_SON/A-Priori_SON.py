#!/usr/bin/env python
# encoding: utf-8
'''
Finding Frequent Itemsets: SON Algorithm by A-Priori algorithm in stage 1


@author: Cheng-Lin Li a.k.a. Clark Li

@copyright:    2017 Cheng-Lin Li@University of Southern California. All rights reserved.

@license:    Licensed under the GNU v3.0. https://www.gnu.org/licenses/gpl.html

@contact:    chenglil@usc.edu, clark.cl.li@gmail.com
@version:    1.1 Performance improvement on SON phase 1 & 2.

@create:    February, 23, 2017
@updated:   March, 9, 2018
'''

from __future__ import print_function 
# Import the necessary Spark library classes, as well as sys
from pyspark import SparkConf, SparkContext
from itertools import combinations
from datetime import datetime
import sys
import math

APP_NAME = 'SON'
SPLITTER = ','
CHUNKS = None
USE_UNICODE = False
DEBUG = True   
PRINT_TIME = True

class A_Priori(object):

    '''
    This class implements the A-Priori algorithm.
    All functions implement by static functions to support Spark RDD to call.
    '''   
    TTable = [] # Does not implement: Translation table {'item_name1',..., 'item_namek'] and we can get the index by [].index('item_namek')
    CTable = {} # Count table {item no.1: count 1, ..., item no. k: count k}    
    mini_sup = 0
    baskets = None
    mini_sup = 0
    max_item = 0
    freq_itemsets = [] #List of itemsets
    k_1_freq_itemsets = [] #List of itemsets
    def __init__(self, baskets=None, mini_sup=0):
        '''
        Constructor
        baskets are transactions
        min_sup = minimum support ratio (that is, for an itemset to be frequent, it should appear in at least 30% of the baskets)
        '''        
        A_Priori.baskets = baskets
        A_Priori.mini_sup = mini_sup
    
    @staticmethod
    def execute(baskets):
        _CTable = A_Priori.CTable
        _baskets = list(baskets)
        _mini_sup = A_Priori.mini_sup

        total_baskets = 0
        count_threshold = 0
        item_pairs = 1
        hasFreqItems = False
        if (_baskets is None):
            hasFreqItems = False
            return None
        else:
            hasFreqItems = True
        if PRINT_TIME : print ('A_Priori.execute=>Start=>%s'%(str(datetime.now())))   
        if DEBUG : print ('A_Priori.execute=>mini_sup=%d, baskets=%s'%(_mini_sup, str(_baskets)))
        
        #Scan the buckets to generate singleton list frequent items and count table with counts
        A_Priori.setFreqItems(_baskets, _CTable, _mini_sup, item_pairs, count_threshold)
        total_baskets = len(_baskets)
        
        count_threshold = math.ceil(total_baskets * _mini_sup)
        while hasFreqItems:
            if item_pairs > 1 : hasFreqItems= A_Priori.getFreqItems(_baskets, _CTable, _mini_sup, item_pairs, count_threshold)
            # Check frequent threshold.
            for _itemp, _counts in _CTable.items():
                if _counts >= count_threshold:
                    A_Priori.k_1_freq_itemsets.append(_itemp)
                    A_Priori.freq_itemsets.append(_itemp)
            item_pairs += 1
            _CTable = A_Priori.getCountTable(A_Priori.k_1_freq_itemsets, item_pairs) 
            A_Priori.k_1_freq_itemsets = []
        
        if DEBUG : print ('A_Priori.execute=>self.freq_itemsets=%s'%(str(A_Priori.freq_itemsets)))   
        if PRINT_TIME : print ('A_Priori.execute=>Finish=>%s'%(str(datetime.now())))         
        return iter(A_Priori.freq_itemsets)
    

    @staticmethod 
    def setFreqItems(baskets, count_table, mini_sup, item_pair_no, _count_threshold):
        # First time to build up count table base on singleton itemset by scan whole baskets.
        count_table = count_table
        if PRINT_TIME : print ('A_Priori.setFreqItems=>Start=>%s'%(str(datetime.now())))
        
        for _basket in baskets:
            for _itemset in _basket:
                _ct = count_table.get(_itemset)
                if _ct == None: #First time scan baskets, build the singleton item into count_table
                    #build the count table
                    count_table.update({_itemset:1})
                elif _ct != None:
                    count_table[_itemset] += 1  
                                                                  
        if PRINT_TIME : print ('A_Priori.setFreqItems=>Finish=>%s'%(str(datetime.now())))                    
        if DEBUG : print ('A_Priori.setFreqItems=>Time:%s, count_table==%s'%(str(datetime.now()),str(count_table)))              
        return count_table
    
    @staticmethod    
    def getFreqItems(baskets, count_table, mini_sup, item_pair_no, _count_threshold):
        #Get frequent K=item_pair_no itemsets. (6)
        count_table = count_table
        hasFreqItems = False
        if PRINT_TIME : print ('A_Priori.getFreqItems=>Start=>%s'%(str(datetime.now())))
                
        for _basket in baskets:
            
            for _itemset in count_table.keys():            
            # Remove over threshold frequent itemsets from count table 
                if (count_table[_itemset] >= _count_threshold):
                    A_Priori.k_1_freq_itemsets.append(_itemset)
                    A_Priori.freq_itemsets.append(_itemset)
                    del count_table[_itemset]                            
            for _itemset in count_table.keys():
                if set(_itemset.split(SPLITTER)).issubset(_basket):
                    hasFreqItems = True
                    count_table[_itemset] += 1

        if PRINT_TIME : print ('A_Priori.getFreqItems=>Finish=>%s'%(str(datetime.now())))                                
        if DEBUG : print ('A_Priori.getFreqItems=>Time:%s, count_table==%s'%(str(datetime.now()), str(count_table)))              
        return hasFreqItems

    @staticmethod 
    def getCountTable(fre_itemsets, item_pair_no):
        # Base on k-1 pass frequent itemsets and target number of pairs to build count_table for pass k
        itemsets = set()
        tmp_itemsets = set()
        count_table = {}
        fre_itemsets = fre_itemsets
        if PRINT_TIME : print ('A_Priori.getCountTable=>Start=>%s'%(str(datetime.now())))
        if DEBUG : print ('A_Priori.getCountTable=>item_pair_no=%d, fre_itemsets==%s'%(item_pair_no, str(fre_itemsets)))

        # Get each item from frequent itemsets and put into a set.
        for _items in fre_itemsets: 
            #if DEBUG : print ('A_Priori.getCountTable=>_items==%s'%(str(_items)))
            for _item in _items.split(SPLITTER): # Countable as baskets                   
                tmp_itemsets.add((_item))
        
        #Base on previous frequent itemsets to construct count table
        for _c in combinations(tmp_itemsets, item_pair_no):
            itemsets.add(SPLITTER.join(sorted(_c, key=int)))
        
        if DEBUG : print('fre_itemsets=%s'%fre_itemsets)
        if DEBUG : print('itemsets=%s'%itemsets)    
        for _item in itemsets:
            #if DEBUG : print ('A_Priori.getCountTable=>_item=='+str(_item))
            for _newset in combinations(_item.split(SPLITTER), item_pair_no-1): 
                #check all new candidate itemsets should be in previous itemsets
                _isExist = False
                for _exist_it in fre_itemsets:
                    #if DEBUG : print ('_exist_it=%s, SPLITTER.join(_newset)=%s _exist_it == _newset[0]=>%r'%(_exist_it, SPLITTER.join(_newset), _exist_it == SPLITTER.join(_newset)))
                    if _exist_it == SPLITTER.join(_newset):
                        _isExist = True
                        break
                #if any one of combination which generate from new candidate sets not exist in previous frequent itemsets, skip this new candidate itemsets.
                if _isExist == False:
                    break
                else:    
                    count_table.update({(_item):0})
        if PRINT_TIME : print ('A_Priori.getCountTable=>Finish=>%s'%(str(datetime.now())))        
        if DEBUG : print ('A_Priori.getCountTable=>item_pair_no=%d, count_table==%s'%(item_pair_no, str(count_table)))            
        return count_table
                
    
class SON(object):
    '''
    This class implements Savasere, Omiecinski, and Navathe (SON) algorithm.
    '''   
    def __init__(self, baskets='', mini_sup=None, chunks = CHUNKS, algorithm = 'A-Priori'):
        '''
        Constructor
        baskets are transactions
        mini_sup = minimum support ratio (that is, for an itemset to be frequent, it should appear in at least 30% of the baskets)
        algorithm = algorithm in stage one 
        '''        
        self.baskets = baskets
        self.mini_sup = mini_sup
        self.chunks = chunks
        self.total_buskets = 0
        self.algorithm = algorithm
        if (self.algorithm == 'A-Priori'):
            self.phase1_alg = A_Priori(mini_sup=self.mini_sup) 
        else:
            self.phase1_alg = None     
        self.max_item = 0
        self.local_freq_itemsets = None
        self.global_freq_itemsets = None
        self.TTable = [] # Translation table {'item_name1',..., 'item_namek'] and we can get the index by [].index('item_namek')
        self.CTable = {} # Count table {item no.1: count 1, ..., item no. k: count k}
        self.conf = SparkConf().setAppName(APP_NAME).setMaster("local[*]")
        # Create a context for the job.
        self.sc = SparkContext(conf=self.conf)
    
    def execute(self, baskets = '', mini_sup = None):
        _baskets = baskets
        _mini_sup = mini_sup
        _chunks = self.chunks
        if PRINT_TIME : print ('SON.execute=>Start=>%s'%(str(datetime.now())))                
        if _baskets == '' and self.baskets == '': return None
        elif _baskets == '' and self.baskets != '': _baskets = self.baskets

        if _mini_sup is None and self.mini_sup is None: return None
        elif (_mini_sup is None) and (self.mini_sup is not None): _mini_sup = self.mini_sup        
        
        #creating RDD from external file for Baskets.
        if (_chunks is None):
            _rddStrBaskets = self.sc.textFile(_baskets, use_unicode=USE_UNICODE) # ["1,2,4,10,14,15,16"]
        else:
            _rddStrBaskets = self.sc.textFile(_baskets, minPartitions=_chunks, use_unicode=USE_UNICODE) #['1,2,3', '1,2,5', '1,3,4',...]
        
        self.total_buskets = _rddStrBaskets.count()
        
        self.local_freq_itemsets = self.Phase1(_rddStrBaskets, _mini_sup)
        self.global_freq_itemsets = self.Phase2(_rddStrBaskets, _mini_sup, self.local_freq_itemsets)
        if PRINT_TIME : print ('SON.execute=>Finish=>%s'%(str(datetime.now())))         
        return self.global_freq_itemsets
           
    def Phase1(self, rddbaskets, mini_sup):
        mini_sup = mini_sup
        local_freq_itemsets = None
        if PRINT_TIME : print ('SON.Phase1=>Start=>%s'%(str(datetime.now())))           
        rddListBaskets = rddbaskets.map(lambda line: [ _item for _item in line.strip().split(SPLITTER)]).persist()
        if DEBUG : print ('SON.Phase1=>rddListBaskets=%s'%(str(rddListBaskets.collect())))

        local_freq_itemsets = rddListBaskets.mapPartitions(self.phase1_alg.execute).collect()        
        #mergeLocalFreqItems = self.sc.parallelize(local_freq_itemsets).map(lambda _itemset: (_itemset,1)).reduceByKey(lambda _accum, _val: _accum + _val).map(lambda (_itemset, _count): _itemset).collect()
        mergeLocalFreqItems = self.sc.parallelize(local_freq_itemsets).map(lambda _itemset: (_itemset,1)).reduceByKey(lambda _accum, _val: _val).collect()
        if DEBUG : print ('SON.Phase1=>mergeLocalFreqItems=%s'%(str(mergeLocalFreqItems)))
        if PRINT_TIME : print ('SON.Phase1=>Finish=>%s'%(str(datetime.now())))                 
        return mergeLocalFreqItems    
        
    def Phase2(self, rddbaskets, mini_sup, local_freq_itemsets):
        global_freq_itemsets = None
        count_threshold = math.ceil(self.total_buskets * mini_sup)
        
        if PRINT_TIME : print ('SON.Phase2=>Start=>%s'%(str(datetime.now())))
        if DEBUG : print ('SON.Phase2=>rddbaskets=>%s'%(rddbaskets.collect()))            
        #rddglobal_counts = rddbaskets.flatMap(lambda basket: [(_itemset[0], 1) for _itemset in local_freq_itemsets if set(str(_itemset[0]).split(SPLITTER)).issubset(basket.split(SPLITTER))])

        local_freq_itemsets = self.sc.broadcast(local_freq_itemsets) #broadcast the itemsets data to each worker in distribution environment.
        # local_freq_itemsets.value to retrieve the data list.
        rddglobal_counts = rddbaskets.mapPartitions(lambda basket: SON.getCount(basket, local_freq_itemsets)).collect()
        #rddglobal_counts = rddbaskets.mapPartitions(lambda baskets: iter([(_itemset[0], 1) for _basket in baskets for _itemset in local_freq_itemsets.value if set(str(_itemset[0]).split(SPLITTER)).issubset(_basket.split(SPLITTER))])).collect()
        
        if DEBUG : print ('SON.Phase2=>rddglobal_counts=%s'%(str(rddglobal_counts)))
        global_freq_itemsets = self.sc.parallelize(rddglobal_counts).reduceByKey(lambda _accum, _val: _accum + _val).filter(lambda (_itemset, _counts): _counts >= count_threshold).collect()
                
        if DEBUG : print ('SON.Phase2=>global_freq_itemsets=%s'%(str(global_freq_itemsets)))
        if PRINT_TIME : print ('SON.Phase2=>Finish=>%s'%(str(datetime.now())))         
        return global_freq_itemsets
         
    @staticmethod 
    def getCount(baskets, local_freq_itemsets):
        list = []
        count = 0
        for _basket in baskets: 
            for _itemset in local_freq_itemsets.value:
                count += 1 
                #print('count=%d, str(_itemset[0]).split(SPLITTER)=%s, _basket.split(SPLITTER)=%s,set(str(_itemset[0]).split(SPLITTER)).issubset(_basket.split(SPLITTER))=%r'%(count,str(_itemset[0]).split(SPLITTER), _basket.split(SPLITTER), set(str(_itemset[0]).split(SPLITTER)).issubset(_basket.split(SPLITTER))))
                if set(str(_itemset[0]).split(SPLITTER)).issubset(_basket.split(SPLITTER)) :
                    list.append((_itemset[0], 1))      
        if DEBUG : print('str(list))=%s'%(str(list)))
        return iter(list)
        
        
    
if __name__ == "__main__":

    '''
        Main program.
            Read input file
            Construct SON algorithm with input data.
            Print out results
    '''
    # Get input and output parameters
    if len(sys.argv) != 4:
        print('Usage: ' + sys.argv[0] + ' <baskets.txt> <.3> <output.txt>')
        print('<.3> = minimum support ratio (that is, for an itemset to be frequent, it should appear in at least 30% of the baskets)')
        sys.exit(1)
    
    # Assign the input and output variables
    baskets = sys.argv[1]
    mini_sup_ratio = float(sys.argv[2])
    output = sys.argv[3]
    
    son = SON(baskets, mini_sup_ratio)
    results = son.execute()


    #print the results
    try:
        if DEBUG != True :
            orig_stdout = sys.stdout
            f = file(output, 'w')
            sys.stdout = f
        else:
            pass
             
        for _pairs in results:
            print('%s'%(_pairs[0]))
     
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
         
