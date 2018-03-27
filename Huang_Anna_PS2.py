import numpy as np
import pandas as pd
import math
import scipy.sparse as ss
import time

#start process definitions
    
def read_data_coo_matrix(filename='links.txt'):
    "Read data file and return sparse matrix in coordinate format."
    data = pd.read_csv(filename, sep=',', header=None, encoding = 'utf-8')
    rows = data[0]  
    cols = data[1]
    vals = data[2]
    coo_matrix = ss.coo_matrix((vals, (rows, cols)))
    matrix = coo_matrix.toarray()
    return matrix
    
def clean_matrix(matrix):
    """cleans matrix and prepare for network"""
    #matrix = read_data_coo_matrix(filename='links.txt')
    #matrix = matrix_test
    #set diagonal to zero
    np.fill_diagonal(matrix,0)
    #normalize columns
    H = matrix/matrix.sum(axis=0)
    H = np.nan_to_num(H)
    return H
    
    
def find_dangling_nodes(matrix):
    """identify dangling nodes"""
    H = clean_matrix(matrix)
    d = H.sum(axis=0)
    d = [int(i==0) for i in d]
    d = np.matrix(d)
    return d
    
def get_artical_vector(matrix):
    """calculate artical vector"""
    H = clean_matrix(matrix)
    x, y = H.shape
    
    if H.shape == (6, 6):
        articles = np.array([3, 2, 5, 1, 2, 1]) # refers to matrix Z from example
    else:
        articles = np.ones(H.shape[1])
        
    a = articles/articles.sum()
    a = a.reshape(a.shape[0], 1)
    return a
    
def get_influence_vector(matrix):
    """calculate influence vector"""
    alpha = 0.85
    epsilon = 0.00001
    H = clean_matrix(matrix)
    a = get_artical_vector(matrix)
    d = find_dangling_nodes(matrix)
    #initial start vector
    pi = np.matrix([1/len(a) for i in a]).transpose()
    #Influece vector
    iteration = 0 
    
    while True:
        iteration +=1
        pi_start=pi
        pi=alpha*np.dot(H,pi) + float((alpha*np.dot(d,pi))+(1-alpha))*a
        residual=np.absolute(pi-pi_start).sum()
        if residual<epsilon:
            break
    return (H, pi, iteration)   
    
def get_ef(matrix):
    """calculate EF"""
    start_time = time.time()
    H, pi, iteration = get_influence_vector(matrix)
    EF = (100*np.dot(H,pi))/(np.dot(H,pi).sum())
    EF = pd.DataFrame(EF, columns = ["EF"])
    EF_sort = EF.sort_values(by = "EF", ascending = False)
    end_time = time.time()
    time_lapse = end_time - start_time
    return EF_sort.head(20),iteration, time_lapse


"""start test matrix result"""
#create sample matrixundefined
matrix_test = np.array([[1,0,2,0,4,3],
                [3,0,1,1,0,0],
                [2,0,4,0,1,0],
                [0,0,1,0,0,1],
                [8,0,3,0,5,2],
                [0,0,0,0,0,0]])

matrix = read_data_coo_matrix(filename='links.txt')

ef_df, iteration, runtime = get_ef(matrix_test)
print("The EF scores: ",ef_df)
print("Interation: ", iteration)
print("Runtime: ", runtime)

ef_df, iteration, runtime = get_ef(matrix)
print("The top 20 EF scores: ",ef_df)
print("Interation: ", iteration)
print("Runtime: ", runtime)

"""
Test matrix run results:

The EF scores:            
0  34.051006
4  32.916632
1  17.203742
2  12.175455
3   3.653164
5   0.000000

Interation:  18

Runtime:  0.0030031204223632812 secs
"""

"""
Links data results:

The top 20 EF scores:
8930  1.108640
725   0.247396
239   0.243818
6523  0.235173
6569  0.226118
6697  0.225255
6667  0.216701
4408  0.206480
1994  0.201435
2992  0.185031
5966  0.182744
6179  0.180768
1922  0.175082
7580  0.170443
900   0.170201
1559  0.167996
1383  0.163567
1223  0.150738
422   0.149371
5002  0.149002

Interation:  34

Runtime:  14.767848491668701 secs
"""





