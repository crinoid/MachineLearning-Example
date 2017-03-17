#------------------------------------------------------------
# This file is to load data, shuffle & normalize & classify
#------------------------------------------------------------
from sklearn import preprocessing
import pickle
import numpy as np
import random

def getData():
    with open('forest_data.pickle', 'rb') as f:
        data = pickle.load(f)  # data features, multiple columns
        target = pickle.load(f)  # target, one column
    
    n = data.shape[0]  # get data size
#----------------------------------------------  
#     find missing values  
#----------------------------------------------      
    print "missing values:"
    for i in range(n):
        for j in range(54):
            if data[i, j] == "":
                print "missing value index:"
                print i, j         
        
#----------------------------------------------  
#     save random sequence to order.txt   
#----------------------------------------------  
#     order = range(n)
#     random.shuffle(order)
#     f=open("order.txt","w")
#     f.write(str(order)[1:len(str(order))-1])
#     f.close()

#----------------------------------------------  
#     read the random sequence from order.txt
#----------------------------------------------  
    f = open("order.txt", "r")
    order = (f.read().split(","))
      
    data = data[order]
    target = target[order]
    
#----------------------------------------------  
#     data normalization (only first 10 columns)
#----------------------------------------------  
    arr_norm = preprocessing.normalize(data[:, 0:10])
    norm_data = np.concatenate((arr_norm, data[:, 10:]), axis=1)

#----------------------------------------------  
#     divide data into train(80%) and test(20%)
#----------------------------------------------      

    index = int(0.80 * n)
    train_data = norm_data[:index]
    train_target = target[:index]
    test_data = norm_data[index:]
    test_target = target[index:]
    
    print "Finish data loading."
    
    return train_data, train_target, test_data, test_target
