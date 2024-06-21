import json
import numpy as np   # check out how to install numpy
from collections import Counter
from utils import load, plot_sample

# =========================================
#       Homework on K-Nearest Neighbors
# =========================================
# Course: Introduction to Information Theory
# Lecturer: Haim H. Permuter.
#
# NOTE:
# -----
# Please change the variable ID below to your ID number as a string.
# Please do it now and save this file before doing the assignment

ID = '316389584'

# Loading and plot a sample of the data
# ---------------------------------------
# The MNIST database contains in total of 60000 train images and 10000 test images
# of handwritten digits.
# This database is a benchmark for many classification algorithms, including neural networks.
# For further reading, visit http://yann.lecun.com/exdb/mnist/
#
# You will implement the KNN algorithm to classify two digits from the database: 3 and 5.
# First, we will load the data from .MAT file we have prepared for this assignment: MNIST_3_and_5.mat

Xtrain, Ytrain, Xvalid, Yvalid, Xtest = load('MNIST_3_and_5.mat')

# The data is divided into 2 pairs:
# (Xtrain, Ytrain) , (Xvalid, Yvalid)
# In addition, you have unlabeled test sample in Xtest.
#

# Each row of a X matrix is a sample (gray-scale picture) of dimension 784 = 28^2,
# and the digit (number) is in the corresponding row of the Y vector.
#
# To plot the digit, see attached function 'plot_sample.py'

sampleNum = 0
plot_sample(Xvalid[sampleNum, :], Yvalid[sampleNum, :])

# Build a KNN classifier based on what you've learned in class:
#
# 1. The classifier should be given a train dataset (Xtrain, Ytain),  and a test sample Xtest.
# 2. The classifier should return a class for each row in test sample in a column vector Ytest.
#
# Finally, your results will be saved into a <ID>.txt file, where each row is an element in Ytest.
#
# Note:
# ------
# For you conveniece (and ours), give you classifications in a 1902 x 1 vector named Ytest,
# and set the variable ID at the beginning of this file to your ID.

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2)) # distance function
    return distance

def predict(x, k):
    # function we use to guess whether it's 3 or 5
    distances = [euclidean_distance(x, x_train) for x_train in Xtrain]
    k_indices = np.argsort(distances)[:k]   # Returns the indices that would sort an array.
    results = []
    for i in range(0,k):
        results.append(Ytrain[k_indices[i]])
    if results.count(3) > results.count(5):
        prediction = 3
    else:
        prediction = 5
    return prediction

def validate():
    # finding optimal k using Xvalid and Yvalid
    precisions = []
    for k in range(1, 21):
        precise = 0
        for i in range(len(Xvalid)):
            s1 = predict(Xvalid[i], k) # "question vector"
            s2 = int(Yvalid[i]) # "answer vector"
            print("Test number:",i+1)
            print("prediction:",s1,",real number:",s2)
            if s1 == s2: # checking validation
                precise = precise +1
                print(precise,"/",i+1)
        precisions.append(precise)
    print("k optimal is=", precisions.index(max(precisions))+1)


#validate()

def test(Xtest):
    # after training we now predict Ytest with Xtest
    np.loadtxt('316389584.txt')
    with open('316389584.txt','w')as f:
        for i in range(len(Xtest)): # testing new vector without "answers", predicting the answers
            Ytest[i] = predict(Xtest[i], 17) # while running validate from 1 to 20 we found out k=17 is optimal
            f.write(str(Ytest[i])+'\n')
            print("Test number:",i+1)
            print("prediction:",Ytest[i])
        np.savetxt(f'{ID}.txt', Ytest, delimiter=", ", fmt='%i')
        return Ytest

# < your code here >


# Example submission array - comment/delete it before submitting:
Ytest = np.arange(0, Xtest.shape[0])

# save classification results
print('saving')
np.savetxt(f'{ID}.txt', Ytest, delimiter=", ", fmt='%i')
print('done')

test(Xtest)
