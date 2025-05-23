# **2. Write a program to perform the basic matrix operations**

import numpy as np
A = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])

B= np.array([[10,11,12],[13,14,15],[16,17,18]])

print("Matrix A:\n",A,"\n")
print("Matrix B:\n",B)

print("addition:\n",A+B)
print("\nsubtraction\n",A-B)
print("\nElement wise multiplication\n",A*B)
print("\ntranspose\n",A.T)
print("\nMatrix Multiplication\n",np.dot(A,B.T))


# Batch Matrix Operations



batch_size=16
input_features=10
output_features = 5
input_batch = np.random.randint(0,101,size=(batch_size,input_features))
weights = np.random.randint(0,101,size=(input_features,output_features))
output_batch = np.dot(input_batch,weights)

print("Input batch\n",input_batch)
print("\nOutput Batch\n",output_batch)
print("\nweights\n",weights)

# Matrix Concatenation and Splitting

#Matrix concatenation (np.concatenate((A, B), axis=1)) and splitting (np.split(concatenated_matrix, 2, axis=1)) can be used in neural network architectures that involve concatenating feature maps or splitting inputs/outputs for parallel processing or multiple branches within the network.

A=np.random.randint(1,101,size=(3,4))
B=np.random.randint(1,101,size=(3,4))

concatenatedmatrix= np.concatenate((A,B),axis=1)
print("concatenated matrix\n",concatenatedmatrix)

p1,p2 = np.split(concatenatedmatrix,2,axis=1)

print("\npart1\n",p1,"\n\npart2\n",p2)


print("\nShape of matrix A:",A.shape)
print("Shape of matrix A:",A.shape)
print("Shape of concatenated matrix",concatenatedmatrix.shape)
print("Shape of matrix p1 after splitting:",p1.shape)
print("Shape of matrix p2 after splitting:",p2.shape)



#Eigen value and eigen matrix

A = np.array([[1,2],[3,4]])

rankA=np.linalg.matrix_rank(A)
eigenVal,eigenVect=np.linalg.eig(A)
A_inv = np.linalg.inv(A)

print("Rank of A:",rankA)
print("\nEigenvalue:",eigenVal)
print("\nEigenVector",eigenVect)
print("\nInverse of matrix",A_inv)
