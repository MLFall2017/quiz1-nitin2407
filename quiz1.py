import numpy as np
import matplotlib.pyplot as plt

def pca(input):
    
    #get input and skip headers
    input = np.genfromtxt(input,delimiter=",",skip_header=1) 

    #find correlation coeficient matrix
    cov_in = np.corrcoef(input.T)

    #find eigen values and vectors
    eig_val, eig_vec = np.linalg.eig(cov_in)

    #pairing of eigen values and vectors
    eig_pairs = [(np.abs(eig_val[i]),eig_vec[:,i]) for i in range(len(eig_val))]

    #sorting and finding cumulative matrix for eigen values
    eig_pairs.sort()
    eig_pairs.reverse()
    tot = sum(eig_val)
    eigval_per = [(i/tot)*100 for i in sorted(eig_val,reverse=True)]
    cum_sum = np.cumsum(eigval_per)

    #plotting
    range_x = np.array([1,50,100])
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(range_x,cum_sum)
    fig.show()

    #finding the new reduced data matrix
    matrix_w = np.hstack((eig_pairs[0][1].reshape(3,1),eig_pairs[1][1].reshape(3,1)))
    y = input.dot(matrix_w)
    return y

#getting the dataset
data = np.genfromtxt("dataset_1.csv",delimiter=",",skip_header=1)

#seperating indivisual columns
x = data.T[0]
y = data.T[1]
z = data.T[2]

#finding variance for each variable
var_x = np.var(x)
var_y = np.var(y)
var_z = np.var(z)

#finding covariance
cov_xy = np.cov(x,y)
cov_yz = np.cov(y,z)

#getting pca output
out = pca("dataset_1.csv")

