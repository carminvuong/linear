import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
import utils

def load_data(filename): # filename is a string with the filename
    raw_data = np.genfromtxt(filename, delimiter=',')
    data = np.delete(raw_data, 0, axis=0)
    data = np.delete(data, -1, axis=1)
    # print("data shape:", data.shape)
    return data
    

def find_eigenfaces(data, num): # finding first num eigenvalues based on SVD
    # finding mean face
    mean_face = np.average(data, axis=0)

    # mean centering the data
    meaned = data - mean_face # now we can use this to do SVD and stuff

    # linearalgebra
    U, S, V = np.linalg.svd(meaned.T)
    # S are the singular values in decreasing order and U are the eigenfaces
    # print(U.shape)
    return U[:, 0:num], S, mean_face, meaned # every row, 0 - num columns AND the singular values AND mean_face

def show_eigenfaces(eigenfaces, S): # eigenfaces is (4096 x num) # shows first 10
    rows, cols = eigenfaces.shape # rows is 4096
    fig, axes = plt.subplots(1, cols, figsize=(15, 8))
    for col in range(10):
        ax = axes[col]
        ef = eigenfaces[:, col].reshape(64, 64)
        ax.imshow(ef, cmap="gray")
        ax.axis("off")
        ax.set_title(str(int(S[col])))

    plt.show()

def show_face(face, label): # eigenfaces is (4096 x num)
    fig, axes = plt.subplots(1, 1, figsize=(10, 4))
    ax = axes
    ef = face.reshape(64, 64)
    ax.imshow(ef, cmap="gray")
    ax.axis("off")
    ax.set_title(label)

    plt.show()

def all_weights(meaned_data, eigenfaces): # weights matrix of ALL faces (using dot product to find a weight) should be (410 x 4096)
    return np.matmul(meaned_data, eigenfaces)
    # (410 x 4096) * (4096 * num) = (410 * num) = 410 row vector with num entries cuz its 1 for each eigenface

def one_weight(face_data, eigenfaces):
    return np.matmul(face_data, eigenfaces) # will return row vecotr with 1 scalar per eigenvalue (total is num)

def reconstruct_one(weights, eigenfaces, meaned):
    # print((np.matmul(weights, eigenfaces.T) + meaned).shape)
    # print(weights.shape)
    # print(eigenfaces.shape)
    # print(meaned.shape)
    return np.matmul(weights.T, eigenfaces.T) + meaned

def best_match(face, eigenfaces, all_weights, all_faces):
    mean_face = np.average(all_faces, axis=0)
    self_weight = one_weight(face-mean_face, eigenfaces)
    all_diffs = all_weights - self_weight
    squared = all_diffs ** 2 # squared so all entries are positive

    summed = squared.sum(axis=1) # will be a column vector

    # the smallest summed squared distance will mean the closest match
    i = np.argmin(summed)
    print("sum of squares: ", summed[i])
    return all_faces[i]
    

if __name__ == "__main__":
    # olivetti_clone.csv is a CSV file with all the data
    # "data" is the matrix with all the data on it
    data = load_data("olivetti_clone.csv")

    # converts an image into a vector
    me = utils.convert("test_me.jpg") 

    # in this case, "eigenfaces" is a (4096x200) matrix containing 200 eigenfaces
    # "S" are the singular values as a list
    # "meaned" is the mean face 
    # "all_meaned" is the mean-centered data
    eigenfaces, S, meaned, all_meaned = find_eigenfaces(data, 200)

    # "all_w" is the weights matrix
    all_w = all_weights(all_meaned, eigenfaces)
    
    # best_face is the face that is "closest" to our testing image
    best_face = best_match(me, eigenfaces, all_w, data)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    ax = axes[0]
    face = me.reshape(64, 64)
    ax.imshow(face, cmap='gray')
    ax.axis('off')
    ax.set_title(f'me')

    ax = axes[1]
    face = best_face.reshape(64, 64)
    ax.imshow(face, cmap='gray')
    ax.axis('off')
    ax.set_title(f'best match')

    plt.show()