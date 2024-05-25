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

def show_eigenfaces(eigenfaces, S): # eigenfaces is (4096 x num)
    rows, cols = eigenfaces.shape # rows is 4096
    fig, axes = plt.subplots(1, cols, figsize=(15, 8))
    for col in range(cols):
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

def all_weights(meaned_data, eigenfaces): # matrix for weights of ALL faces (using dot producting to find a weight) should be (410 x 4096)
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
    squared = all_diffs ** 2 # square so all entries are positive
    summed = squared.sum(axis=1) # will be a column vector

    # the smallest summed squared distance will mean the closest match
    
    i = np.argmin(summed)
    print(summed[i])
    print("index of best match face:", i)
    return all_faces[i]
    

if __name__ == "__main__":
    data = load_data("olivetti.csv")
    me = utils.convert("me.jpg")
    eigenfaces, S, meaned, all_meaned = find_eigenfaces(data, 20)
    # show_eigenfaces(eigenfaces, S)
    w = one_weight(me-meaned, eigenfaces)
    r = reconstruct_one(w, eigenfaces, meaned)
    all_w = all_weights(all_meaned, eigenfaces)
    
    # print(data[0].shape)
    best_face = best_match(me, eigenfaces, all_w, data)
    show_face(best_face, "best match")
    # show_face(r)