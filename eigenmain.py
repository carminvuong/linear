import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im

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
    return U[:, 0:num], S, mean_face # every row, 0 - num columns AND the singular values AND mean_face

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

def show_face(face): # eigenfaces is (4096 x num)
    fig, axes = plt.subplots(1, 1, figsize=(10, 4))
    ax = axes
    ef = face.reshape(64, 64)
    ax.imshow(ef, cmap="gray")
    ax.axis("off")
    ax.set_title("face")

    plt.show()

def all_weights(meaned_data, eigenfaces): # matrix for weights of ALL faces (using dot producting to find a weight) should be (410 x 4096)
    return np.matmul(meaned_data, eigenfaces)
    # (410 x 4096) * (4096 * num) = (410 * num) = 410 row vector with num entries cuz its 1 for each eigenface

def one_weight(face_data, eigenfaces):
    return np.matmul(face_data, eigenfaces) # will return row vecotr with 1 scalar per eigenvalue (total is num)

def reconstruct_one(weights, eigenfaces, meaned):
    # print((np.matmul(weights, eigenfaces.T) + meaned).shape)
    print(weights.shape)
    print(eigenfaces.shape)
    print(meaned.shape)
    return np.matmul(weights.T, eigenfaces.T) + meaned

if __name__ == "__main__":
    data = load_data("olivetti.csv")
    eigenfaces, S, meaned = find_eigenfaces(data, 100)
    # show_eigenfaces(eigenfaces, S)
    w = one_weight(data[0], eigenfaces)
    r = reconstruct_one(w, eigenfaces, meaned)
    show_face(r)