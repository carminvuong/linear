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
    return U[:, 0:num], S # every row, 0 - num columns AND the singular values

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

if __name__ == "__main__":
    data = load_data("olivetti.csv")
    eigenfaces, S = find_eigenfaces(data, 15)
    show_eigenfaces(eigenfaces, S)