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
    # data = load_data("olivetti.csv")
    # me = utils.convert("me.jpg")
    # eigenfaces, S, meaned, all_meaned = find_eigenfaces(data, 10)
    # # show_eigenfaces(eigenfaces, S)

    # w = one_weight(me-meaned, eigenfaces)
    # r = reconstruct_one(w, eigenfaces, meaned)
    # # all_w = all_weights(all_meaned, eigenfaces)
    
    # print(w)
    # # print(data[0].shape)
    # # best_face = best_match(me, eigenfaces, all_w, data)
    # # show_face(best_face, "best match")
    # show_face(r, "20")

    # putting my face in the dataset
    # showiong my faces
    # fig, axes = plt.subplots(1, 10, figsize=(15, 8))
    # for i in range(1, 11):
    #     me = utils.convert(str(i)+".jpg")
    #     ax = axes[i-1]
    #     face = me.reshape(64, 64)
    #     ax.imshow(face, cmap='gray')
    #     ax.axis('off')
    #     ax.set_title(f'me {i}')
    

    # plt.show()

    # working with new dataset
    data = load_data("olivetti_clone.csv")
    # print(data.shape)

    eigenfaces, S, mean_face, all_meaned = find_eigenfaces(data, 300)
    # show_eigenfaces(eigenfaces, S) # shows eigenfaces along with their singular values

    # let's find the weights for one of my faces
    me = utils.convert("test_3.jpg")
    # show_face(me, "me")
    weights = one_weight(me, eigenfaces)
    reconstructed = reconstruct_one(weights, eigenfaces, mean_face)
    # show_face(reconstructed, "reconstructed me")

    all_ws = all_weights(all_meaned, eigenfaces)
    best_face = best_match(me, eigenfaces, all_ws, data)

    show_face(best_face, "best approximation")
