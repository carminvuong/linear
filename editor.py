import csv
import numpy as np
import utils
from eigenmain import show_face


def load_data(filename): # filename is a string with the filename
    raw_data = np.genfromtxt(filename, delimiter=',')
    data = np.delete(raw_data, 0, axis=0)
    data = np.delete(data, -1, axis=1)
    # print("data shape:", data.shape)
    return data

# data = list(load_data("olivetti_clone.csv"))
# print(len(data))
def add_face(face_file, dataset):
    vector = list(utils.convert(face_file))
    vector.append("stub")

    with open(dataset, "a+", newline="\n") as file:
        writer = csv.writer(file)

        writer.writerow(vector)
    
    # print(data.shape)

if __name__ == "__main__":
    data = load_data("olivetti_clone.csv")
    print(data.shape)

    # for i in range(2, 11):
    #     add_face(f"formatted_{i}.jpg", "olivetti_clone.csv")

    # add_face(f"formatted_{1}.jpg", "olivetti_clone.csv")
    show_face(data[420], "added")