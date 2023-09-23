from matrix import Matrix

custom = [[1, 1, 1, -1], # x = -1
          [1, 2, 4, 3],  # y = 3
          [1, 3, 9, 3]]  # z = 3

m = Matrix(custom) # hard coded matrix for testing first

# ----- TESTING -----
# m.swap(1, 3)
# m.add(1, 5, 2) # adds 5x of row1 to row2, so row2 changes
# m.scale(2, 10) # scales row2 by 10

m.display()