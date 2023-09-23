class Matrix:

    def __init__(self, matrix):
        self.m = matrix

    def display(self):
        for row in self.m:
            print(row)
    
    def add(self,  row1, constant1, row2): # adds row1 * constant1 to row2 (row2 changes)
        newRow = []
        for i in range(len(self.m[row1-1])): # assumes each row has same number of entries
            newRow.append(self.m[row1-1][i]*constant1 + self.m[row2-1][i])
        self.m[row2-1] = newRow
    
    def swap(self, row1, row2): # interchanges row1 and row2
        temp = self.m[row1-1]
        self.m[row1-1] = self.m[row2-1]
        self.m[row2-1] = temp

    def scale(self, row, constant): # scales row by constant
        for i in range(len(self.m[row])):
            self.m[row-1][i] *= constant

