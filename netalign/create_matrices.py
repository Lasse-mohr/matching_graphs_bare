import numpy as np


class LMatrixCreater:
    
    TYPES = ['identity', 'binary', 'exponential', 'k-neighbourd']

    def __init__(self, type='binary', eps=10**(-10)):
        if not type in self.TYPES:
            raise ValueError(f"Invalid type: {type}. Must be one of {self.TYPES}")
        self.type = type
        self.eps = eps
        
    def construct_L(self, matrix_A, matrix_B):
        L = np.zeros((matrix_A.shape[0], matrix_B.shape[1]))

        if self.type == 'binary':
            for row in range(matrix_A.shape[0]):
                for col in range(matrix_B.shape[1]):
                    distance = np.linalg.norm(matrix_A[row,:] - matrix_B[:, col], axis=0, ord=2)**2 
                    print((row, col, distance))
                    if distance < self.eps:
                        L[row, col] = 1

        return L            

