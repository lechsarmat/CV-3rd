import numpy as np

def CSP2(Q, G):
    """Implements Arc-Consistency algorithm for the Sudoku problem.
       >>> CSP2(np.ones((9,9,9), dtype = int),np.zeros((9,9,9,9,9,9), dtype = int))[0][:,:,0]
       array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0]])
       >>> CSP2(0, 0)
       Traceback (most recent call last):
        ...
       TypeError: Q and G must be numpy arrays
       >>> CSP2(np.zeros((9,9,9)), np.zeros((9,9,9,9,9,9)))
       Traceback (most recent call last):
        ...
       TypeError: wrong type of elements
       >>> CSP2(np.zeros((9,9,9), dtype = int), np.zeros((9,9,9), dtype = int))
       Traceback (most recent call last):
        ...
       ValueError: wrong shape of Q or G
       >>> CSP2(np.zeros((9,9,9), dtype = int) + 2, np.zeros((9,9,9,9,9,9), dtype = int))
       Traceback (most recent call last):
        ...
       ValueError: elements of Q and G must be equal to 0 or 1
    """
    if type(Q) != np.ndarray:
        raise TypeError( "Q and G must be numpy arrays" )
    if Q.dtype != 'int' or G.dtype != 'int':
        raise TypeError( "wrong type of elements" )
    if Q.shape != (9,9,9) or G.shape != (9,9,9,9,9,9):
        raise ValueError( "wrong shape of Q or G" )
    if np.prod(Q >= 0) * np.prod(Q <= 1) == 0 or np.prod(G >= 0) * np.prod(G <= 1) == 0:
        raise ValueError( "elements of Q and G must be equal to 0 or 1" )
    X = np.mgrid[0:9,0:9,0:9,0:9,0:9,0:9]
    Y = (np.maximum(np.maximum(X[0] == X[3],X[1] == X[4]),(np.floor(X[0] / 3) == np.floor(X[3] / 3))
    * (np.floor(X[1] / 3) == np.floor(X[4] / 3))) * np.maximum(X[0] != X[3],X[1] != X[4])).astype(int)
    while True:
        NG = np.broadcast_to(Q,(9,9,9,9,9,9)) * np.moveaxis(np.broadcast_to(Q,(9,9,9,9,9,9)),[0,1,2],[3,4,5]) * G
        NQ = Q * np.min(np.where(Y[:,:,:,:,:,0] == 1, np.max(NG, axis = 5), np.ones((9,9,9,9,9), dtype = int)), axis = (3,4))
        if np.sum(np.abs(NQ - Q)) + np.sum(np.abs(NG - G)) == 0:
            break
        Q = NQ
        G = NG
    return Q, G

def SudokuPolymorphSolver(S):
    """Solves the Sudoku problem with polymorphism.
       >>> SudokuPolymorphSolver(np.zeros((9,9), dtype = int))
       array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0]])
       >>> SudokuPolymorphSolver(0)
       Traceback (most recent call last):
        ...
       TypeError: S must be numpy array
       >>> SudokuPolymorphSolver(np.zeros((9,9)))
       Traceback (most recent call last):
        ...
       TypeError: wrong type of elements
       >>> SudokuPolymorphSolver(np.zeros((4,9), dtype = int))
       Traceback (most recent call last):
        ...
       ValueError: wrong shape of S
       >>> SudokuPolymorphSolver(np.zeros((9,9), dtype = int) - 1)
       Traceback (most recent call last):
        ...
       ValueError: elements of S must be >= 0 and <= 9
    """
    if type(S) != np.ndarray:
        raise TypeError( "S must be numpy array" )
    if S.dtype != 'int':
        raise TypeError( "wrong type of elements" )
    if S.shape != (9,9):
        raise ValueError( "wrong shape of S" )
    if np.prod(S >= 0) * np.prod(S <= 9) == 0:
        raise ValueError( "elements of S must be >= 0 and <= 9" )
    X = np.mgrid[0:9,0:9,0:9,0:9,0:9,0:9]
    G = ((np.maximum(np.maximum(X[0] == X[3],X[1] == X[4]),(np.floor(X[0] / 3) == np.floor(X[3] / 3))
    * (np.floor(X[1] / 3) == np.floor(X[4] / 3))) * (X[2] != X[5])) * np.maximum(X[0] != X[3],X[1] != X[4])).astype(int)
    Q = np.array([[np.where(np.arange(1,10) != S[i][j],np.zeros((9,)),np.ones((9,)))
    if S[i][j] != 0 else np.ones((9,)) for j in range(9)] for i in range(9)]).astype(int)
    Q, G = CSP2(Q, G)
    while np.max(np.sum(Q, axis = 2)) > 1:
        I = np.unravel_index(np.argmax(np.sum(Q, axis = 2), axis = None), np.sum(Q, axis = 2).shape)
        P = np.array([Q[I]])[0]
        while True:
            Q[I] = np.zeros((9,), dtype = int)
            Q[I[0],I[1],np.argmax(P)] = 1
            NQ, NG = CSP2(Q, G)
            P[np.argmax(P)] = 0
            if (np.min(np.sum(NQ, axis = 2)) == 0 and np.max(P) == 0) or np.min(np.sum(NQ, axis = 2)) != 0:
                Q = NQ
                G = NG
                break
    return (np.argmax(Q, axis = 2) + 1).astype(int) if np.max(np.sum(Q, axis = 2)) == 1 else np.zeros((9,9), dtype = int)

if __name__ == "__main__":
    import doctest
    doctest.testmod()