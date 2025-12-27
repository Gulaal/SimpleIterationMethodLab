import numpy as np

def rearrangeMatrix(A, b):
    n = A.shape[0]
    newA = A.copy()
    newB = b.copy()
    
    for i in range(n):
        max_row = i
        for j in range(i, n):
            if abs(newA[j, i]) > abs(newA[max_row, i]):
                max_row = j
        if max_row != i:
            newA[[i, max_row]] = newA[[max_row, i]]
            newB[[i, max_row]] = newB[[max_row, i]]

    print('Система после преобразования')
    print(f"Новая матрица\n {newA}")
    print(f"Новый вектор {newB}")

    return newA, newB


def transformSystem(matrix, bVector):
    matrixShape = matrix.shape
    resultMatrix = np.zeros(shape=matrixShape, dtype=float)
    resultVector = np.zeros(shape=matrixShape[0], dtype=float)
    for i in range(0, matrixShape[0]):
        if matrix[i,i] == 0:
            raise ValueError("Нулевой элемент на диагонали")
        for j in range(0, matrixShape[1]):
            if i == j:
                resultMatrix[i,j] = 0
            else:
                resultMatrix[i,j] = -matrix[i,j] / matrix[i, i]
    for i in range(resultVector.shape[0]):
        resultVector[i] = bVector[i] / matrix[i,i]
    return (resultMatrix, resultVector)

def main():
    MAX_ITERATIONS = 1000
    EPSILON = 1e-4
    # A1 = np.array([
    #     [3.5, 1, 2.1],
    #     [1, 4, 2.5],
    #     [2.1, 2.5, 4.7],
    # ], dtype=float)
    # b2 = np.array([0.56, 0.61, 0.96], dtype=float)

    A = np.array([
        [-0.68, -0.18, 0.02, 0.21],
        [0.16, -0.88, -0.14, 0.27],
        [0.37, 0.27, -1.02, -0.23],
        [0.12, 0.21, -0.18, -0.75]
    ], dtype=float)
    b = np.array([-1.83, 0.65, -2.23, 1.13], dtype=float)

    x = np.array([0,0,0,0], dtype=float)

    rearrangedSystem = rearrangeMatrix(A, b)
    A = rearrangedSystem[0]
    b = rearrangedSystem[1]

    for i in range(A.shape[0]):
        if abs(A[i, i]) <= np.sum(abs(A[i])) - abs(A[i, i]):
            print("Предупреждение: Отсутствует диагональное преобладание.")
            break

    try:
        transformedSystem = transformSystem(A, b)
        T = transformedSystem[0]
        c = transformedSystem[1]

        if np.max(np.sum(np.abs(T), axis=1)) >= 1:
            print("Сходимость не гарантирована") 

    except ValueError as e:
        print(e)
        return
    
    for k in range(0, MAX_ITERATIONS):
        print(f"Номер итерации: {k+1}")
        x_new = np.dot(T, x) + c

        if np.linalg.norm(x_new - x) < EPSILON:
            print(np.dot(A, x_new)-b)
            print (f"Ответ: {x_new}")
            return
        else:
            x = x_new
            r = np.dot(A, x) - b
            print(r)
            print(x)

if __name__ == '__main__':
    main()

    # A = np.array([
    #     [3.5, 1, 2.1],
    #     [1, 4, 2.5],
    #     [2.1, 2.5, 4.7],
    # ], dtype=float)
    # b = np.array([0.56, 0.61, 0.96], dtype=float)
    # print(np.linalg.solve(A, b))