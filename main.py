import numpy as np

def rearrangeMatrix(A, b):
    size = A.shape[0]
    newA = A.copy()
    newB = b.copy()

    for i in range(size):
        if newA[i, i] == 0:
            for j in range(i + 1, size):
                if newA[j, i] != 0:
                    newA[[i, j]] = newA[[j, i]]
                    newB[[i, j]] = newB[[j, i]]
                    break
            else:
                for j in range(i + 1, size):
                    if np.any(newA[j] != 0):
                        newA[[i, j]] = newA[[j, i]]
                        newB[[i, j]] = newB[[j, i]]
                        break
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
    # Фиксированное количество итераций
    MAX_ITERATIONS = 100
    # Число для опеределения, когда нам нужно остановиться
    EPSILON = 1e-6
    # Инициализируем систему
    A = np.array([
        [10, -1, 2, 0],
        [-1, 11, -1, 3],
        [2, -1, 0, -4],
        [0, 3, -1, 8]
    ], dtype=float)
    b = np.array([6, 25, -11, 15], dtype=float)
    # Стартовый вектор x0
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
    except ValueError as e:
        print(e)
        return
    
    for k in range(0, MAX_ITERATIONS):
        print(f"Номер итерации: {k+1}")
        x_new = np.dot(T, x) + c    
        if np.linalg.norm(x_new - x) < EPSILON:
            print (f"Ответ: {x_new}")
            return
        else:
            x = x_new
            print(x)

if __name__ == '__main__':
    main()