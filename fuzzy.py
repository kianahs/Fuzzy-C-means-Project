import numpy as np
from numpy import genfromtxt


def read_dataset(filename):
    data = genfromtxt(filename, delimiter=',')
    # print(data.shape)
    data = data.flatten()
    # print(data.shape)
    # print(data)
    return data


def main():
    data = read_dataset('data1.csv')
    print("Enter number of clusters")
    c = input()
    m = 2
    centroids = np.random.choice(data.shape[0], int(c), replace=False)
    v = np.zeros(shape=(np.size(data),int(c)))
    u = np.ones(shape=(int(c), np.size(data)))
    cal = np.zeros(shape=(int(c), np.size(data)))
    # print(v.shape)
    for i in range(int(c)):
        v[:, i] = data[centroids[i]]

    # print(v.shape)
    # print(v)
    for i in range(int(c)):
        for j in range(int(c)):
            cal[i, :] += np.transpose(np.power(np.divide(np.subtract(data, v[:, i]), np.subtract(data, v[:, j])), (2/(m-1))))

    u = np.divide(u,cal)



if __name__ == '__main__':
    main()
