import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

def read_dataset(filename):
    data = genfromtxt(filename, delimiter=',')
    # print(data.shape)
    data = data.flatten()
    # print(data.shape)
    print(data)
    return data


def calculate_membership_update_centroids(c,data,centroids,m,u):
    for r in range(100):
        flag = 0
        for i in range(int(c)):
            for k in range(np.size(data)):
                sigma = 0
                for j in range(int(c)):
                    if data[k] != centroids[i] and data[k] != centroids[j]:
                        sigma += pow(abs(data[k] - centroids[i]) / abs(data[k] - centroids[j]), (2 / (m - 1)))
                    elif data[k] == centroids[i]:
                        sigma = 1
                    elif data[k] != centroids[j]:
                        flag = 1
                if flag == 0:
                    u[i][k] = 1 / sigma
                else:
                    u[i][k] = 0
                    flag = 0

        for n in range(int(c)):
            nominator = 0
            denominator = 0
            for t in range(np.size(data)):
                # print("hi")
                nominator += pow(u[n][t], m) * data[t]
                denominator += pow(u[n][t], m)
                centroids[n] = nominator / denominator


def calculate_costs(data,c,u,m,centroids):
    # cost calculation
    item_cost = 0
    for j in range(np.size(data)):
        for i in range(int(c)):
            item_cost += pow(u[i][j], m) * pow((data[j] - centroids[i]), 2)

    return item_cost

def main():
    data = read_dataset('data1.csv')
    costs = []
    for p in range(0, 7):
        c = p
        m = 3
        centroids = np.random.choice(data.shape[0], int(c), replace=False)
        u = np.ones(shape=(int(c), np.size(data)))
        calculate_membership_update_centroids(c, data, centroids, m, u)
        costs.append(calculate_costs(data, c, u, m, centroids))

    plt.plot(costs)
    plt.show()

if __name__ == '__main__':
    main()
