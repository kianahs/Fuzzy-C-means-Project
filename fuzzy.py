import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import random
def read_dataset(filename):
    data = genfromtxt(filename, delimiter=',')
    # print(data.shape)

    # print(data.shape)
    # print(data)
    return data


def calculate_membership_update_centroids(c, data, centroids, m, u):
    sigma = np.zeros(shape=(1, data.shape[1]))
    ones = np.ones(shape=(1, data.shape[1]))
    flag = 0
    for r in range(100):
        for i in range(int(c)):
            for k in range(data.shape[0]):
                for j in range(int(c)):
                    if np.array_equal(data[k, :], centroids[i, :]) == False and np.array_equal(data[k, :], centroids[j, :]) == False:
                        sigma[0, :] += np.power(np.divide(np.abs(np.subtract(data[k, :], centroids[i, :])), np.abs(np.subtract(data[k, :], centroids[j, :]))), (2 / (m - 1)))
                    elif np.array_equal(data[k, :], centroids[i, :]):
                        sigma[0, :] = np.full(sigma.shape, 1)
                    elif np.array_equal(data[k, :], centroids[j, :]):
                        flag = 1
                if flag == 0:
                    u[i, k, :] = np.divide(ones, sigma)
                else:
                    u[i, k, :] = np.full(sigma.shape, 0)
                    flag = 0
                sigma[0, :] = np.full(sigma.shape, 0)

        for n in range(int(c)):
            nominator = np.zeros(shape=(1, data.shape[1]))
            denominator = np.zeros(shape=(1, data.shape[1]))
            for t in range(data.shape[0]):
                nominator[0, :] += np.multiply(np.power(u[n, t, :], m), data[t, :])
                denominator[0, :] += np.power(u[n, t, :], m)
                centroids[n, :] = np.divide(nominator, denominator)


def calculate_costs(data, c, u, m, centroids):
    # cost calculation
    item_cost = np.zeros(shape=(1, data.shape[1]))
    for j in range(data.shape[0]):
        for i in range(int(c)):
            item_cost[0, :] += np.multiply(np.power(u[i, j, :], m), np.power(np.subtract(data[j, :], centroids[i, :]), 2))
    # print("cost{} ".format(item_cost))
    total_cost = 0
    # print(item_cost.shape)
    for index in range(item_cost.shape[1]):
        total_cost += pow(item_cost[0, index], 2)

    # print(pow(total_cost, 0.5))
    return pow(total_cost, 0.5)

def plot_clusters(data, u, centroids):

    membership = np.zeros(shape=(u.shape[0], u.shape[1]))

    for i in range(u.shape[0]):
        for k in range(u.shape[1]):

            membership[i, k] = pow((pow(u[i, k, 0], 2) + pow(u[i, k, 1], 2)), 0.5)

    colors = np.array(["red", "green", "blue", "yellow", "pink", "black", "orange", "purple", "beige", "brown", "gray", "cyan", "magenta"])

    c = u.shape[0]
    picked_colors = colors[0:c]
    data_colors = []


    for x in range(data.shape[0]):
        maximum = membership[0, x]
        cluster = 0
        for i in range(c):
            if membership[i, x] > maximum:
                maximum = membership[i, x]
                cluster = i
        data_colors.append(picked_colors[cluster])

    plt.scatter(np.transpose(data[:, 0]), np.transpose(data[:, 1]), c=data_colors)
    plt.scatter(np.transpose(centroids[:, 0]),np.transpose(centroids[:, 1]), c=picked_colors, s=[500, 500, 500])
    plt.show()

def optional_part():

    print("enter m for c=3")
    m = input()
    data = read_dataset('data1.csv')
    c = 3
    centroids = np.ones(shape=(int(c), data.shape[1]))
    for turn in range(int(c)):
        centroids[turn] = data[random.randrange(0, data.shape[0], 1)]
    u = np.ones(shape=(int(c), data.shape[0], data.shape[1]))

    while m != "end":
        calculate_membership_update_centroids(c, data, centroids, int(m), u)
        plot_clusters(data, u, centroids)
        print("enter m for c=3")
        m = input()



def main():

    data = read_dataset('data1.csv')
    costs = []
    for p in range(0, 5):
        c = p
        m = 2
        centroids = np.ones(shape=(int(c), data.shape[1]))

        for turn in range(int(c)):
            centroids[turn] = data[random.randrange(0, data.shape[0], 1)]
        u = np.ones(shape=(int(c), data.shape[0], data.shape[1]))
        calculate_membership_update_centroids(c, data, centroids, m, u)
        costs.append(calculate_costs(data, c, u, m, centroids))

    plt.plot(costs)
    plt.show()

    # comment content of main to run optional part
    # optional_part()





if __name__ == '__main__':
    main()

