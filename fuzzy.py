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
    sigma = 0
    flag = 0
    for r in range(100):
        for i in range(int(c)):
            for k in range(data.shape[0]):
                for j in range(int(c)):
                    if np.array_equal(data[k, :], centroids[i, :]) == False and np.array_equal(data[k, :], centroids[j, :]) == False:
                        nominator = np.subtract(data[k, :], centroids[i, :])
                        denominator = np.subtract(data[k, :], centroids[j, :])
                        r_nom = 0
                        r_denom = 0
                        for item in range(data.shape[1]):
                            r_nom += pow(nominator[item], 2)  #0,item?
                            r_denom += pow(denominator[item], 2)

                        r_nom = pow(r_nom, 0.5)
                        r_denom = pow(r_denom, 0.5)
                        sigma += pow((r_nom/r_denom), (2 / (m - 1)))

                    elif np.array_equal(data[k, :], centroids[i, :]):
                        sigma = 1
                    elif np.array_equal(data[k, :], centroids[j, :]):
                        flag = 1
                if flag == 0:
                    u[i, k] = 1/sigma
                else:
                    u[i, k] = 0
                    flag = 0
                sigma = 0

        for n in range(int(c)):
            nominator = np.zeros(shape=(1, data.shape[1]))
            denominator = 0
            for t in range(data.shape[0]):
                nominator[0, :] += np.multiply(pow(u[n, t], m), data[t, :])
                denominator += pow(u[n, t], m)
                centroids[n, :] = np.divide(nominator, denominator)


def calculate_costs(data, c, u, m, centroids):
    # cost calculation
    cost = 0
    for j in range(data.shape[0]):
        for i in range(int(c)):
            nominator = np.subtract(data[j, :], centroids[i, :])
            r_nom = 0
            for item in range(data.shape[1]):
                r_nom += pow(nominator[item], 2)  # 0,item?

            cost += (pow(u[i, j], m) * r_nom)

    return cost
def plot_clusters(data, u, centroids):

    colors = np.array(["red", "green", "blue", "yellow", "pink", "black", "orange", "purple", "beige", "brown", "gray", "cyan", "magenta"])

    c = u.shape[0]
    picked_colors = colors[0:c]
    data_colors = []


    for x in range(data.shape[0]):
        maximum = u[0, x]
        cluster = 0
        for i in range(c):
            if u[i, x] > maximum:
                maximum = u[i, x]
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
    u = np.ones(shape=(int(c), data.shape[0]))

    while m != "end":
        calculate_membership_update_centroids(c, data, centroids, int(m), u)
        plot_clusters(data, u, centroids)
        print("enter m for c=3")
        m = input()



def main():

    # data = read_dataset('data4.csv')
    # costs = []
    # for p in range(0, 8):
    #     c = p
    #     m = 2
    #     centroids = np.ones(shape=(int(c), data.shape[1]))
    #
    #     for turn in range(int(c)):
    #         centroids[turn] = data[random.randrange(0, data.shape[0], 1)]
    #     u = np.ones(shape=(int(c), data.shape[0]))
    #     calculate_membership_update_centroids(c, data, centroids, m, u)
    #     costs.append(calculate_costs(data, c, u, m, centroids))
    #
    # plt.plot(costs)
    # plt.show()

    # comment content of main to run optional part
    optional_part()





if __name__ == '__main__':
    main()

