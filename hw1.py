import time

import numpy as np

class PerceptronMistakes:
    def __init__(self):
        pass


    def percetron_no_offset(self, init_theta, x_set, y_set):
        theta = np.array(init_theta)
        while True:
            mistake_update = 0
            for i in range(len(x_set)):
                xi = x_set[i]
                yi = y_set[i]
                val = (np.matmul(theta.transpose(), xi)) * yi
                mistake = val <= 0
                print("xi", xi, "val:", val, "mistake:", mistake)
                if mistake:
                    theta += yi * xi
                    mistake_update += 1
            if mistake_update == 0:
                print("theta:", theta)
                break

    def percetron(self, init_theta, init_offset, x_set, y_set):
        theta = np.array(init_theta)
        offset = init_offset
        while True:
            mistake_update = 0
            for i in range(len(x_set)):
                xi = x_set[i]
                yi = y_set[i]
                val = (np.matmul(theta.transpose(), xi) + offset) * yi
                mistake = val <= 0
                print("xi", xi, "val:", val, "mistake:", mistake)
                if mistake:
                    theta += yi * xi
                    offset += yi
                    mistake_update += 1
            if mistake_update == 0:
                print("theta:", theta, "offset:", offset)
                break

    def progression_of_separating_hyperplane(self, init_theta, x_set, y_set):
        theta = np.array(init_theta)
        process = []
        mistake_cnt = 0
        while True:
            mistake_update = 0
            for i in range(len(x_set)):
                xi = x_set[i]
                yi = y_set[i]
                val = np.matmul(theta.transpose(), xi) * yi
                mistake = val <= 0
                print("xi", xi, "val:", val, "mistake:", mistake)
                if mistake:
                    theta += yi * xi
                    process.append(np.copy(theta))
                    mistake_update += 1
                    print("process:", process)
            mistake_cnt += mistake_update
            if mistake_update == 0:
                print("no mistake, process:", "[" + ",".join(["[" + ",".join([str(j) for j in i.tolist()])+ "]" for i in process]) + "]", "mistake_cnt:", mistake_cnt)
                break


if __name__ == "__main__":
    pm = PerceptronMistakes()
    # x3 = [-1, 1.5]
    x3 = [-1, 10]
    x1_set = np.array([[-1, -1], [1, 0], x3])
    x2_set = np.array([[1, 0], x3, [-1, -1]])
# x_set = np.array([[-1, -1], [1, 0], [-1, 10.0]])
    y1_set = np.array([1.0, -1.0, 1.0])
    y2_set = np.array([-1.0, 1.0, 1.0])

    init_theta = np.array([0.0, 0.0])
    # init_theta = x_set[0]
    # print(x_set[1])
    # print(y_set[1])
    # print(init_theta.transpose())
    # print(np.matmul(init_theta.transpose(), x_set[1]) * y_set[1])

    # pm.progression_of_separating_hyperplane(init_theta, x1_set, y1_set)
    # print("##############")
    # pm.progression_of_separating_hyperplane(init_theta, x2_set, y2_set)

    # pm.percetron(np.array([0,0]), 0, np.array([[-1, 0], [0, 1]]), np.array([1, 1]))

    # pm.percetron_no_offset(np.array([0,0]), np.array([[-1, 0], [0, 1]]), np.array([1, 1]))

    pm.percetron_no_offset(np.array([0,0,0]), np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]), np.array([1, 1, 1]))
