import numpy

"""
    This is single level neural network with activation code with dynamic weightage upgrade
    Here we are using implementation for Logical AND
    Threshold and weightage values can be random
    
    For threshold = 0.19
    weightage = [0.3, -0.1] : loop run time is 5
    weightage = [0.1, -0.3] : loop run time is 6
    weightage = [0.3, 1] : loop run time is 10
    
    For threshold = 0.25
    weightage = [0.3, -0.1] : loop run time is 4    
    weightage = [0.3, 1] : loop run time is 9
    weightage = [1, 1] : loop run time is 9
    weightage = [-1, -1] : loop run time is 13
"""

x_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_outputs = [0, 0, 0, 1]
# Logical OR
# y_outputs = [0, 1, 1, 1]
threshold = 0.19
alpha = 0.1
weightage = [0.3, -0.1]

def activation_code(x, w):
    product = numpy.dot(x, w)
    if product >= threshold:
        return 1
    return 0


def new_weightage(w, x, error) -> weightage:
    new_w = []
    for i in range(len(w)):
        w_n = w[i] + alpha * x[i] * error
        #new_w.append(numpy.round(w_n, 2))
        new_w.append(w_n)
    return new_w


def find_error(y, y_d):
    return y - y_d

def is_match(y_derived):
    for i in range(len(y_derived)):
        if y_outputs[i] != y_derived[i]:
            return False
    return True


def main():
    global weightage
    stop_loop = False
    trigger_end = False
    loop_count = 0
    max_loop_count = 100
    y_derived = []

    while not (stop_loop | trigger_end):
        y = []
        for i in range(len(x_inputs)):
            y_derived = activation_code(x_inputs[i], weightage)
            error = find_error(y_outputs[i], y_derived)
            y.append(y_derived)
            weightage = new_weightage(weightage, x_inputs[i], error)
            print(y_outputs[i], y_derived, error, weightage)

        loop_count = loop_count + 1
        if is_match(y):
            stop_loop = True
            y_derived = y
        if loop_count >= max_loop_count:
            trigger_end = True
            y_derived = y

    print("Weightage: ", weightage)
    print("Loop ran for {} times".format(loop_count))
    print("Y derived: ", y_derived)


if __name__ == "__main__":
    main()



