import numpy as np
import math

def CalculateDistance(inputs, weights):
    sum = 0
    for x in range(0, len(weights)):
        value = weights[x] - inputs[x]
        sum += value * value
        print("Weight", sum)
        return sum
    
def UpdateWeights(inputs, weights, l):
    weight_update = np.array([0.0,0.0,0.0,0.0], dtype='float')
    print(weights, l, inputs, weights)
    weight_update = weights + (l * (inputs - weights))
    return weight_update

def train(inputs, weights, l):
    run = True
    epoch = 0
    wt_update = np.array([0.0,0.0,0.0,0.0], dtype = 'float')
    while run:
        print("-----------------------")
        epoch += 1
        
        for x in range(0, len(inputs)):
            value = 0
            vector = 0
            for y in range(0, len(weights)):
                print("input", inputs[x])
                print('weights', weights[x])
                a_output = CalculateDistance(inputs[x], weights[y])
                print('Distance', y, ":", a_output)
                print()
                if value == 0:
                    value = a_output
                    vector = y
                elif a_output < value:
                    value = a_output
                    vector = y
        print('Winner for iteration', x, 'Distance', value, "weight vector", vector + 1)
        wt_update = UpdateWeights(inputs[x], weights[vector], l)
        weights[vector] = wt_update 
        print("Weight Update for vector", vector + 1, wt_update)
        print("update weigths", weights)
    l = l * 0.5
    if epoch >=2:
        run = False
    else:
        run = True

x = np.array([[1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1]], dtype='float')
w = np.array([[0.2, 0.6, 0.5, 0.9], [0.8, 0.4, 0.7, 0.3]], dtype='float')
l = 0.6

train(x, w, l)
