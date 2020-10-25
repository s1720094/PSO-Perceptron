import numpy as np


# Number of points M=2
x = {1:(2,2), -1:(-2,-2)}

# Generate number of s/2 data vectors for each of the points by adding an amount
# of noise

def genRandom(label, point, s):
    x = point[0]
    y = point[1]
    # List of point of form (x,y,label)
    points = []
    for i in range (int(s/2)):
        x_var = x + np.random.normal(0, .1)
        y_var = y +  np.random.normal(0, .1)
        points.append((x_var, y_var, label))


    return points

def makePoints(x, s):
    allpoints = []
    for point in x:
        allpoints.append(genRandom(point, x[point], 10))
    return allpoints

print(makePoints(x, 10))
