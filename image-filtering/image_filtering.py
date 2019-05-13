import cv2
import numpy as np

image = cv2.imread('image.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

height, width = image.shape

edges = np.zeros((height,width), dtype=np.int16)
mask = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) * -1
for row in range(1, height - 1):
    for col in range(1, width - 1):
        I = image[row-1:row+2, col-1:col+2]
        edges[row, col] = np.sum(I * mask)

print(edges.max())
edges = np.interp(edges, (edges.min(), edges.max()), (0, 255))
edges = np.uint8(edges)

cv2.imshow('Edges', edges)
cv2.waitKey(0)