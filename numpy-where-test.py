# numpy-where-test.py

from __future__ import print_function
# import cv2 as cv
import numpy as np

a = np.array([[1,2,3],[4,5,6],[7,8,9]])

w = np.where(a%2==1)

# print(np.c_[w[0],w[1]])

print(np.column_stack(w))

