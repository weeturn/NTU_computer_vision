#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
# binary
img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
binary = np.zeros([512, 512], dtype=img.dtype)
for i in range(512):
    for j in range(512):
        if img[i][j] < 128:
            binary[i][j] = 0
        else:
            binary[i][j] = 255

plt.imshow(binary, cmap = 'gray')
plt.axis('off')
plt.show()
cv2.imwrite('binary.png', binary)


# In[9]:


#histogram
import imageio
img = imageio.imread("lena.bmp")
count = [int(0)] * 256
for i in img:
    for j in i:
        count[j] = count[j] + 1
x = np.arange(256)
plt.bar(x,count)
plt.savefig('histogram.png')
plt.show()


# In[10]:


#connected component
labels = np.zeros([512, 512], dtype=int)
labelCount = 0
change = 1

#initialize
for i in range(512):
    for j in range(512):
        if binary[i][j] == 255:
            labelCount += 1
            labels[i][j] = labelCount

while change == 1:
    change = 0
    for i in range(512):
        for j in range(512):
            if labels[i][j] != 0:
                if i > 0 and labels[i-1][j] != 0:
                    if labels[i-1][j] < labels[i][j]:
                        labels[i][j] = labels[i-1][j]
                        change = 1
                if i < 511 and labels[i+1][j] != 0:
                    if labels[i+1][j] < labels[i][j]:
                        labels[i][j] = labels[i+1][j]
                        change = 1
                if j > 0 and labels[i][j-1] != 0:
                    if labels[i][j-1] < labels[i][j]:
                        labels[i][j] = labels[i][j-1]
                        change = 1
                if j < 511 and labels[i][j+1] != 0:
                    if labels[i][j+1] < labels[i][j]:
                        labels[i][j] = labels[i][j+1]
                        change = 1
    for i in range(511, -1, -1):
        for j in range(511, -1, -1):
            if labels[i][j] != 0:
                if i > 0 and labels[i-1][j] != 0:
                    if labels[i-1][j] < labels[i][j]:
                        labels[i][j] = labels[i-1][j]
                        change = 1
                if i < 511 and labels[i+1][j] != 0:
                    if labels[i+1][j] < labels[i][j]:
                        labels[i][j] = labels[i+1][j]
                        change = 1
                if j > 0 and labels[i][j-1] != 0:
                    if labels[i][j-1] < labels[i][j]:
                        labels[i][j] = labels[i][j-1]
                        change = 1
                if j < 511 and labels[i][j+1] != 0:
                    if labels[i][j+1] < labels[i][j]:
                        labels[i][j] = labels[i][j+1]
                        change = 1
                        
#bounding box
img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
done = []
for i in range(512):
    for j in range(512):
        if labels[i][j] != 0 and labels[i][j] not in done:
            area = np.sum(labels == labels[i][j])
            done.append(labels[i][j])

            #get rectangle
            if area > 500:
                min_x = min_y = 999
                max_x = max_y = -1
                sum_x = sum_y = count = 0
                for x in range(512):
                    for y in range(512):
                        if labels[x][y] == labels[i][j]:
                            sum_x += x
                            sum_y += y
                            count += 1
                            if x < min_x:
                                min_x = x
                            if y < min_y:
                                min_y = y
                            if x > max_x:
                                max_x = x
                            if y > max_y:
                                max_y = y
                cv2.rectangle(img, (min_y, min_x), (max_y, max_x), (255, 255, 0), 2)
                sum_x /= count
                sum_y /= count
                sum_x = int(sum_x)
                sum_y = int(sum_y)
                cv2.circle(img, (sum_y, sum_x), 6, (0, 0, 255), 2)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis('off')
plt.show()
cv2.imwrite('connected_components.png', img)

