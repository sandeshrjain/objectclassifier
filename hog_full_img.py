# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import scipy
import scipy.signal as sig
import matplotlib.pyplot as plt 
# With mode="L", we force the image to be parsed in the grayscale, so it is
# actually unnecessary to convert the photo color beforehand.
img = scipy.misc.imread("manu-2004.jpg", mode="L")

# Define the Sobel operator kernels.
kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

G_x = sig.convolve2d(img, kernel_x, mode='same') 
G_y = sig.convolve2d(img, kernel_y, mode='same') 

# Plot them!
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# Actually plt.imshow() can handle the value scale well even if I don't do 
# the transformation (G_x + 255) / 2.
ax1.imshow((G_x + 255) / 2, cmap='gray'); ax1.set_xlabel("Gx")
ax2.imshow((G_y + 255) / 2, cmap='gray'); ax2.set_xlabel("Gy")
plt.show()
v = (G_x.shape[0])
h = (G_x.shape[1])
b=16

N_BUCKETS = 9
CELL_SIZE = int(h/32)  # Each cell is 8x8 pixels
BLOCK_SIZE = 2  # Each block is 2x2 cells

def assign_bucket_vals(m, d, bucket_vals):
    print(d)
    left_bin = int(d / 20.)
    # Handle the case when the direction is between [160, 180)
    if(left_bin == 8):
        right_bin = 0
        left_val= - m * (right_bin * 20 - d) / 20
        right_val = - m * (d - left_bin * 20) / 20        
    else:
        right_bin = (int(d / 20.) + 1) 
        left_val= m * (right_bin * 20 - d) / 20
        right_val = m * (d - left_bin * 20) / 20
    #assert 0 <= left_bin < right_bin < N_BUCKETS


    bucket_vals[left_bin] += left_val
    bucket_vals[right_bin] += right_val
    return bucket_vals
def get_magnitude_hist_cell(loc_x, loc_y):
    # (loc_x, loc_y) defines the top left corner of the target cell.
    cell_x = G_x[loc_x:loc_x + CELL_SIZE, loc_y:loc_y + CELL_SIZE].flatten()
    cell_y = G_y[loc_x:loc_x + CELL_SIZE, loc_y:loc_y + CELL_SIZE].flatten()
    magnitudes = np.sqrt(cell_x * cell_x + cell_y * cell_y)
    print(magnitudes)
    directions = np.zeros(len(magnitudes))
    for d in range(len(cell_x)):
        
        if(cell_y[d]*cell_x[d] < 0 ):
            
            directions[d] = 180 - (np.abs( np.arctan(cell_y[d] /(0.01 + cell_x[d]))) * 180 / np.pi )
        else:
            directions[d] = np.arctan(cell_y[d] /(0.01 + cell_x[d])) * 180 / np.pi

    bucket_vals = np.zeros(N_BUCKETS)
    for i in range(len(magnitudes)):
        assign_bucket_vals(magnitudes[i], directions[i], bucket_vals)

    return bucket_vals
import functools
def get_magnitude_hist_block(loc_x, loc_y):
    # (loc_x, loc_y) defines the top left corner of the target block.
    return functools.reduce(
        lambda arr1, arr2: np.concatenate((arr1, arr2)),
        [get_magnitude_hist_cell(x, y) for x, y in zip(
            [loc_x, loc_x + CELL_SIZE, loc_x, loc_x + CELL_SIZE],
            [loc_y, loc_y, loc_y + CELL_SIZE, loc_y + CELL_SIZE],
        )]
    )
loc_x = loc_y = 0
img_bin = []
for i in range(int(16*v/h)):
    for j in range(16):
        img_bin.append(get_magnitude_hist_block(loc_x, loc_y))
        loc_x += CELL_SIZE
    loc_x = 0
    loc_y += CELL_SIZE
    
# Random location [200, 200] as an example.
loc_x = loc_y = 20

ydata = get_magnitude_hist_block(loc_x, loc_y)
ydata = ydata / np.linalg.norm(ydata)

xdata = range(len(ydata))
bucket_names = np.tile(np.arange(N_BUCKETS), BLOCK_SIZE * BLOCK_SIZE)

assert len(ydata) == N_BUCKETS * (BLOCK_SIZE * BLOCK_SIZE)
assert len(bucket_names) == len(ydata)

plt.figure(figsize=(10, 3))
plt.bar(xdata, ydata, align='center', alpha=0.8, width=0.9)
plt.xticks(xdata, bucket_names * 20, rotation=90)
plt.xlabel('Direction buckets')
plt.ylabel('Magnitude')
plt.grid(ls='--', color='k', alpha=0.1)
plt.title("HOG of block at [%d, %d]" % (loc_x, loc_y))
plt.tight_layout()