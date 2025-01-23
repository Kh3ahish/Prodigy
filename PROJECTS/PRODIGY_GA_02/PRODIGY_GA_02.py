!pip install opencv-python

import numpy as np



# for reading in images

import matplotlib.image as mpimg



import matplotlib.pyplot as plt



# computer vision library for data manipulation in images

import cv2



%matplotlib inline

# Read in the image

image = mpimg.imread('lenna.png')



# Print out the image dimensions

print('Image dimensions:', image.shape)

# To see the image 

plt.imshow(image);

# The image is stored as numbers in a 3D numpy array

image[:, :, 0]

# Get the pixel value at row 100, column 200 in the first channel

pixel_value = image[100, 200, 0]

pixel_value

# 0 is red channel

red_channel = image[:, :, 0]

plt.imshow(red_channel);

# 1 is green channel

green_channel = image[:, :, 1]

plt.imshow(green_channel);

# 2 is blue channel

blue_channel = image[:, :, 2]

plt.imshow(blue_channel);

# 2 is transparency channel

transparent_channel = image[:, :, 2]

plt.imshow(transparent_channel);

# Read in the image

#image = mpimg.imread('seven.png')

#image = mpimg.imread('lenna.png')

image = mpimg.imread('astronaut.png')



# Print out the image dimensions

print('Original Image dimensions:', image.shape)



plt.imshow(image);

# Change from color to grayscale

gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

plt.imshow(gray_image, cmap='gray');



print('Grayscale Image dimensions:', gray_image.shape)

gray_image.shape

# Finds the maximum and minimum grayscale values in this image



max_val = np.amax(gray_image)

min_val = np.amin(gray_image)



print('Max: ', max_val)

print('Min: ', min_val)

# Prints specific grayscale pixel values

# What is the pixel value at x = 400 and y = 300



x = 300

y = 300



print(gray_image[y,x])

# Creating a 5x5 image using just grayscale, numerical values

tiny_image = np.array([[0, 20, 30, 150, 120],

                      [200, 200, 250, 70, 3],

                      [50, 180, 85, 40, 90],

                      [240, 100, 50, 255, 10],

                      [30, 0, 75, 190, 220]])



# To show the pixel grid, use matshow

plt.imshow(tiny_image, cmap='gray')

# Reduce the size of the image (Image downscaling)



resized_image = cv2.resize(gray_image, (28, 28))



# Display the grayscale image

plt.imshow(resized_image, cmap='gray')


# Finds the maximum and minimum grayscale values in this image



max_val = np.amax(resized_image)

min_val = np.amin(resized_image)



print('Max: ', max_val)

print('Min: ', min_val)

resized_image.shape