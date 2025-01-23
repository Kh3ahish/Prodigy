
import tensorflow_hub as hub
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

# Load the pre-trained model from TensorFlow Hub
model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Function to load and preprocess an image
def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img

# Load content and style images
content_image_path = '/Screenshot 2023-11-12 164138.png'  # Update path as necessary
style_image_path = '/Screenshot (9).png'  # Update path as necessary

content_image = load_image(content_image_path)
style_image = load_image(style_image_path)

# Display content and style images
plt.imshow(np.squeeze(content_image))
plt.title("Content Image")
plt.axis('off')
plt.show()

plt.imshow(np.squeeze(style_image))
plt.title("Style Image")
plt.axis('off')
plt.show()

# Generate the stylized image
stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]

# Display the stylized image
plt.imshow(np.squeeze(stylized_image))
plt.title("Stylized Image")
plt.axis('off')
plt.show()
