# load_model_sample.py
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Activation, Flatten
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os


def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(200, 200))
    img = img_to_array(img)
    img = img.reshape(1, 200, 200, 3)
    # center pixel data
    img = img.astype('float32')/255
    # predict the classresult = model.predict(img)
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)  
    img_tensor.astype('float32')/255       # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)                                   # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor


if __name__ == "__main__":
