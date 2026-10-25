import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

import os
import sys

img_width, img_height = 128, 128

model_path = "final_model.h5" if os.path.exists("final_model.h5") else "best_model.h5"
model = tf.keras.models.load_model(model_path, compile=False)

if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    img_path = sys.argv[1]
elif os.path.exists("val/glioma/Te-gl_0018.jpg"):
    img_path = "val/glioma/Te-gl_0018.jpg"
elif os.path.exists("static/Tr-gl_1246.jpg"):
    img_path = "static/Tr-gl_1246.jpg"
else:
    raise FileNotFoundError("No valid MRI image file found for prediction.")

print(f"Loading image from: {img_path}")
img = image.load_img(img_path, target_size=(img_width, img_height))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

pred_index = np.argmax(prediction)
result = classes[pred_index]
confidence = prediction[0][pred_index] * 100

print("Predicted Class:", result)
print("Confidence: {:.2f}%".format(confidence))
