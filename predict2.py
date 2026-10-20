import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

img_width, img_height = 128, 128

model = tf.keras.models.load_model("final_model.h5")

img_path = "val/glioma/Te-gl_0018.jpg"

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
