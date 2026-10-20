import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

img_width, img_height = 128, 128

model = tf.keras.models.load_model("final_model.h5")

img_path = "test_image/sample16.jpg"

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

print("\nClass-wise Probabilities:")
for i, prob in enumerate(prediction[0]):
    print(f"{classes[i]} : {prob*100:.2f}%")

with open("prediction_result.txt", "w") as f:
    f.write(f"Predicted Class: {result}\n")
    f.write(f"Confidence: {confidence:.2f}%\n\n")
    f.write("Class-wise Probabilities:\n")

    for i, prob in enumerate(prediction[0]):
        f.write(f"{classes[i]} : {prob*100:.2f}%\n")

print("\nResult saved to prediction_result.txt")

