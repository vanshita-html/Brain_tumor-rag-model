import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog, Label, Button, messagebox
from PIL import Image, ImageTk

# Load trained model safely
model_path = "final_model.h5" if os.path.exists("final_model.h5") else "best_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError("Model file not found. Ensure 'final_model.h5' or 'best_model.h5' is present.")
model = tf.keras.models.load_model(model_path, compile=False)

img_width, img_height = 128, 128
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

def predict_image(file_path):
    img = image.load_img(file_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    pred_index = np.argmax(prediction)

    result = classes[pred_index]
    confidence = prediction[0][pred_index] * 100

    result_label.config(text=f"Prediction: {result}\nConfidence: {confidence:.2f}%")

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((250, 250))
        img = ImageTk.PhotoImage(img)

        panel.config(image=img)
        panel.image = img

        predict_image(file_path)

# GUI Window
root = tk.Tk()
root.title("Brain Tumor Detection")
root.geometry("400x450")

upload_btn = Button(root, text="Upload MRI Image", command=upload_image, padx=10, pady=5)
upload_btn.pack(pady=20)

panel = Label(root)
panel.pack()

result_label = Label(root, text="", font=("Arial", 12))
result_label.pack(pady=20)

root.mainloop()
