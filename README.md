# 🧠 Brain Tumor Detection Using Deep Learning

A deep learning-based web application designed to classify brain MRI scans and assist in identifying different types of brain tumors.

## 📌 About the Project

Brain tumors require careful medical evaluation, and analyzing MRI scans is an important part of the diagnostic process. This project uses deep learning to classify brain MRI images into different categories through a web-based interface.

The repository also contains supporting utilities for document processing, embeddings, and retrieval-augmented generation (RAG).

## ✨ Features

- 🧠 **MRI Image Classification** — Predicts the category of a brain MRI scan using a trained deep learning model.
- 🔍 **Multiple Categories** — Supports Glioma, Meningioma, Pituitary Tumor, and No Tumor classification, subject to the trained model's configuration.
- 🌐 **Web Application** — Provides an interface for interacting with the application.
- 📚 **Knowledge Base** — Includes resources for knowledge retrieval and document-related functionality.
- 🤖 **NLP and Embeddings** — Includes utilities using sentence-transformer embeddings and text processing.

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Web Framework:** Streamlit
- **Image Processing:** NumPy
- **PDF Processing:** PyMuPDF
- **NLP and Embeddings:** Sentence Transformers
- **Text Splitting:** LangChain Text Splitters

## 📂 Project Structure

```text
Brain_tumor-rag-model/
├── knowledge_base/
├── static/
├── templates/
├── .streamlit/
├── app.py
├── cnn.py
├── build_index.py
├── model_utils.py
├── predict2.py
├── predict3.py
├── rag_utils.py
├── gui.py
├── best_model.h5
├── final_model.h5
├── prediction_result.txt
└── requirements.txt
```

*Note: The files listed above represent the main components of the project. The actual repository may contain additional files.*

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/vanshita-html/Brain_tumor-rag-model.git
cd Brain_tumor-rag-model
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.



## 🔮 Future Improvements

- Improve model performance and generalization.
- Enhance prediction visualization and explainability.
- Expand knowledge-base retrieval and question-answering capabilities.
- Improve the user interface and error handling.

## ⚠️ Disclaimer

This project is intended for educational and research purposes only. Its predictions are not a substitute for professional medical advice, diagnosis, or treatment.

## 👩‍💻 Author

**Vanshita Shivankar**

[GitHub Profile](https://github.com/vanshita-html)
