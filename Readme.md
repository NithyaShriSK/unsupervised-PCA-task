# Unsupervised PCA Clustering with Gradio

This project demonstrates **unsupervised learning** using **PCA (Principal Component Analysis)** for dimensionality reduction, followed by clustering techniques including **KMeans**, **Hierarchical Clustering**, and **DBSCAN**. The project is deployed as an interactive **Gradio app**.

---

## 🔗 Links

- **Hugging Face Space**: [PCA Gradio App](https://huggingface.co/spaces/NithyaShriSK/PCA_Gradio)
- **GitHub Repository**: [unsupervised-PCA-task-Hugging Space link](https://github.com/NithyaShriSK/unsupervised-PCA-task)

---

## 📌 Features

- Upload your dataset (CSV format)
- Automatic preprocessing with optional outlier removal (IQR method)
- Dimensionality reduction using **PCA**
- Apply clustering algorithms:
  - **KMeans**
  - **Agglomerative Hierarchical Clustering**
  - **DBSCAN**
- Visualize results in **2D PCA scatter plots**
- View cluster summary and evaluation metrics (Silhouette Score, Davies–Bouldin Index)

---

## ⚙️ Installation

Clone the repository and install required dependencies:

```bash
git clone https://github.com/NithyaShriSK/unsupervised-PCA-task.git
cd unsupervised-PCA-task
pip install -r requirements.txt
```

---

## 🚀 Running the App

To run locally:

```bash
python app.py
```

The app will launch in your browser at `http://127.0.0.1:7860/`.

---

## 📊 Dataset

The app works with any CSV dataset containing numeric features.  
For testing, you can use datasets like:
- Wine Quality Dataset (`winequality-red.csv`, `winequality-white.csv`)
- Wholesale Customers Dataset

---

## 📷 Screenshots

*(Add images of your Gradio app here, e.g., cluster plots and UI screenshots)*

---

## 🤝 Contributing

Contributions are welcome!  
Fork the repo, make improvements, and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License.

---
