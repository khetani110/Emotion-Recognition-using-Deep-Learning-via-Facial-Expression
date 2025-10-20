# 🧠 Emotion Recognition using Deep Learning via Facial Expression

---

## 👥 Team Members

| Name                     | Student ID |
| ------------------------ | ---------- |
| **MM Al Irfan Bhuiyan**  | S372087    |
| **Hasnain Khan**         | S371833    |
| **Hasnain Reza Khetani** | S374688    |
| **Hamza Gul**            | S372115    |

---

## 📘 Project Overview

This project implements an **Emotion Recognition System** using an **ensemble of nine convolutional neural networks (CNNs)**, including ResNet, DenseNet, EfficientNet, VGG19, and others.
It classifies facial emotions from the **FER-2013 dataset** and provides an **interactive Streamlit interface** for image upload or webcam capture.
All experiments and deployment are designed to run entirely in **Google Colab** (no local setup required).

---

## 📁 Folder Structure

```
SOURCE CODE/
│
├── application/
│   └── app.py                  ← Streamlit application
│
├── ensembling/
│   ├── ensembletesting.ipynb
│   └── final_application_and_ensembling.ipynb
│
├── Models Training/
│   ├── modeltraining1.ipynb
│   ├── modeltraining2.ipynb
│   ├── modeltraining3.ipynb
│   └── modeltraining4.ipynb
│
├── Trained_Models/             ← Folder intentionally empty on GitHub
│   └── (Download models from Google Drive link below)
│
└── requirements.txt
```

📦 **Trained Models (.pth) Download Link:**
[➡️ Google Drive – Trained_Models Folder](https://drive.google.com/drive/folders/1pNNT_7XInDT6Leu3wqZOArZ6_JzlwNex?usp=sharing)

---

## ⚙️ 1. Environment Setup (Google Colab)

### Step 1 – Install Required Packages

Run the following in a Colab cell:

```bash
!pip install streamlit torch torchvision timm pillow pandas numpy matplotlib opencv-python-headless pyngrok
```

These are the same packages listed in `requirements.txt`.

---

## ☁️ 2. Mount Google Drive

Upload your project folder (including `application`, `ensembling`, `Models Training`, and empty `Trained_Models`) to Google Drive, then mount it:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Update the model path inside **`app.py`**:

```python
models_path = "/content/drive/MyDrive/Trained_Models"
```

Now, download all `.pth` models from the shared Drive folder above and place them inside this same directory in your Drive.

---

## 📦 3. Download FER-2013 Dataset via Kaggle API

1. Upload your **`kaggle.json`** to Colab:

   ```python
   from google.colab import files
   files.upload()   # select kaggle.json
   ```
2. Configure and download dataset:

   ```bash
   !mkdir -p ~/.kaggle
   !mv kaggle.json ~/.kaggle/
   !chmod 600 ~/.kaggle/kaggle.json
   !kaggle datasets download -d msambare/fer2013
   !unzip fer2013.zip -d /content/dataset
   ```

---

## 🚀 4. Run the Streamlit Application

In a Colab cell, copy the app file and run it using **ngrok or localtunnel** for a public URL:

```bash
!cp /content/drive/MyDrive/application/app.py /content/app.py
!streamlit run app.py & npx localtunnel --port 8501
```

After a few seconds, Colab will display a **public URL** — click it to access your live app.

---

## 🖼️ 5. Using the Application

* Upload or capture an image through the Streamlit interface.
* The ensemble model predicts the facial emotion by combining outputs from all CNN models.
* Final prediction and confidence levels are displayed in real-time.

---

## 🧠 6. Retraining Models (Optional)

If you wish to retrain models:

1. Open any notebook from `Models Training/`.
2. Adjust dataset and Drive paths.
3. Run cells sequentially to generate new `.pth` weights.
4. Upload the new weights to your Google Drive `Trained_Models` folder.

---

## 🏁 Summary

| Task                 | Where to Run | Command / Action                                                                                   |
| -------------------- | ------------ | -------------------------------------------------------------------------------------------------- |
| Mount Drive          | Google Colab | `drive.mount('/content/drive')`                                                                    |
| Install Dependencies | Google Colab | `!pip install -r requirements.txt`                                                                 |
| Download Dataset     | Google Colab | `!kaggle datasets download -d msambare/fer2013`                                                    |
| Run Application      | Google Colab | `!streamlit run app.py & npx localtunnel --port 8501`                                              |
| Access Models        | Google Drive | [Drive Link](https://drive.google.com/drive/folders/1pNNT_7XInDT6Leu3wqZOArZ6_JzlwNex?usp=sharing) |

---

### 🧾 Notes

* All experiments and app deployment must be done **inside Google Colab**.
* **Do not run locally** — paths and dependencies are configured for Colab and Drive.
* Model files are hosted on Google Drive to avoid GitHub size limits.
* For any errors related to paths, recheck `models_path` and dataset directory.


