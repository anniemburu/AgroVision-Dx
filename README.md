# 🌿 AgroVision-Dx2: Plant Disease Classification

AgroVision-Dx2 is a specialized deep learning tool designed to identify plant pathologies from leaf images. Unlike traditional Convolutional Neural Networks (CNNs) that focus on local pixel neighborhoods, this project utilizes a Vision Transformer (ViT).

By breaking images into patches and applying Self-Attention mechanisms, the model understands the global context of a leaf, allowing it to distinguish subtle differences between similar-looking fungal spots and bacterial blights.

## 📂 Project Structure
To maintain a professional, production-ready workflow, the project is organized into a modular package:

```
AgroVision-Dx/
├── src/
│   ├── __init__.py      # Makes 'src' a Python package
│   ├── model.py         # ViT Architecture (timm-based)
│   ├── dataset.py       # Data Loading & Augmentation logic
│   └── train.py          # Main training & validation engine
├── models/              # Directory for saved .pth model weights
├── data/                # Dataset (train/val subfolders)          
├── app.py               # Streamlit Web Application (Deployment)
└── requirements.txt     # Project dependencies
```

## 🛠️ Installation & Setup
### 1. Clone the Repository

```
git clone https://github.com/your-username/AgroVision-Dx.git
cd AgroVision-Dx
```

### 2. Source Data
Download or use Kaggle APIs to download this [DATA](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).

### 3. Install Dependencies

```pip install -r requirements.txt```

## 🏋️ Training the Model
Before training, ensure your data/ folder is organized as follows:

```
data/
├── train/
│   ├── Healthy/
│   └── Late_Blight/
└── val/
    ├── Healthy/
    └── Late_Blight/
```
To start the training process:

```python -m src.train.py```

The script will automatically use your GPU if available, track validation accuracy, and save the best-performing model to models/best_model.pth.

## 🌐 Web Deployment
Once the model is trained, you can launch the interactive dashboard to test it with real images:

```streamlit run app.py```

  1. Upload a .jpg or .png of a plant leaf.
  2. The ViT model will process the image.
  3. The app will display the Diagnosis and a Confidence Score.
