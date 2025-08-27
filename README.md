# ENTRep Challenge Solution

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/Python-3.8+-brightgreen.svg" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch 2.0+">
  <img src="https://img.shields.io/badge/Kaggle-Ready-orange.svg" alt="Kaggle Ready">
</p>

This project contains a comprehensive solution for the **ENTRep Challenge**, which aims to advance vision-language AI for ENT (Ear, Nose, Throat) endoscopy analysis. It implements multiple state-of-the-art models and retrieval methods for classification, image-to-image, and text-to-image tasks, leveraging both vision and language modalities.

## 🚀 Key Features

- **Multi-task Solution**: Supports classification, image-to-image retrieval, and text-to-image retrieval
- **State-of-the-Art Models**: Implements CLIP, BioCLIP, SimCLR, ResNet, DenseNet, EfficientNet, SwinTransformer, DINOv2
- **Advanced Augmentations**: Includes MixUp, CutMix, Mosaic, and more sophisticated data augmentation techniques
- **Deep Learning Stack**: Built with PyTorch, Transformers, FAISS, Albumentations
- **Kaggle-Ready**: All modules are designed for seamless execution on Kaggle platform
- **Efficient Retrieval**: FAISS-powered similarity search for fast image and text retrieval
- **Comprehensive Evaluation**: Built-in evaluation metrics and submission generation

## 📋 Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Models](#models)
- [Tasks](#tasks)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 16GB+ RAM (recommended)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/magnusdtd/ENTRep.git
cd ENTRep

# Install required packages
pip install -r requirements.txt
```

### Key Dependencies

- **Deep Learning**: PyTorch (2.7.1), TorchVision (0.22.1), Transformers (4.49.0)
- **Computer Vision**: Albumentations (2.0.8), OpenCV, Pillow, scikit-image
- **Machine Learning**: scikit-learn (1.7.0), FAISS (1.11.0), TimM (1.0.15)
- **Data Processing**: Pandas (2.3.0), NumPy (2.3.0)
- **Visualization**: Matplotlib (3.10.3), Seaborn (0.13.2)
- **Additional**: CLIP, Sentence Transformers, Datasets

## 📁 Project Structure

```
ENTRep/
├── BioCLIP/                 # BioCLIP implementation and notebooks
│   ├── main.py             # Main BioCLIP training/inference script
│   ├── TextToImage.ipynb   # Text-to-Image retrieval notebook
│   └── ...
├── CLIP/                   # Custom CLIP implementation
│   ├── CLIP.py            # CLIP model architecture
│   ├── main.py            # Training and evaluation script
│   └── ...
├── classification/         # Image classification modules
│   ├── classification.py  # Main classification class
│   ├── dataset.py         # Dataset handling
│   └── ...
├── i2i/                   # Image-to-Image retrieval
│   ├── main.py            # I2I main script
│   ├── ENTRepDataset.py   # Custom dataset class
│   └── ...
├── FAISS/                 # FAISS-based similarity search
│   ├── pipeline.py        # FAISS indexing pipeline
│   ├── feature_extractor.py
│   └── ...
├── utils/                 # Utility functions
│   ├── augmentations/     # Advanced augmentation techniques
│   ├── data.py           # Data loading utilities
│   └── ...
├── scripts/               # Training and evaluation scripts
│   ├── select_model.py    # Model selection script
│   ├── select_adv_aug.py  # Augmentation selection
│   └── ...
├── requirements.txt       # Project dependencies
└── README.md             # This file
```

## 🎯 Usage

### 1. Classification Task

Train a classification model for ENT endoscopy image classification:

```bash
# Train ResNet50 for classification
cd classification
python main.py --model resnet50 --epochs 50 --batch_size 32

# Evaluate model performance
python evaluate.py --model_path ./checkpoints/best_model.pth
```

### 2. Image-to-Image Retrieval

Perform image-to-image retrieval using learned embeddings:

```bash
# Train I2I model
cd i2i
python train.py --embedding_dim 512 --epochs 100

# Run inference
python main.py --query_image path/to/query.jpg --top_k 10
```

### 3. Text-to-Image Retrieval

Use CLIP or BioCLIP for text-to-image retrieval:

```bash
# BioCLIP approach
cd BioCLIP
python main.py

# Custom CLIP approach
cd CLIP
python main.py --epochs 50 --batch_size 16
```

### 4. FAISS-powered Retrieval

For large-scale efficient retrieval:

```bash
cd FAISS
python pipeline.py --feature_extractor dinov2 --index_type IVF
```

### 5. Advanced Augmentation Selection

Experiment with different augmentation strategies:

```bash
cd scripts
python select_adv_aug.py --config adv_aug_space.json
```

## 🤖 Models

### Implemented Architectures

| Model | Type | Use Case | Performance |
|-------|------|----------|-------------|
| **BioCLIP** | Vision-Language | Text-to-Image Retrieval | State-of-the-art |
| **CLIP** | Vision-Language | Multi-modal tasks | High |
| **ResNet50** | CNN | Classification | 71.93% accuracy |
| **DenseNet** | CNN | Classification | High |
| **EfficientNet** | CNN | Classification | Efficient |
| **SwinTransformer** | Transformer | Classification | Modern |
| **DINOv2** | Self-supervised | Feature extraction | Robust |
| **SimCLR** | Self-supervised | Representation learning | Contrastive |

### Model Selection

Use the automated model selection script:

```bash
cd scripts
python select_model.py --config model_space.json
```

## 📊 Tasks

### 1. **Classification**
- Multi-class classification of ENT endoscopy images
- Categories: nose-right, nose-left, ear-right, ear-left, vc-open, vc-closed, throat
- Support for various CNN architectures and transformers

### 2. **Image-to-Image Retrieval**
- Find similar images based on visual content
- Support for learned embeddings and pre-trained features
- FAISS integration for efficient similarity search

### 3. **Text-to-Image Retrieval**
- Retrieve relevant images based on text descriptions
- BioCLIP and CLIP implementations
- Cross-modal understanding capabilities

## 🏆 Results

### Classification Results
- **ResNet50**: 71.93% accuracy
- Multiple architectures evaluated with comprehensive metrics

### Retrieval Performance
- Recall@K metrics for retrieval tasks
- Mean Reciprocal Rank (MRR) evaluation
- Comprehensive evaluation scripts included

## 🔬 Advanced Features

### Data Augmentation
- **MixUp**: Convex combinations of training examples
- **CutMix**: Regional dropout and label mixing
- **Mosaic**: Multi-image composition augmentation
- **Standard**: Rotation, flipping, color jittering, etc.

### Training Enhancements
- Early stopping with patience
- Learning rate scheduling
- K-fold cross-validation
- Advanced loss functions (Focal Loss, Learnable Loss)

### Kaggle Integration
- Seamless execution on Kaggle platform
- Automated submission generation
- GPU optimization for Kaggle environments

## 🚀 Getting Started

### Quick Start Example

```python
# Load and use a pre-trained model
from classification.classification import Classification
from utils.data import get_train_df

# Initialize model
model = Classification(model_name='resnet50', num_classes=7)

# Load data
df = get_train_df()

# Train model
model.train(df, epochs=10)

# Make predictions
predictions = model.predict(test_images)
```

### Kaggle Notebook Usage

1. Clone the repository in Kaggle
2. Navigate to desired task folder
3. Run the corresponding Jupyter notebook
4. Follow the step-by-step instructions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/magnusdtd/ENTRep.git
cd ENTRep

# Create development environment
pip install -r requirements.txt

# Run tests (if available)
python -m pytest tests/
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@misc{entrep2025,
  title={ENTRep Challenge Solution: Comprehensive Vision-Language AI for ENT Endoscopy Analysis},
  author={Đàm Tiến Đạt},
  year={2025},
  url={https://github.com/magnusdtd/ENTRep}
}
```

## 🎯 Acknowledgments

- ENTRep Challenge organizers
- Open source community for the foundational models
- Kaggle platform for computational resources

---

<p align="center">
  <strong>Built with ❤️ for advancing medical AI</strong>
</p>