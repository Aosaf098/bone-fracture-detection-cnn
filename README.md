# Bone Fracture Detection using CNN 🦴

## Overview
This project implements a **Convolutional Neural Network (CNN)** from scratch using **PyTorch** to detect and classify different types of bone fractures from X-ray images.

The model assumes a multi-class classification problem, identifying specific fracture types such as Avulsion, Comminuted, Greenstick, Hairline, and more.

## 🚀 Key Features
* **Custom CNN Architecture**: Built a flexible, dynamic CNN class (`LayeredCNN`) that allows for easy experimentation with the number of layers and filters.
* **3-Layer Model**: The final model utilizes 3 convolutional layers with ReLU activation, MaxPooling, and Batch Normalization.
* **Data Augmentation**: Used `torchvision.transforms` for resizing and normalizing X-ray images.
* **Evaluation Metrics**: Includes Training/Validation Loss & Accuracy curves and a Confusion Matrix to analyze model performance.
* **Prediction Visualization**: Visualizes model predictions on unseen data with color-coded labels (Green = Correct, Red = Incorrect).

## 🛠️ Tech Stack
* **Language**: Python 3.11
* **Deep Learning**: PyTorch (`torch`, `torchvision`)
* **Computation**: CUDA (GPU support)
* **Visualization**: Matplotlib, Scikit-learn (Confusion Matrix)
* **Environment**: VS Code, Jupyter Notebook

## 📂 Project Structure
```text
├── Assignment_1/
│   ├── Assignment_1_template_with_split.ipynb  # Main Project Notebook
│   ├── cnn_3layer_bone_break_classification.pt # Saved Model Weights
├── Dataset/                                    # (Not included in repo due to size)
├── .gitignore                                  # Git configuration
└── README.md                                   # Project Documentation

## 🧠 Model Architecture
The project uses a custom 3-Layer Convolutional Neural Network (CNN) designed for medical image classification.

**Architecture Breakdown:**
* **Input Layer:** Accepts resized X-ray images (224x224 pixels, RGB).
* **Convolutional Blocks (x3):**
    1.  **Layer 1:** 16 Filters | Kernel 3x3 | ReLU Activation | MaxPolling (2x2)
    2.  **Layer 2:** 32 Filters | Kernel 3x3 | ReLU Activation | MaxPolling (2x2)
    3.  **Layer 3:** 64 Filters | Kernel 3x3 | ReLU Activation | MaxPolling (2x2)
* **Flatten Layer:** Converts 2D feature maps into a 1D vector.
* **Fully Connected Layers:**
    * **Dense Layer 1:** 50,176 Input Neurons $\rightarrow$ 128 Output Neurons.
    * **Output Layer:** 128 Neurons $\rightarrow$ Number of Classes (Fracture Types).

**Hyperparameters:**
* **Optimizer:** Adam (`lr=0.001`, `weight_decay=1e-4`)
* **Loss Function:** CrossEntropyLoss
* **Batch Size:** 32
* **Epochs:** 50

---

## 📈 Results & Analysis

### 1. Training vs. Validation Performance
The model was trained over 50 epochs.
* **Training Loss:** Consistently decreased, indicating the model was learning features effectively.
* **Validation Accuracy:** Peaked at approximately **70%**, though fluctuations occurred due to the small and imbalanced validation set.

![Loss and Accuracy Curves](images/loss_curve.png)
*(Replace 'images/loss_curve.png' with your actual file path)*

### 2. Confusion Matrix
The confusion matrix highlights which fracture types were most easily confused.
* **Class 8 (Humerus/Similar):** The model showed a bias towards this class due to its high prevalence in the validation split.

![Confusion Matrix](images/confusion_matrix.png)
*(Replace 'images/confusion_matrix.png' with your actual file path)*

### 3. Sample Predictions
Below are sample predictions from the validation set. Green titles indicate correct predictions, while Red indicates errors.

![Sample Predictions](images/predictions.png)
*(Replace 'images/predictions.png' with your actual file path)*

---

## ⚙️ How to Run

### Prerequisites
Ensure you have **Python 3.8+** installed. You will also need the following libraries:
* PyTorch & Torchvision
* Matplotlib
* Scikit-learn
* Jupyter

### Installation Steps

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/YourUsername/bone-fracture-detection-cnn.git](https://github.com/YourUsername/bone-fracture-detection-cnn.git)
    cd bone-fracture-detection-cnn
    ```

2.  **Install Dependencies**
    ```bash
    pip install torch torchvision matplotlib scikit-learn numpy
    ```

3.  **Dataset Setup**
    * Download the **Bone Fracture Classification Dataset**.
    * Create a folder named `Dataset` in the root directory.
    * Extract the files so the path looks like: `Dataset/BoneBreakClassification/[Fracture_Folders]`.

4.  **Run the Notebook**
    * Open `Assignment_1_template_with_split.ipynb` in VS Code or Jupyter Notebook.
    * Select your Python kernel (ensure it matches the environment where you installed libraries).
    * Click **Run All** to train the model and generate results.