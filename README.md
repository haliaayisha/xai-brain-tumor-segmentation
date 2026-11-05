# Brain Tumor Segmentation Using U-Net and Explainable AI

A deep learning project that explores brain tumor segmentation from MRI images using a custom U-Net architecture implemented in PyTorch. The project also incorporates Grad-CAM and Occlusion Sensitivity to visualize model activation patterns and investigate which image regions influence its predictions.

## Project Overview

Brain tumor segmentation involves identifying and outlining tumor regions in Magnetic Resonance Imaging (MRI) scans.

This project implements a custom U-Net architecture to perform pixel-level segmentation of brain MRI images. The model learns to predict tumor regions using corresponding ground-truth segmentation masks.

In addition to segmentation, two explainability techniques are explored:

- **Grad-CAM:** Visualizes activation patterns and gradient-based feature importance in an intermediate encoder layer.
- **Occlusion Sensitivity:** Examines changes in model predictions when individual image regions are masked.

The objective is to explore both segmentation performance and the interpretability of deep learning models in medical imaging.

## Key Features

- Brain MRI segmentation using a custom U-Net.
- Pixel-level tumor mask prediction.
- Encoder-decoder architecture with skip connections.
- Combined Binary Cross-Entropy and Dice Loss.
- Training, validation and testing pipeline.
- Dice coefficient and Intersection over Union (IoU) evaluation.
- Grad-CAM-based activation visualization.
- Occlusion Sensitivity analysis.
- Visualization of MRI images, ground-truth masks, predicted masks and explainability heatmaps.
- Model checkpoint saving and loading using PyTorch.

## Dataset

**LGG MRI Segmentation Dataset**

- Source: Mateusz Buda et al.
- Platform: Kaggle.
- Dataset identifier: `mateuszbuda/lgg-mri-segmentation`
- Images: Brain MRI scans.
- Annotations: Corresponding binary segmentation masks.
- Original image-mask pairs: 3,929.
- Input resolution: 256 × 256 pixels.
- Input channels: 3 (RGB).

The dataset contains MRI images and their corresponding segmentation masks organized into patient-specific folders.

For the final experimental pipeline, image-mask pairs with non-empty segmentation masks are retained.

This results in 1,373 image-mask pairs.

## Data Preprocessing

The preprocessing pipeline includes:

1. Loading MRI images and corresponding segmentation masks.
2. Matching images with their respective mask files.
3. Removing samples with empty segmentation masks.
4. Resizing images and masks to 256 × 256 pixels.
5. Normalizing image pixel values to the range [0, 1].
6. Converting images into PyTorch tensors.
7. Binarizing segmentation masks.
8. Dividing the retained samples into training, validation and testing sets.

### Dataset Split

| Dataset | Samples |
|---|---:|
| Training | 961 |
| Validation | 206 |
| Testing | 206 |
| **Total** | **1,373** |

The split uses an image-level random split with `random_state=42` and an approximate 70:15:15 distribution.

Note: This is not a patient-level split. Multiple slices from the same patient may appear across different subsets.

## Tech Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch |
| Image Processing | OpenCV |
| Numerical Computing | NumPy |
| Data Splitting | Scikit-learn |
| Visualization | Matplotlib |
| Explainable AI | Grad-CAM, Occlusion Sensitivity |
| Environment | Kaggle Notebooks |

## Model Architecture

A custom U-Net architecture is implemented using PyTorch.

The network follows an encoder-decoder structure with skip connections.

### Architecture Components

- **Encoder:** Extracts hierarchical spatial features through convolutional blocks and max pooling.
- **Bottleneck:** Learns deeper feature representations using additional convolutional layers.
- **Decoder:** Reconstructs spatial information using transposed convolutions.
- **Skip Connections:** Combine encoder and decoder feature maps to preserve spatial information.
- **Output Layer:** Generates a single-channel segmentation map.

Each convolutional block uses two convolutional layers, batch normalization and ReLU activation, followed by dropout.

The network uses four encoder stages and a bottleneck, with corresponding decoder stages.

## Training Configuration

| Parameter | Configuration |
|---|---|
| Architecture | Custom U-Net |
| Input Resolution | 256 × 256 |
| Input Channels | 3 |
| Output Channels | 1 |
| Batch Size | 8 |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | Dice + BCE Loss |
| BCE Weight | 0.5 |
| Dice Loss Weight | 0.5 |
| Training Epochs | 10 |
| Prediction Threshold (evaluation) | 0.5 |
| Framework | PyTorch |

The combined loss function is calculated as:

`Loss = 0.5 × BCE Loss + 0.5 × Dice Loss`

The best model checkpoint is selected based on validation loss.

## Evaluation Metrics

The model is evaluated using:

**1. Dice Coefficient**

Measures the overlap between the predicted segmentation mask and the ground-truth mask.

**2. Intersection over Union (IoU)**

Measures the intersection between the predicted and ground-truth regions relative to their union.

## Results

The following results were recorded during evaluation on the held-out test DataLoader.

| Metric | Recorded Result |
|---|---:|
| Mean Dice Score | 0.7783 |
| Mean IoU Score | 0.6728 |

The evaluation uses a prediction threshold of 0.5.

These are experimental results from an image-level random split and should not be interpreted as patient-independent generalization performance.

## Explainable AI

Two visualization techniques are implemented to explore the model's prediction behavior.

### 1. Grad-CAM

Gradient-weighted Class Activation Mapping (Grad-CAM) is used to visualize gradient-based activation patterns.

The implementation:

- Uses the `enc4` encoder layer as the target layer.
- Registers forward and backward hooks to capture feature activations and gradients.
- Computes gradient-based weights.
- Generates an activation heatmap.
- Resizes the heatmap to the original input resolution.
- Visualizes the heatmap and its overlay on the MRI image.

### 2. Occlusion Sensitivity

Occlusion Sensitivity is implemented to investigate how masking image regions affects model predictions.

The implementation:

- Uses a 16 × 16 pixel occlusion patch.
- Moves the patch across the image with a stride of 16 pixels.
- Replaces each selected image region with zeros.
- Measures the change in mean predicted probability.
- Constructs an occlusion heatmap from the resulting prediction differences.

These techniques provide exploratory visual insights into model behavior. They do not establish clinical reasoning, explain causality or guarantee model correctness.

## Model Checkpoint

The trained model weights are saved as:

`best_model.pth`

The checkpoint corresponds to the model selected using validation loss.

## Project Structure

```text
brain-tumor-segmentation/
│
├── brain-mri-segmentation.ipynb
├── README.md
└── best_model.pth
```

The model checkpoint should be included only if you intend to distribute the trained weights.

## How to Run

### Option 1: Kaggle Notebooks

1. Open the notebook in Kaggle.
2. Add the LGG MRI Segmentation Dataset.
3. Execute the notebook cells in sequence.
4. Run preprocessing and dataset splitting.
5. Train the U-Net model.
6. Evaluate the model using Dice and IoU.
7. Generate segmentation predictions.
8. Run Grad-CAM and Occlusion Sensitivity visualizations.

### Option 2: Google Colab

1. Upload the notebook to Google Colab.
2. Download or connect the LGG MRI Segmentation Dataset.
3. Update the dataset path.
4. Execute the notebook cells in sequence.

### Dependencies

```bash
pip install torch torchvision
pip install numpy opencv-python matplotlib scikit-learn
```

## Limitations and Future Improvements

- Implement patient-level dataset splitting to reduce potential data leakage.
- Evaluate the model on an independent patient-level test set.
- Explore additional U-Net configurations.
- Experiment with data augmentation.
- Compare different segmentation loss functions.
- Investigate additional explainability methods.
- Compare Grad-CAM and Occlusion Sensitivity across different MRI samples.
- Evaluate the stability of explainability visualizations.

## Disclaimer

This project is an academic deep learning implementation for experimentation and educational purposes. It is not intended for clinical diagnosis or medical decision-making.

## Author

**Halia Ayisha Abdul Khader**

M.Tech – Artificial Intelligence and Data Science

GitHub: [@haliaayisha](https://github.com/haliaayisha)
