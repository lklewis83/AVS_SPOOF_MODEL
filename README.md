# AVS Spoof Detection Model

## Overview

This project focuses on detecting **audio spoofs** using a deep learning model trained on the **Automatic Speaker Verification Spoofing and Countermeasures Challenge (ASVspoof) 2019 dataset**. The model identifies **bonafide** and **spoofed speech samples**, including **text-to-speech (TTS)** and **voice conversion (VC) attacks**.

### Features:

- **Deep Learning-Based Spoof Detection**
- **K-Fold Cross-Validation for Performance Evaluation**
- **Feature Extraction Using Mel-Frequency Cepstral Coefficients (MFCCs)**
- **Model Generalization & Real-World Application**
- **Text-to-Speech (TTS) Integration for Accessibility**

---

## Dataset

### AVSSpoof 2019 Dataset:

- The dataset consists of **bonafide and spoofed speech samples**.
- Spoofed data includes **text-to-speech (TTS) and voice conversion (VC) attacks** categorized under different attack types (**A01–A19**).

### Preprocessing Steps:

- **Feature Extraction:** Extracted **Mel-Frequency Cepstral Coefficients (MFCC)** features from each audio sample.
- **Normalization:** Standardized the extracted MFCCs for consistency.
- **Reshaping:** Adjusted the feature shape to match the input requirements of the neural network.

---

## Installation & Setup

### Requirements:

Ensure you have the required dependencies installed:

```bash
pip install tensorflow numpy scikit-learn librosa gtts
```

### Running the Model:

Clone the repository, update the file paths in the code files if necessary, and execute the following command:

```bash
python AVSSpoof_Model_ORG.ipynb
```

This Jupyter notebook contains the original model code used for training the audio spoofing detection model.

To apply the model for evaluation, use `AVSSpoof_FUNCTION.ipynb`. This notebook applies the pre-trained model to evaluation data to validate its performance.

---

## Model Architecture

### Neural Network Design:

- **Input Layer:** 200 neurons
- **Hidden Layers:**
  - Layer 1: **128 neurons** (ReLU activation)
  - Layer 2: **64 neurons** (ReLU activation)
- **Output Layer:** **1 neuron** (Sigmoid activation)
- **Optimizer:** Adam
- **Batch Size:** 32
- **Epochs:** 50

---

## Training & Cross-Validation

- **Callout:** This model was trained for binary classification (**bonafide vs. spoofed speech**).
- **5-Fold Cross-Validation** was implemented to evaluate performance.
- The best model (highest AUC) was saved as `audio_spoofing_model.keras`.

**Best Performance Achieved:** AUC **1.0000** on validation data.

---

## Evaluation & Results

### Cross-Validation Performance:

| Metric                  | Value  |
| ----------------------- | ------ |
| Best Model (Fold 4) AUC | 1.0000 |
| Average Accuracy        | 0.9981 |
| Average Precision       | 0.9996 |
| Average Recall          | 0.9983 |
| Average AUC             | 1.0000 |

### Evaluation on AVSSpoof Test Set:

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 0.7073 |
| Precision | 0.9620 |
| Recall    | 0.7013 |
| AUC       | 0.8048 |

### Singular Spoof Evaluation

To validate the model's decision-making process, we tested individual spoofed audio samples. The model successfully classified the most spoofed attacks, particularly those generated using Griffin-Lim (A11), waveform concatenation (A16), and vocoder-based methods (A13, A12). However, the model struggled with some voice conversion attacks (e.g., A17 and A18) and certain text-to-speech vocoder-based methods (e.g., A14), where it produced invalid predictions (values ≥1), indicating potential issues with output scaling.

#### AVS Spoof Detection Success Matrix

| Successfully Classified            | Challenging Cases                                |
| ---------------------------------- | ------------------------------------------------ |
| - Griffin-Lim (A11)                | - Voice Conversion Attacks (A17, A18)            |
| - Waveform Concatenation (A16)     | - TTS Vocoder-based methods (A14)                |
| - Vocoder-based methods (A12, A13) | - (Some predictions produced invalid values ≥ 1) |

---

## Converting Model Results into Audio with gTTS

To improve accessibility, model results can be **converted into speech** using **gTTS (Google Text-to-Speech)**.

- **Why?** This makes results accessible to **visually impaired users** or those preferring audio output.
- **How?** The `FUNCTION_TTS.ipynb` notebook demonstrates how to **convert text predictions into speech**.
- **No API Key Required:** gTTS provides free **text-to-speech conversion** without authentication.

---

## Files in this Repository

- **AVSSpoof\_Model\_ORG.ipynb** – This Jupyter notebook contains the original model code used for training the audio spoofing detection model.
- **AVSSpoof\_FUNCTION.ipynb** – A notebook for applying the pre-trained model to evaluation data to validate its performance.
- **FUNCTION\_TTS.ipynb** – This notebook demonstrates how to use text-to-speech technology to audibly output the results of the model using gTTS.
- **README.md** – The guide documenting the project's purpose, structure, and usage instructions.
- **audio\_spoofing\_model.keras** – The trained Keras model that can be loaded to predict new data or further refine the model.

---

## ASV Spoof Takeaway

This study demonstrates the feasibility of using TensorFlow for audio spoof detection. While the model achieved near-perfect performance during training, its evaluation performance suggests potential room for improvement in generalization. Future work includes optimizing feature extraction, addressing output scaling issues, and refining classification thresholds for improved real-world accuracy. Also, the consideration of alternative architectures, such as Convolutional Neural Networks (CNNs) or Transformer models, to further enhance model performance and robustness.

---

## Contributor

- **Lani Lewis**&#x20;

For inquiries, reach out at [Lani.k.Lewis2@gmail.com](mailto\:Lani.k.Lewis2@gmail.com)
