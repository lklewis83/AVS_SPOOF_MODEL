# AVS SPOOF MODEL
Tensorflow Audio Spoof Detection Model Trained with AVS Spoof 2019 Dataset

## Introduction
This project focuses on detecting audio spoofs using a deep learning model trained on the AVSSpoof 2019 dataset. The model identifies bonafide and spoofed speech samples, including text-to-speech (TTS) and voice conversion (VC) attacks.

## Data Preprocessing
I utilized the [AVSSpoof 2019 dataset](https://www.kaggle.com/c/asvspoof-2019), which consists of bonafide and spoofed speech samples. The spoofed data includes text-to-speech (TTS) and voice conversion (VC) attacks, categorized under different attack types (A01–A19). The preprocessing steps included:
- **Feature Extraction**: Extracted Mel-Frequency Cepstral Coefficients (MFCC) features from each audio sample.
- **Normalization**: Applied feature normalization to standardize the extracted MFCCs.
- **Reshaping**: Adjusted the feature shape to be compatible with the input layer of the neural network.

## Model Architecture
The neural network was built using TensorFlow and consisted of the following layers:
- **Input Layer**: 200 neurons
- **Hidden Layers**: Two layers with 128 and 64 neurons, respectively, both with ReLU activations.
- **Output Layer**: A single neuron with a sigmoid activation function.
- **Optimizer**: Adam
- **Batch Size**: 32
- **Epochs**: 50

## Training and Cross-Validation
The dataset was split into training and validation sets using k-fold cross-validation (k=5). The best-performing model was selected from the fourth fold based on validation performance.

## Cross-Validation Performance Results

| Metric                 | Value  |
|------------------------|--------|
| Best Model (Fold 4) AUC| 1.0000 |
| Average Accuracy       | 0.9981 |
| Average Precision      | 0.9996 |
| Average Recall         | 0.9983 |
| Average AUC            | 1.0000 |

## Evaluation on AVSSpoof Test Set

| Metric     | Value |
|------------|-------|
| Accuracy   | 0.7073|
| Precision  | 0.9620|
| Recall     | 0.7013|
| AUC        | 0.8048|

## Files in this Repository

- **AVSSpoof_Model_ORG.ipynb**: This Jupyter notebook contains the original model code used for training the audio spoofing detection model.
- **AVSSpoof_FUNCTION.ipynb**: A notebook for applying the pre-trained model to evaluation data to validate its performance.
- **FUNCTION_TTS.ipynb**: This notebook demonstrates how to use text-to-speech technology to audibly output the results of the model using gTTS.
- **README.md**: The guide documenting the project's purpose, structure, and usage instructions.
- **audio_spoofing_model.keras**: The trained Keras model that can be loaded to predict new data or further refine the model.

## Converting Model Results into Audio with gTTS

To make the results of the audio spoofing model more accessible, especially for visually impaired users, the results can be converted into spoken audio using the gTTS (Google Text-to-Speech) library. gTTS is a Python library and CLI tool to interface with Google Translate's text-to-speech API. It converts text into spoken audio without requiring an authentication key, which simplifies sharing and reusing the model. You can use gTTS as demonstrated in the `FUNCTION_TTS.ipynb` notebook, providing an effective way to hear the results directly.

## ASV Spoof Takeaway
This study demonstrates the feasibility of using TensorFlow for audio spoof detection. While the model achieved near-perfect performance during training, its evaluation performance suggests potential room for improvement in generalization. Future work includes optimizing feature extraction, addressing output scaling issues, and refining classification thresholds for improved real-world accuracy.
