# AVS SPOOF MODEL
Tensorflow Audio Spoof Detection Model Trained with AVS Spoof 2019 Dataset

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

## Singular Spoof Evaluation
To validate the decision-making process of the model, I tested individual spoofed audio samples. The model successfully classified most spoofed attacks, especially those generated using methods like Griffin-Lim (A11), waveform concatenation (A16), and vocoder-based (A13, A12). However, the model faced challenges with:
- **Successfully Classified**:
  - Griffin-Lim (A11)
  - Waveform Concatenation (A16)
  - Vocoder-based methods (A12, A13)
- **Challenging Cases**:
  - Voice Conversion Attacks (A17, A18)
  - TTS Vocoder-based methods (A14)
  - (Some predictions produced invalid values ≥ 1)

## ASV Spoof Takeaway
This study demonstrates the feasibility of using TensorFlow for audio spoof detection. While the model achieved near-perfect performance during training, its evaluation performance suggests potential room for improvement in generalization. Future work includes optimizing feature extraction, addressing output scaling issues, and refining classification thresholds for improved real-world accuracy.
