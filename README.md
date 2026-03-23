# fall-detection-cnn-lstm-al
A fall detection model combining CNN, LSTM and Active Learning for video/sequential data. Efficiently identifies human falls with minimal labeled data by leveraging active learning to select informative samples.
Fall Detection Training and Usage Guide (Active Learning Version)
This project implements video-level binary classification (fall / non-fall) based on CNN + LSTM, and introduces an active learning mechanism to improve model performance by iteratively mining hard samples. The code uses the PyTorch framework to extract spatial features from video frame sequences and model temporal dependencies, which can be used for fall event detection in surveillance videos.
Features
Uses pre-trained ResNet18 to extract spatial features of each frame, saving training time and improving performance.
Captures temporal dependencies between frames through LSTM, suitable for video classification tasks.
Active learning strategy: After each training round, the current best model identifies misclassified samples in the test set, moves them to the training set, and randomly extracts an equal number of samples from the training set to supplement the test set, maintaining a fixed 6:4 ratio. After multiple iterations, the model focuses more on hard samples and achieves stronger generalization ability.
Automatic dataset splitting: No manual splitting of training/validation sets required. The program reads all samples from data/fall and data/not_fall, and randomly splits them into training and test sets at a 6:4 ratio (fixed random seed ensures reproducibility).
Global optimal model saving: During training, not only the best model of each round (best_cnn_lstm_roundX.pth) is saved, but also the global optimal model with the highest validation accuracy across all rounds (best_cnn_lstm_global.pth), facilitating final deployment.
Adaptive learning rate adjustment (ReduceLROnPlateau) to avoid local optima.
Complete training/validation process with GPU acceleration support (automatic device selection).
Provides a matching inference script agent.py that encapsulates model loading, preprocessing, and result parsing for easy integration.
Environment Requirements
Python 3.7+
PyTorch 1.7+ (1.10+ recommended)
torchvision (matching PyTorch version)
numpy
Pillow
tqdm
Install Dependencies
pip install torch torchvision numpy pillow tqdmIf using GPU, ensure the installed PyTorch version matches your CUDA version. Refer to the PyTorch official website (https://pytorch.org/) for details.
Dataset Preparation
Directory Structure Specification
Organize the dataset according to the following directory structure:data/├── fall/ # Fall category│ ├── video_001/ # Frame images of video 001│ │ ├── frame_0001.jpg│ │ ├── frame_0002.jpg│ │ ├── ...│ ├── video_002/│ └── ...└── not_fall/ # Non-fall category├── video_101/└── ...
Category names: Folder names serve as category labels (fall, not_fall), and the program automatically maps labels (fall=0, not_fall=1).
Video folders: All frames of each video are stored in an independent folder with arbitrary naming.
Frame images: Stored in video folders, supporting .jpg, .png, .jpeg, .bmp formats.
Frame Image Requirements
Image naming should allow sequential sorting (e.g., frame_0001.jpg, frame_0002.jpg). The program uses sorted() to sort filenames, so fixed-digit numeric numbering (e.g., %06d) is recommended.
All images must be RGB three-channel; they will be automatically converted to RGB during training.
Image size is unrestricted; they will be uniformly resized to FRAME_SIZE × FRAME_SIZE (224×224 by default) during training.
Dataset Splitting Instructions
The program automatically collects all video folders meeting the frame count requirement from data/fall and data/not_fall, then randomly splits them into training and test sets at a 6:4 ratio (fixed random seed seed=42 ensures reproducibility). Therefore, you do not need to manually create train and val directories.
Data Volume Recommendations
Each category should contain at least dozens of videos to ensure model generalization ability.
If data volume is limited, consider adding data augmentation (not included in this project, can be added as needed).
Configuration File and Hyperparameters
All hyperparameters are defined in the "Hyperparameters" section at the beginning of the code and can be modified as needed.
Hyperparameter Explanation
| Parameter | Default Value | Description || SEQUENCE_LENGTH | 16 | Number of frames uniformly sampled per video. Videos with fewer frames than this value will be skipped. || FRAME_SIZE | 224 | Resize dimension of input images (width = height). || BATCH_SIZE | 8 | Batch size, adjust according to GPU memory. || EPOCHS | 50 | Number of training iterations within each active learning round. || LEARNING_RATE | 1e-4 | Initial learning rate. || NUM_CLASSES | 2 | Number of classes (fall / non-fall). || ACTIVE_LEARNING_ROUNDS | 3 | Number of active learning rounds. Dataset splitting is updated after each round. || TRAIN_RATIO | 0.6 | Training set ratio (the rest are test set). || DEVICE | Auto-selection | Prioritizes cuda, otherwise uses cpu. |
Dataset Path
The default data root directory in the code is DATA_ROOT = 'data', meaning data is read from data/fall and data/not_fall. To modify it, change:DATA_ROOT = 'your_data_path'
Model Training
Start Training
After ensuring the dataset is ready, run directly:python train.pyThe training process will:
Collect all samples and randomly split them into training and test sets at a 6:4 ratio.
Enter the active learning loop, performing EPOCHS training iterations within each round.
After each training round, use the best model of that round to identify misclassified samples in the test set and update dataset splitting.
Save the best model of each round (best_cnn_lstm_roundX.pth) and the global optimal model (best_cnn_lstm_global.pth) during each round.
Print training/validation loss and accuracy for each round, and display the number of misclassified samples in the current round and the updated dataset size.
Training Process Monitoring
Uses tqdm to display progress bars, allowing intuitive viewing of loss and cumulative accuracy for the current batch.
Prints training loss, accuracy, validation loss, and accuracy after each epoch.
Displays the number of misclassified samples and new training/test set sizes after each active learning round.
Model Saving
Best model per round: best_cnn_lstm_round1.pth, best_cnn_lstm_round2.pth, etc.
Global best model: best_cnn_lstm_global.pth (model with the highest validation accuracy across all rounds).
The model saves only the state_dict (excluding network structure), so the same model class needs to be redefined during inference.
Resume Training
If you want to resume training from a checkpoint, you can load weights after initializing the model and continue iteration. Since active learning involves dataset changes, resuming training requires manual adjustment of rounds and dataset status; running the script completely is recommended.
Detailed Model Structure
The CNNLSTM model consists of four main parts:
CNN Feature ExtractorUses pre-trained ResNet18 (convolutional part after removing the final fully connected layer and average pooling layer), outputting feature maps with shape (batch, 512, 1, 1). We use squeeze to convert it to (batch, 512).resnet = models.resnet18(pretrained=True)modules = list(resnet.children())[:-1] # Remove the final fully connected layerself.cnn = nn.Sequential(*modules)
Feature Mapping LayerMaps 512-dimensional CNN features to 256 dimensions for LSTM usage.self.fc_in = nn.Linear(512, 256)
LSTM Temporal Modeling
Input size: 256
Hidden layer size: 512
Number of layers: 2
Bidirectional: No (unidirectional)
Dropout: 0.3 (only effective when layers > 1)
batch_first=True
The LSTM receives sequences of shape (batch, seq_len, 256) and outputs hidden states for all time steps (batch, seq_len, 512) as well as hidden state and cell state for the last time step.
Classification Layer
Takes the output of the last time step of LSTM (lstm_out[:, -1, :]) and outputs logits (2 dimensions) through a fully connected layer.
self.fc_out = nn.Linear(lstm_hidden_size, num_classes)
Forward Propagation Process
Input x shape: (batch, seq_len, C, H, W)
Merge batch and seq_len: (batch * seq_len, C, H, W)
Pass through CNN: (batch * seq_len, 512, 1, 1) → compressed to (batch * seq_len, 512)
Pass through fc_in: (batch * seq_len, 256)
Restore sequence dimension: (batch, seq_len, 256)
Input to LSTM: (batch, seq_len, 256) → output (batch, seq_len, 512)
Take the last time step: (batch, 512)
Pass through fc_out: (batch, num_classes)
Evaluation and Inference
After training, you can use the matching inference script agent.py (in the same directory) to predict new video frame folders. This script loads the global optimal model best_cnn_lstm_global.pth by default.
Using the Matching Inference Script agent.py
agent.py provides the FallDetectionAgent class with the following usage:from agent import FallDetectionAgent
Initialize agent (automatically loads best_cnn_lstm_global.pth)
agent = FallDetectionAgent(model_dir='.') # model_dir is the directory where the model is located
Predict video frame folder
result = agent.predict("path/to/video_frames")
Print formatted results
agent.print_result(result, verbose=True)
Prediction Result Format
result is a dictionary containing the following fields:
status: "success" or "error"
video_path: Input path
frame_count: Total number of frames in the folder
sampled_frames: Actual number of sampled frames (equal to SEQUENCE_LENGTH)
prediction:
class: Predicted class index (0 for fall, 1 for non-fall)
label: Chinese class label ("跌倒"/"未跌倒")
name: English class name ("fall"/"not_fall")
confidence: Confidence score
probabilities: Probabilities of all classes
description: Descriptive information
emoji: Emoji symbol
timestamp: Prediction timestamp
Using the Model for Prediction Independently
If you do not want to use the agent, you can load the model directly for prediction:import torchfrom PIL import Imageimport torchvision.transforms as transformsimport numpy as np
Define model (same as during training)
model = CNNLSTM(num_classes=2)model.load_state_dict(torch.load('best_cnn_lstm_global.pth'))model.eval()model.to(device)
Preprocessing function
transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])def predict_video(frames_folder):images = []for f in sorted(os.listdir(frames_folder)):if f.lower().endswith(('.jpg','.png')):img = Image.open(os.path.join(frames_folder, f)).convert('RGB')images.append(img)
Uniformly sample 16 frames
indices = np.linspace(0, len(images)-1, 16, dtype=int)sampled = [images[i] for i in indices]
Preprocessing
tensor_list = [transform(img) for img in sampled]clip = torch.stack(tensor_list).unsqueeze(0).to(device) # [1,16,C,H,W]with torch.no_grad():logits = model(clip)probs = torch.softmax(logits, dim=1)return probs.cpu().numpy()
Complete Process for Custom Datasets
Collect videos: Obtain original surveillance videos.
Extract frames: Use tools like FFmpeg to extract frames from videos at a fixed frame rate (e.g., 30fps) into images with unified naming (e.g., frame_%06d.jpg).
ffmpeg -i input.mp4 -vf fps=30 frames/frame_%06d.jpg
Organize directories: Place each video's frame folder into data/fall/ or data/not_fall/ according to its category.
Modify hyperparameters (optional): Adjust SEQUENCE_LENGTH, BATCH_SIZE, etc.
Run training: Execute python train.py directly.
Evaluate: After training, use the accuracy of the test set (internally reserved by the program) as a reference, or test new samples with the agent.
Deploy: Use the trained global optimal model for inference.
Performance Optimization Suggestions
Increase batch size: If memory permits, increasing the batch size can accelerate training and potentially improve stability.
Learning rate scheduling: ReduceLROnPlateau is already used, which automatically adjusts according to validation loss.
Early stopping: Monitor validation loss and terminate training early if it does not decrease for consecutive epochs (can be added manually).
Mixed precision training: Using torch.cuda.amp can reduce memory usage and accelerate training (requires PyTorch 1.6+), which can be integrated manually.
Multi-GPU training: The model can be wrapped with nn.DataParallel, but attention should be paid to batch allocation.
Citation and References
PyTorch Official Documentation (https://pytorch.org/docs/stable/index.html)ResNet: Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385)LSTM: Long Short-Term Memory (https://www.bioinf.jku.at/publications/older/2604.pdf)
This project code is for learning and research purposes only. Please evaluate independently for commercial use.This code was designed by Jayson SHI. For any questions regarding this code, please contact: 2040420809@qq.com
