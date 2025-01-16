# Playing Card Classifier

This project implements a deep learning model for classifying images of playing cards using PyTorch. It leverages the EfficientNet-B0 architecture as the base model, with custom modifications to suit the classification task. The dataset consists of labeled playing card images.

## Features

- **Dataset Loading**: A custom PyTorch `Dataset` class to handle playing card image data.
- **EfficientNet Backbone**: A pre-trained EfficientNet-B0 model for feature extraction.
- **Custom Classifier**: Fully connected layers to classify images into 53 card classes.
- **Training Loop**: Loss computation and optimization using cross-entropy loss and Adam optimizer.

## Requirements

Ensure the following dependencies are installed:

- Python 3.10+
- PyTorch 2.5.1+
- Torchvision 0.20.1+
- pandas 2.2.3
- numpy 2.2.1
- timm
- tqdm
- matplotlib

You can install the dependencies using:

```bash
pip install torch torchvision timm pandas numpy tqdm matplotlib
```

## Dataset

The dataset should be organized in the following structure:

```
/home/om/Downloads/archive_1/train/
    class_1/
        image1.jpg
        image2.jpg
        ...
    class_2/
        image1.jpg
        image2.jpg
        ...
    ...
```

Replace `/home/om/Downloads/archive_1/train/` with the actual path to your dataset.

## Usage

### 1. Check GPU Availability

```python
import torch
print(torch.cuda.is_available())
```

### 2. Install Required Packages

Ensure required libraries like pandas are installed:

```bash
pip install pandas
```

### 3. Dataset Initialization

```python
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

data_dir = '/home/om/Downloads/archive_1/train'
dataset = PlayingCardDataset(data_dir, transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### 4. Model Definition

```python
import timm
import torch.nn as nn

class SimpleCardClassifier(nn.Module):
    def __init__(self, num_classes=53):
        super(SimpleCardClassifier, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output

model = SimpleCardClassifier(num_classes=53)
```

### 5. Training Loop

```python
import torch.optim as optim
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for images, labels in dataloader:
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

### 6. Evaluate

Check the model's output shape and compute loss for a batch:

```python
example_out = model(images)
print(example_out.shape)
loss = criterion(example_out, labels)
print(loss.item())
```

## Key Functions

- **PlayingCardDataset**: Custom dataset class for loading images and labels.
- **EfficientNet-B0**: Pretrained model from the `timm` library, used for feature extraction.
- **Data Augmentation**: Resize and normalize images for input to the model.
- **Loss Function**: Cross-entropy loss for multi-class classification.

## Notes

- Ensure that the dataset path is correctly specified.
- The model uses 53 classes to represent all playing cards, including Jokers.
- Modify `batch_size` and `learning_rate` in the code to suit your hardware.
