import torch
import torch.nn as nn
import torch.nn.functional as func
from collections import OrderedDict
from typing import Tuple, Optional, List, Dict
from transformers import ViTForImageClassification


class DigitModel(nn.Module):
    """
    Model for benchmark experiment on Digits. 
    """
    def __init__(self, num_classes=10, **kwargs):
        super(DigitModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(128)
    
        self.fc1 = nn.Linear(6272, 2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_classes)


    def forward(self, x):
        x = func.relu(self.bn1(self.conv1(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn2(self.conv2(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn3(self.conv3(x)))

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.bn4(x)
        x = func.relu(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x2 = func.relu(x)

        x2 = self.fc3(x2)
        return x2,x


class DigitModel2(nn.Module):
    """
    Model for benchmark experiment on Digits.
    """

    def __init__(self, num_classes=10, **kwargs):
        super(DigitModel2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(6272, 2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = func.relu(self.bn1(self.conv1(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn2(self.conv2(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn3(self.conv3(x)))

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.bn4(x)
        x = func.relu(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x2 = func.relu(x)

        x2 = self.fc3(x2)
        return x2, x


class AlexNet(nn.Module):
    """
    used for DomainNet and Office-Caltech10
    """
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.backbone = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
                ('bn2', nn.BatchNorm2d(192)),
                ('relu2', nn.ReLU(inplace=True)),
                ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
                ('bn3', nn.BatchNorm2d(384)),
                ('relu3', nn.ReLU(inplace=True)),

                ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
                ('bn4', nn.BatchNorm2d(256)),
                ('relu4', nn.ReLU(inplace=True)),

                ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ('bn5', nn.BatchNorm2d(256)),
                ('relu5', nn.ReLU(inplace=True)),
                ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
            ])
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.bottleneck = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(256 * 6 * 6, 4096)),
                ('bn6', nn.BatchNorm1d(4096)),
                ('relu6', nn.ReLU(inplace=True)),

                ('fc2', nn.Linear(4096, 4096)),
                ('bn7', nn.BatchNorm1d(4096)),
                ('relu7', nn.ReLU(inplace=True)),

            ])
        )

        self.head = nn.Linear(4096, num_classes)
        

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.bottleneck(x)
        x2 = self.head(x)
        return x2, x


    def get_parameters(self) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr_mult": 0.1},
            {"params": self.bottleneck.parameters(), "lr_mult": 0.1},
            {"params": self.head.parameters(), "lr_mult": 1.},
        ]
        return params


class ViTWithPrompts(nn.Module):
    """
    Vision Transformer (ViT) with a single set of tunable prompts.
    Matches AlexNet's structure with backbone, bottleneck (two layers), and head.
    """
    def __init__(self, num_classes=10, prompt_length=10, pretrained_model_name='google/vit-base-patch16-224'):
        super(ViTWithPrompts, self).__init__()
        self.backbone_model = ViTForImageClassification.from_pretrained(pretrained_model_name, num_labels=num_classes, ignore_mismatched_sizes=True)

        # Freeze the rest of the ViT model
        for param in self.backbone_model.parameters():
            param.requires_grad = False

        # Backbone: Prompts
        hidden_size = self.backbone_model.config.hidden_size
        self.backbone = nn.Parameter(torch.randn(prompt_length, hidden_size))  # Prompts

        # Bottleneck: Two-layer projection
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_size, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True)
        )

        # Head: Classifier
        self.head = nn.Linear(4096, num_classes)

    def forward(self, x):
        # Patch embedding and positional encoding
        embeddings = self.backbone_model.vit.embeddings(x)  # Shape: (batch_size, seq_len, hidden_size)

        # Add prompts
        batch_size = embeddings.size(0)
        prompts = self.backbone.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, prompt_length, hidden_size)
        combined_inputs = torch.cat([prompts, embeddings], dim=1)  # Shape: (batch_size, seq_len + prompt_length, hidden_size)

        # Pass through frozen encoder
        transformer_output = self.backbone_model.vit.encoder(combined_inputs)
        cls_output = transformer_output[:, 0, :]   # CLS token output

        # Bottleneck and head
        bottleneck_output = self.bottleneck(cls_output)
        logits = self.head(bottleneck_output)
        return logits, bottleneck_output

    def get_parameters(self) -> List[Dict]:
        """
        A parameter list which matches AlexNet's structure with backbone, bottleneck, and head.
        """
        params = [
            {"params": self.backbone.parameters(), "lr_mult": 0.1},  # Prompts
            {"params": self.bottleneck.parameters(), "lr_mult": 0.1},  # Bottleneck
            {"params": self.head.parameters(), "lr_mult": 1.0},  # Head
        ]
        return params
