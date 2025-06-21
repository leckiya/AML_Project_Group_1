import torch
import torch.nn as nn
import timm
from efficientnet_pytorch import EfficientNet

class NutrientPredictor(nn.Module):
    def __init__(self, efficientnet_version='b0', num_outputs=5, pretrained=True):
        """
        EfficientNet-based model for predicting food nutrients
        
        Args:
            efficientnet_version (str): EfficientNet version to use (b0, b1, b2, etc.)
            num_outputs (int): Number of nutrients to predict
            pretrained (bool): Whether to use pretrained weights
        """
        super(NutrientPredictor, self).__init__()
        
        model_name = f'efficientnet-{efficientnet_version}'
        if pretrained:
            self.backbone = EfficientNet.from_pretrained(model_name)
        else:
            self.backbone = EfficientNet.from_name(model_name)
            
        self._fc_features = self.backbone._fc.in_features
        
        self.backbone._fc = nn.Identity()
        
        self.regressor = nn.Sequential(
            nn.Linear(self._fc_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_outputs)
            

            # nn.Linear(self._fc_features, 512),
            # nn.BatchNorm1d(512),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.3),
            # nn.Linear(512, 128),
            # nn.BatchNorm1d(128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.2),
            # nn.Linear(128, num_outputs)
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input image tensor of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Output tensor of shape [B, num_outputs]
        """
        features = self.backbone(x)
        
        outputs = self.regressor(features)
        
        return outputs


def get_model(efficientnet_version='b0', num_outputs=5, pretrained=True):
    """
    Create and initialize a NutrientPredictor model
    
    Args:
        efficientnet_version (str): EfficientNet version to use
        num_outputs (int): Number of nutrients to predict
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        NutrientPredictor: The initialized model
    """
    model = NutrientPredictor(
        efficientnet_version=efficientnet_version,
        num_outputs=num_outputs,
        pretrained=pretrained
    )
    
    return model 

def get_v2_model(efficientnet_version='s', num_outputs=5, pretrained=True):
    model = timm.create_model(f'efficientnetv2_{efficientnet_version}', pretrained=pretrained)
    model.classifier = nn.Linear(model.classifier.in_features, num_outputs)
    return model
