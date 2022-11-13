import torch
from torch import nn
from torchvision import models

class VGGWrapper(nn.Module):
    def __init__(self, dim: int=128):
        super().__init__()
        self.vgg = models.vgg11(pretrained=False)
        self.vgg.avgpool = nn.AdaptiveAvgPool2d( (1, 1) )
        self.vgg.classifier = nn.Linear(512, dim)

    def forward(self, x):
        return self.vgg(x)

class EyeGaze(nn.Module):
    def __init__(self, *,
                 eye_feat: int=128, 
                 face_feat: int=256, 
                 mask_size: tuple=(36, 64), 
                 mask_feat: int=128):
        super().__init__()
        self.eye_vgg  = VGGWrapper(eye_feat)
        self.face_vgg = VGGWrapper(face_feat)
        # Position Embedding
        m, n = mask_size
        self.mask_emb = nn.Sequential(nn.Linear(m*n, 1024), nn.ReLU(), nn.Linear(1024, 128))
        # Feature Fusion Layer
        fusion_feat = eye_feat*2 + face_feat + mask_feat
        self.fusion = nn.Sequential(nn.Linear(fusion_feat, fusion_feat), nn.ReLU(), nn.Linear(fusion_feat, fusion_feat))
        # Classifier Layer
        self.classifier = nn.Linear(eye_feat*2 + face_feat + mask_feat, 9)
    
    def forward(self, left_eye, righteye, face, mask):
        left_eye = self.eye_vgg(left_eye)
        righteye = self.eye_vgg(righteye)
        face = self.face_vgg(face)
        mask_emb = self.mask_emb(mask)

        x = torch.cat([left_eye, righteye, face, mask_emb], dim=1)
        x = self.fusion(x)
        x = self.classifier(x)
        return x

# Define Models
EyeGaze = EyeGaze()
FacialExpression = models.vgg11_bn(pretrained=False)
FacialExpression.avgpool = nn.AdaptiveAvgPool2d((1, 1))
FacialExpression.classifier = nn.Linear(512, 3)
# Load pretrained weights into models
EyeGaze.load_state_dict(torch.load("./checkpoints/EyeGazeModel.pt", map_location="cpu"))
FacialExpression.load_state_dict(torch.load("./checkpoints/FacialExpression.pt", map_location="cpu"))
