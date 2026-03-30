import timm
import torch.nn as nn

class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes, model_name = 'vit_base_patch16_224'):
        super(PlantDiseaseModel, self).__init__()
        
        # Load a pre-trained ViT model
        self.model = timm.create_model(model_name, pretrained=True)

        # Replace the classifier head to match the number of classes in our dataset
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        return self.model(x)