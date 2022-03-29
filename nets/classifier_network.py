import torch.nn as nn
import torch.nn.functional as F


class ClassificationNetwork(nn.Module):
    """
        A simple Network for classification.
    """

    def __init__(self, backbone, num_cats) -> None:
        super().__init__()

        self.num_cats = num_cats

        self.backbone = backbone

        # Classification Layer
        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features, self.num_cats)

        # self.act_classification = nn.Sigmoid()

    def forward(self, _in):
        """
            Forwarding input to output
        """
        out = self.backbone(_in)

        return out

    def embedding(self):
        """
            This function replace two final layer with nn.Identify to extract 
            a embedding for input
        """
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()
