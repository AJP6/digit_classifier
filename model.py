import torch
import torch.nn as nn 
import torch.nn.functional as F

IMAGE_PIXELS = 28 * 28 # =784

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        #init layers
        self.lin1 = nn.Linear(
            in_features=IMAGE_PIXELS, 
            out_features=512
        )
        self.lin2 = nn.Linear(
            in_features=512, 
            out_features=1028
        ) 
        self.lin3 = nn.Linear(
            in_features=1028, 
            out_features=512
        )
        self.final = nn.Linear(
            in_features=512, 
            out_features=10 
        )

    def forward(self, x):
       # shape (batch, 28, 28) -> (batch, 784)
       data = x.view(-1, IMAGE_PIXELS) 
       data = self.lin1(data)
       data =  F.relu(data)
       data = self.lin2(data)
       data =  F.relu(data)
       data = self.lin3(data)
       data =  F.relu(data)
       return self.final(data)