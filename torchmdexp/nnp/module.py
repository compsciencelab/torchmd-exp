import torch
from torch import nn

from .models.model import create_model, load_model
from torchmdexp.utils.parsing import set_hparams

class NNP(nn.Module):
    def __init__(self, hparams, mean=None, std=None):
        super(NNP, self).__init__()

        self.hparams = set_hparams(hparams)        
        
        if self.hparams.load_model:
            self.model = load_model(self.hparams.load_model, args=self.hparams)
        else:
            self.model = create_model(self.hparams, mean, std)

    def forward(self, z, pos, batch=None, q=None, s=None):
        return self.model(z, pos, batch=batch, q=q, s=s)