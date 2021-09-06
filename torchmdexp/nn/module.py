import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import mse_loss, l1_loss

from pytorch_lightning import LightningModule
from torchmdnet.models.model import create_model, load_model



class LNNP(LightningModule):
    def __init__(self, hparams, prior_model=None, mean=None, std=None):
        super(LNNP, self).__init__()
        self.save_hyperparameters(hparams)
        
        if self.hparams.load_model:
            self.model = load_model(self.hparams.load_model, device=self.hparams.device, 
                                    derivative=self.hparams.derivative
                                   )
            ckpt = torch.load(self.hparams.load_model, map_location="cpu")
            self.save_hyperparameters(ckpt["hyper_parameters"])
            
        else:
            self.model = create_model(self.hparams, prior_model, mean, std)
