import torch
from pytorch_lightning import LightningModule
from torchmdnet.models.model import create_model, load_model
from torch_scatter import scatter



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
            self.model.to(self.hparams.device)
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.hparams.step_size, gamma=0.8
        )
        return [optimizer], [scheduler]
    
    
    def forward(self, z, pos, batch=None):
        return self.model(z, pos, batch=batch)

    def training_step(self, z, pos, batch):
        return self.step(z, pos, batch, "train")
    
    def validation_step(self, z, pos, batch):
        Upot = self.step(z, pos, batch, "val")
        return Upot.detach()

    def step(self, z, pos, batch, stage):
        
        #pos = pos.to(self.device).type(torch.float32).reshape(-1, 3)
        #batch = torch.arange(z.size(0), device=device).repeat_interleave(
        #    z.size(1)
        #)
        #z = z.reshape(-1).to(device)
        
        with torch.set_grad_enabled(stage == "train" or self.hparams.derivative):
            # TODO: the model doesn't necessarily need to return a derivative once
            # Union typing works under TorchScript (https://github.com/pytorch/pytorch/pull/53180)
            Upot, force = self(z, pos, batch)
        Upot = scatter(Upot, batch, dim=0, reduce='add')
        return Upot