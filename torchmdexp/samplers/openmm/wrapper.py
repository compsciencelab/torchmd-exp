import torch as pt


# Create a wrapper
class Wrapper(pt.nn.Module):

    def __init__(self, model, elements):
        super().__init__()

        self.model = model
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.register_buffer('elements', elements)

    def forward(self, positions):

        positions = positions * 10.0 # nm --> A
        energy = self.model(self.elements, positions)[0]
        energy = energy.flatten() * 4.184 # kcal/mol --> kJ/mol

        return energy
