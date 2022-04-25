import torch

class Losses:
    
    def __init__(self,
                goal,
                fn_name = 'squared_error',
                y = 1.0,
                margin = 0.0,
                ):
        
        self.fn_name = fn_name
        self.goal = goal
        self.y = y
        self.margin = margin
    
    def __call__(self, value):
        if self.fn_name == 'squared_error':
            return self.squared_error(self.goal, value)
        elif self.fn_name == 'margin_ranking':
            return self.margin_ranking(value, self.goal, self.margin, self.y)
    
    def squared_error(self, a, b):
        device = b.device
        return (a-b) #.pow(2).sqrt()
    
    def margin_ranking(self, x1, x2=0.0, margin = 0.0, y=1.0):
        device = x1.device
        return torch.max(torch.tensor(0.0, device=device), y*(x1 - x2) + margin) 
