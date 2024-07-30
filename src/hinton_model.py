from util._util import * 
from src.base_model import *

class hinton_model(base_model):
    def __init__(self, trainer):
        super().__init__(trainer)
        self.model_type = trainer.sm # WARNING take care same sm for all
        self.identifier = "hinton"
        self.knoweldge_type = "response"
        self.create_common(trainer)
        self.optimizer = torch.optim.SGD(self.trainable_list.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        if(trainer.resume):
            state = torch.load(self.ckp_path)
            self.model.load_state_dict(state['state_dict'])
            self.optimizer.load_state_dict(state['optimizer'])


    def calc_offline_loss(self, trainer):
        # Zero the optimizer gradient 
        self.optimizer.zero_grad()
        self.predictions = self.model(trainer.inputs)

        
        self.classification_loss = self.CL(self.predictions, trainer.labels) * self.CL_weight
        self.kl_loss = self.HKD(self.predictions, trainer.t_predictions) * self.HKD_weight

        
        if(trainer.train_online == True):
            # Weights not normalized yet here
            self.offline_loss_no_kl = self.classification_loss
            self.multiplier = 10 # Equivalent to the mystery factor 
            self.loss_weights_no_kl = self.CL_weight 
        else:
            # ERROR: I am not sure why are you multiplying by 10 after normalizing the weights!
            # Weights should be normalized here in this case
            self.total_loss = (self.classification_loss  + self.kl_loss) / (self.CL_weight + self.HKD_weight) * 10

