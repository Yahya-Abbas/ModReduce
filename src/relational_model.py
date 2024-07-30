from util._util import * 
from src.base_model import *

class relational_model(base_model):
    def __init__(self, trainer):
        super().__init__(trainer)
        # Create variables to track the data
        self.model_type = trainer.sm # WARNING take care same sm for all
        self.identifier = "relational"
        self.knoweldge_type = trainer.rk_type
        self.create_common(trainer)

        
        # Random data used in creating embed layer
        rnd_data = torch.randn(2, 3, 32, 32).cuda()
        if(trainer.rk_type == 'rkd'):
            self.RELATIONAL_LOSS = RKDLoss()
            self.relational_loss_weight  = 1
        elif(trainer.rk_type == 'crd'):
            trainer.t_model.eval()
            self.model.eval()
            feat_t, _ = trainer.t_model(rnd_data, is_feat=True)
            feat_s, _ = self.model(rnd_data, is_feat=True)
            trainer.opt.s_dim = feat_s[-1].shape[1]
            trainer.opt.t_dim = feat_t[-1].shape[1]
            trainer.opt.n_data = trainer.n_data
            self.RELATIONAL_LOSS = CRDLoss(trainer.opt, device=trainer.device)
            self.trainable_list.append(self.RELATIONAL_LOSS.embed_s)
            self.trainable_list.append(self.RELATIONAL_LOSS.embed_t)
            self.relational_loss_weight  = 0.8
            # self.HKD_weight = 0 # This value overwrites that of the base class (to be like CRD repo)
        

        self.optimizer = torch.optim.SGD(self.trainable_list.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        if(trainer.resume):
            state = torch.load(self.ckp_path)
            self.model.load_state_dict(state['state_dict'])
            self.optimizer.load_state_dict(state['optimizer'])
            # ERROR: This resuming is not considering the embed 
            # layers, please check.


    def calc_offline_loss(self, trainer):
        # Zero the optimizer gradient 
        self.optimizer.zero_grad()

        if(trainer.rk_type == 'rkd'):
            self.predictions = self.model(trainer.inputs)
            self.relational_loss = self.RELATIONAL_LOSS(self.predictions, trainer.t_predictions) * self.relational_loss_weight
        elif(trainer.rk_type == 'crd'):
            self.feat, self.predictions = self.model(trainer.inputs, is_feat = True)
            self.relational_loss = self.RELATIONAL_LOSS(self.feat[-1], trainer.feat_t[-1], trainer.index, trainer.contrast_idx) \
                * self.relational_loss_weight
        
        self.classification_loss = self.CL(self.predictions, trainer.labels) * self.CL_weight
        self.kl_loss = self.HKD(self.predictions, trainer.t_predictions) * self.HKD_weight

        # Weights not normalized yet here
        self.offline_loss_no_kl = self.classification_loss + self.kl_loss + self.relational_loss 

        if(trainer.train_online == True):
            # Weights not normalized yet here
            self.offline_loss_no_kl = self.classification_loss + self.relational_loss 
            self.multiplier = 1 # Equivalent to the mystery factor
            self.loss_weights_no_kl = self.CL_weight + self.relational_loss_weight
        else:
            # ERROR: I am not sure why are you multiplying by 100 after normalizing the weights!
            # also it's 100 here and 10 in hinton which is incosistent
            # Weights should be normalized here in this case
            # \ is just a line break, don't remove
            self.total_loss = (self.classification_loss + self.kl_loss +self.relational_loss ) \
            / (self.CL_weight + self.HKD_weight + self.relational_loss_weight) * 1

                