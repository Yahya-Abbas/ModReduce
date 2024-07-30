from util._util import * 
from src.base_model import *

class feature_model(base_model):
    def __init__(self, trainer):
        super().__init__(trainer)
        # Create variables to track the data
        self.model_type = trainer.sm # WARNING take care same sm for all
        self.identifier = "feature"
        self.knoweldge_type = trainer.fk_type
        self.create_common(trainer)


        # Create dummy feature outputs to create
        # regressor or self_attention layers
        rnd_data = torch.randn(2, 3, 32, 32).cuda()
        trainer.t_model.eval()
        self.model.eval()
        feat_t, _ = trainer.t_model(rnd_data, is_feat=True)
        feat_s, _ = self.model(rnd_data, is_feat=True)
        

        if(trainer.fk_type == 'fitnets'): 
            # Creating regressor for fitnets 
            hint_layer = 1
            self.regressor = ConvReg(feat_s[hint_layer].shape, feat_t[hint_layer].shape).to(trainer.device)
            self.FEATURE_LOSS = HintLoss()
            # Weights for different losses
            self.feature_loss_weight = 100
            self.trainable_list.append(self.regressor)
        
        elif(trainer.fk_type == 'semckd'):
            s_n = [f.shape[1] for f in feat_s[1:-1]]
            t_n = [f.shape[1] for f in feat_t[1:-1]]
            self.FEATURE_LOSS = SemCKDLoss()
            self.self_attention = SelfA(len(feat_s)-2, len(feat_t)-2, trainer.batch_size, s_n, t_n).cuda()
            # Weights for different losses
            self.feature_loss_weight = 400
            self.trainable_list.append(self.self_attention)


        self.optimizer = torch.optim.SGD(self.trainable_list.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        if(trainer.resume):
            state = torch.load(self.ckp_path)
            self.model.load_state_dict(state['state_dict'])
            self.optimizer.load_state_dict(state['optimizer'])
            # ERROR: This resuming is not considerig the self attention or the hint  
            # layers, please check.


    def calc_offline_loss(self, trainer):
        # Zero the optimizer gradient 
        self.optimizer.zero_grad()
        
        # Calculate student predictions
        self.feat, self.predictions = self.model(trainer.inputs, is_feat = True)

        # Calculate Classification loss and Hinton loss.
        # if you want to zero these values set their weights in self to zero
        self.classification_loss = self.CL(self.predictions, trainer.labels) * self.CL_weight
        self.kl_loss = self.HKD(self.predictions, trainer.t_predictions) * self.HKD_weight


        if(trainer.fk_type == 'fitnets'):
            f_s, f_t = self.regressor(self.feat[self.hint_layer], trainer.feat_t[self.hint_layer])
            self.feature_loss = self.FEATURE_LOSS(f_s, f_t) * self.feature_loss_weight
        elif(trainer.fk_type == 'semckd'):
            s_value, f_target, weight = self.self_attention(self.feat[1:-1], trainer.feat_t[1:-1])
            self.feature_loss = self.FEATURE_LOSS(s_value, f_target, weight) * self.feature_loss_weight
        

        if(trainer.train_online == True):
            # Weights not normalized yet here
            self.offline_loss_no_kl = self.classification_loss + self.feature_loss
            self.multiplier = 100 # Equivalent to the mystery factor
            self.loss_weights_no_kl = self.CL_weight + self.feature_loss_weight
        else:
            # ERROR: I am not sure why are you multiplying by 100 after normalizing the weights!
            # also it's 100 here and 10 in hinton which is incosistent
            # Weights should be normalized here in this case
            # \ is just a line break, don't remove
            self.total_loss = (self.classification_loss + self.kl_loss + self.feature_loss) \
            / (self.CL_weight + self.HKD_weight + self.feature_loss_weight) * 100
        
