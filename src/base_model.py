from util._util import *
from util.loss import  RKDLoss, HintLoss, KLD, SemCKDLoss
import matplotlib.pyplot as plt

class base_model():
    def __init__(self, trainer):
        self.model_type = None
        self.model = None
        self.max_validation_acc = 0
        self.training_loss_arr = []
        self.training_acc_arr = []
        self.validation_loss_arr = []
        self.validation_accuracy_arr = []
        self.lr = 0.05
        # General loss functions that are used by different knoweldge types
        self.CL = nn.CrossEntropyLoss()
        self.HKD = KLD(temperature = 4.0)
        # General weights for the main two losses
        self.CL_weight = 1
        self.HKD_weight = 1
        
        
    def create_common(self, trainer):
        # Create Paths
        self.best_weights_path = f'./student_models/runIdentifiers_Index_{trainer.run_index}_onlineType_{trainer.ol_type}_tmType_{trainer.tm_type}_smType_{trainer.sm}_knoweldgeTypes_{trainer.hk_type}_{trainer.rk_type}_{trainer.fk_type}_best_{self.identifier}_studentModel.pth'


        self.ckp_path = f'./checkpoints/runIdentifiers_Index_{trainer.run_index}_onlineType_{trainer.ol_type}_tmType_{trainer.tm_type}_smType_{trainer.sm}_knoweldgeTypes_{trainer.hk_type}_{trainer.rk_type}_{trainer.fk_type}_checkpoint_{self.identifier}_studentModel.pth'



        # Create Model
        self.model = create_model(use_all_gpus = trainer.multi_gpus, device=trainer.device, model_type= self.model_type)

        # edit learning rate if model_type is within the chosen 3 from CRD
        if self.model_type in ['mobilenetv2', 'ShuffleV1', 'ShuffleV2']:
            self.lr = 0.01

        # create ema model for hinton student model, used in peer mean teacher (mutual learning)
        if(trainer.ol_type == 'pcl'):
            self.ema_model = create_model(use_all_gpus = trainer.multi_gpus, device=trainer.device, model_type= self.model_type)
            for param in self.ema_model.parameters():
                param.detach_()
            # ERROR: please check if the ema_model shall be added to an optimizer to be trained.
            # Yahya: No, The official implementation of temporal mean model (ema used in PCL) i found doesn't do this
        
        # Create The trainable list for all different known learning training (hinton it has the model only)
        self.trainable_list = nn.ModuleList([])
        self.trainable_list.append(self.model)
    
    def forward_epoch_start(self, trainer):
            # pcl specific condition
            if(trainer.ol_type == 'pcl' and trainer.train_online == True):
                self.ema_model.train()
            self.trainable_list.train()
            self.step_training_loss = np.array([])
            adjust_learning_rate(trainer.epoch, self.lr, trainer.lr_decay_rate, trainer.lr_decay_epochs, self.optimizer)
            self.top1 = AverageMeter()

    def calc_ema_logit(self, trainer):
        ema_model_out = self.ema_model(trainer.inputs)
        self.ema_logit = ema_model_out.detach().data

    def calc_total_loss_pcl(self, trainer):
        self.kl_loss = self.kl_loss * trainer.consistency_weight
        self.peer_mean_loss = 0
        for model in trainer.student_models_arr:
            if(model != self):
                self.peer_mean_loss += (trainer.PCL_KL(self.predictions, model.ema_logit) * trainer.consistency_weight)
        
        self.total_loss = (self.offline_loss_no_kl + self.kl_loss +self.peer_mean_loss) \
            / (self.loss_weights_no_kl + self.HKD_weight + trainer.consistency_weight) * self.multiplier
        # ERROR: I am not sure why are you multiplying by 100 after normalizing the weights!
        # also it's 100 here and 10 in hinton which is incosistent

    def calc_total_loss_no_pcl(self, trainer):
        self.offline_loss = (self.offline_loss_no_kl + self.kl_loss) \
            / (self.loss_weights_no_kl + self.HKD_weight) * self.multiplier
        
        self.online_loss = self.HKD(self.predictions, trainer.group_out)

        self.total_loss = (trainer.w_ol * self.online_loss) + (1 - trainer.w_ol) * self.offline_loss


    def back_propagate(self, trainer):
        self.total_loss.backward()
        self.optimizer.step()
        self.step_training_loss = np.append(self.step_training_loss, self.total_loss.item())    
        accurate_preds = accuracy(self.predictions, trainer.labels)
        self.top1.update(accurate_preds[0].item(), trainer.inputs.size(0))
        trainer.postfix[f'loss of {self.identifier} = '] = self.total_loss.item()

    def post_epoch_log(self, trainer):
        train_metric = self.top1.avg
        self.training_acc_arr.append(train_metric)
        self.training_loss_arr.append(np.mean(self.step_training_loss))
        print(f"Accuracy on training set for {self.identifier} loss model in epoch {trainer.epoch}: {train_metric:.3f}")

        # Validation accuracy
        eval_metric, eval_loss = eval_model(self.model, trainer.evalloader, trainer.device)
        print(f"Accuracy on evaluation set for student model with loss {self.identifier} :{eval_metric:.3f}")
        if(eval_metric > self.max_validation_acc):
            self.max_validation_acc = eval_metric
            torch.save(self.model.state_dict(), self.best_weights_path)
        
        # Save Checkpoint
        # WARNING: CHECKPOINTING IS NOT IMPLEMENTED HERE
        # if (epoch+1)%ckp_frequency==0:
        #     ckp_state = {
        #         'epoch': epoch+1,
        #         'state_dict': s_model_f.state_dict(),
        #         'acc': eval_metric_f,
        #         'best_acc': max_valid_acc_f,
        #         'optimizer': optimizer_f.state_dict(),
        #     }
        #     torch.save(ckp_state, ckp_PATH_f)

        self.validation_accuracy_arr.append(eval_metric)
        self.validation_loss_arr.append(eval_loss)    

    def update_ema_variables(self, trainer):
        update_ema_variables(self.model, self.ema_model, trainer.epoch, alpha=0.999)


    def create_plots(self, trainer):
        
        # Accuracy Plot
        acc_plot_path = f'./plots/runIdentifiers_index_{trainer.run_index}_onlineType_{trainer.ol_type}_tmType_{trainer.tm_type}_smType_{trainer.sm}_knoweldgeTypes_{trainer.hk_type}_{trainer.rk_type}_{trainer.fk_type}_studentModel_{self.identifier}_Accuracey.png'
        plt.figure()
        plt.plot(range(1, len(self.training_acc_arr)+1), self.training_acc_arr, 'r--')
        plt.plot(range(1, len(self.validation_accuracy_arr)+1), self.validation_accuracy_arr, 'b-')
        plt.legend(['Training Accuracy', 'Validation Accuracy'])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'{self.identifier} Student')
        plt.savefig(acc_plot_path)

        # Loss Plot
        loss_plot_path = f'./plots/runIdentifiers_index_{trainer.run_index}_onlineType_{trainer.ol_type}_tmType_{trainer.tm_type}_smType_{trainer.sm}_knoweldgeTypes_{trainer.hk_type}_{trainer.rk_type}_{trainer.fk_type}_studentModel_{self.identifier}_Loss.png'
        plt.figure()
        plt.plot(range(1, len(self.training_loss_arr)+1), self.training_loss_arr, 'r--')
        plt.plot(range(1, len(self.validation_loss_arr)+1), self.validation_loss_arr, 'b-')
        plt.legend(['Training Loss', 'Validation Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{self.identifier} Student')
        plt.savefig(loss_plot_path)
        
        # Evaluate
        self.model.load_state_dict(torch.load(self.best_weights_path))
        eval_metric, test_loss = eval_model(self.model, trainer.testloader, trainer.device)
        print(f"Best model accuracey on test set for student_model_{self.identifier}_{self.knoweldge_type} :{eval_metric:.3f}")