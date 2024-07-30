# libraries import
from os import error
import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from util.loss import RKDLoss, HintLoss, KLD, SemCKDLoss
from util._util import create_model, create_data_loaders, eval_model, adjust_learning_rate, SelfA, mkdir_p
import sys
from util._util import ConvReg
from util._util import AverageMeter, accuracy
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from util.loss import CRDLoss
from torchsummary import summary

print_summary = True


# Choose which gpu to use, am using cuda:0 as the my default
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Mentioned to accelerate training
torch.backends.cudnn.benchmark = True



## adding arguments 
parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--index', type=int, default=0, help='index indicating current run')
parser.add_argument('--tm', type=str, default="resnet32x4", help='teacher model type',
                    choices=['resnet8', 'resnet20', 'resnet32', 'resnet56', 'resnet110','resnet8x4', 
                                'resnet32x4', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2','vgg8', 'vgg13',
                                'resnet50', 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
parser.add_argument('--sm', type=str, default="resnet8x4", help='student model type')
parser.add_argument('--gpu_id', type=str, default='0', help='use CRD for relational loss?')
parser.add_argument('--multi_gpus', type=int, default=0, help='use CRD for relational loss?')
parser.add_argument('--use_hinton', type=int, default=0, help="calcualte hinton loss or not?")
parser.add_argument('--use_online', type=int, default=1, help='decide whether to use online learning or not.')
parser.add_argument('--rk_type', default='crd', type=str, choices = ['rkd', 'crd'])
parser.add_argument('--fk_type', default='semckd', type=str, choices=['fitnets', 'semckd'])
parser.add_argument('--batch_size', default=64, type=int, help='Determines the batch size.')
parser.add_argument('--num_epochs', default=240, type=int, help='Determines the number of epochs.')
parser.add_argument('--ol', default='One', type=str, choices=['One', 'FC', 'W_Avg'], 
                    help='online learning method. i.e. ONE, FC, LinearWeightedAvg')
parser.add_argument('--resume', default=0, type=int, metavar='PATH',
                    help='Whether to resume training from checkpoint or not')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--feat_dim', type=int, default=128, help='feature dimension')
parser.add_argument('--nce_k', type=float, default=16384, help='number of negative samples for NCE')
parser.add_argument('--nce_t', type=float, default=0.07 , help='temperature parameter for softmax')
parser.add_argument('--nce_m', type=float, default=0.5, help='momentum for non-parametric updates')
parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])


opt = parser.parse_args()


# Transform function, used for random augmentation for each students (peer collaboartive learning)
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    # Cutout(n_holes = 1, length=5)    
    ])

# Choose which gpu to use
device = torch.device("cuda:"+opt.gpu_id if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device) 


torch.backends.cudnn.benchmark = True #Accelerates training
batch_size = opt.batch_size
num_epochs = opt.num_epochs
start_epoch = opt.start_epoch
ckp_frequency = 20
# test_num_epochs = 10
# test_ckp_frequency = 3
num_classes = 100
t_model_type = opt.tm
s_model_type = opt.sm
run_index = opt.index
weights_path = "models/"+str(t_model_type)+".pth"


# create data loaders
trainloader, evalloader, testloader, n_data = create_data_loaders(batch_size= batch_size, augment_train=True, opt= opt)




# Creating teacher model
t_model = create_model(use_all_gpus = opt.multi_gpus, device=device, model_type= t_model_type, pretrained=True, weights_path=weights_path)


# General always used loss_fuctions
CL = nn.CrossEntropyLoss()
HKD = KLD(temperature = 4.0)

# Student Model
s_model = create_model(use_all_gpus = opt.multi_gpus, device=device, model_type= s_model_type)
PATH_s = './student_models/best_cifar_model_student_run_'+str(run_index)+'_'+str(t_model_type)+'_'+str(s_model_type)+'_PCL.pth'
ckp_PATH = './checkpoints/'+str(run_index)+'_'+str(t_model_type)+'_'+str(s_model_type)+'_Hinton'+'_PCL.pth'
max_valid_acc = 0
epoch_training_loss = []
training_accuracy = []
validation_loss = []
validation_accuracy = []

# Weights for different losses
w_h_cl = 1
w_h_hl = 1


# Relational Related
r_trainable_list = nn.ModuleList([])
if(opt.rk_type == 'crd'):
    rnd_data = torch.randn(2, 3, 32, 32).cuda()
    t_model.eval()
    s_model.eval()
    feat_t, _ = t_model(rnd_data, is_feat=True)
    feat_s, _ = s_model(rnd_data, is_feat=True)
    opt.s_dim = feat_s[-1].shape[1]
    opt.t_dim = feat_t[-1].shape[1]
    opt.n_data = n_data
    RELATIONAL_LOSS = CRDLoss(opt, device=device)
    r_trainable_list.append(RELATIONAL_LOSS.embed_s)
    r_trainable_list.append(RELATIONAL_LOSS.embed_t)
    w_r_cl = 1
    w_r_hl  = 0
    w_r_rl  = 0.8


# Feature Related
if(opt.fk_type != None):
    rnd_data = torch.randn(2, 3, 32, 32).cuda()
    t_model.eval()
    s_model.eval()
    feat_t, _ = t_model(rnd_data, is_feat=True)
    feat_s, _ = s_model(rnd_data, is_feat=True)
    f_trainable_list = nn.ModuleList([])
    s_n = [f.shape[1] for f in feat_s[1:-1]]
    t_n = [f.shape[1] for f in feat_t[1:-1]]
    FEATURE_LOSS = SemCKDLoss()
    self_attention = SelfA(len(feat_s)-2, len(feat_t)-2, opt.batch_size, s_n, t_n).cuda()
    # Weights for different losses
    w_f_cl = 1
    w_f_hl = 1
    w_f_fl = 400
    f_trainable_list.append(self_attention)



# Defining optimizers params as CRD and semCKD repos
warmup_ratio  = 0.3
anneal_start = int(warmup_ratio * num_epochs)
anneal_ratio = 1- warmup_ratio
anneal_data_iterations = int(num_epochs * anneal_ratio)
lr_decay_rate = 0.1
lr_decay_epochs = [150, 180, 210]


lr = 0.05
if s_model_type in ['mobilenetv2', 'ShuffleV1', 'ShuffleV2']:
    lr = 0.01

optimizer = torch.optim.SGD(s_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

if(opt.resume):
    state = torch.load(ckp_PATH)
    s_model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])

optimizer_r = torch.optim.SGD(r_trainable_list.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

optimizer_f = torch.optim.SGD(f_trainable_list.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)





if(opt.resume):
    sys.stdout = open('./logs/' + str(run_index) + '_' + str(t_model_type) + '_' + str(s_model_type) + '_Hinton' + '_' + str(opt.rk_type)+ '_' + str(opt.fk_type) + '_PCL_' + '_benchmark.log', 'a')
else:
    sys.stdout = open('./logs/' + str(run_index) + '_' + str(t_model_type) + '_' + str(s_model_type) + '_Hinton' + '_' + str(opt.rk_type)+ '_' + str(opt.fk_type) + '_PCL_' + '_benchmark.log', 'w')



# Make checkpoints directory if not exists
if not os.path.isdir('checkpoints'):
    mkdir_p('checkpoints')

if opt.resume:
    print('==> Resuming from checkpoint..')



for epoch in range(start_epoch, num_epochs):
    tepoch = tqdm(trainloader, unit="batch")
    
    t_model.eval()
    num_elems  = 0

    
    s_model.train()
    accurate = 0
    step_training_loss = np.array([])

    adjust_learning_rate(epoch, lr, lr_decay_rate, lr_decay_epochs, optimizer)

    if(opt.rk_type != None):
        adjust_learning_rate(epoch, lr, lr_decay_rate, lr_decay_epochs, optimizer_r)
        r_trainable_list.train()

    if(opt.fk_type != None):
        adjust_learning_rate(epoch, lr, lr_decay_rate, lr_decay_epochs, optimizer_f)
    
    if(opt.fk_type == 'semckd'):
        self_attention.train()
    

    top1_h = AverageMeter()

    for step, batch in enumerate(tepoch):
        tepoch.set_description(f"Epoch {epoch}")
        if(opt.rk_type == 'crd'):
            inputs, labels, index, contrast_idx = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device) 
        else:
            inputs, labels = batch[0].to(device), batch[1].to(device)
        
        if(inputs.shape[0] != batch_size and opt.fk_type == 'semckd'): continue
 

        # Finding Teacher Predictions
        with torch.no_grad():
            feat_t, t_predictions = t_model(inputs, is_feat = True)
            feat_t   = [f.detach() for f in feat_t]

        total_losses = {}

        # Do hinton
        if(opt.use_hinton == True):
            optimizer.zero_grad()
            s_predictions = s_model(inputs)
            cl_h = CL(s_predictions, labels) * w_h_cl
            kldoh_h = HKD(s_predictions, t_predictions) * w_h_hl
                    
            total_losses['ModReduce'] = cl_h + kldoh_h
            

        # Do relational
        if(opt.rk_type != None):
            optimizer_r.zero_grad()
            # r_inputs = transform(inputs)
            if(opt.rk_type == 'crd'):
                feat_s_r, s_predictions = s_model(inputs, is_feat = True)
                rl_r = RELATIONAL_LOSS(feat_s_r[-1], feat_t[-1], index, contrast_idx) * w_r_rl
            
            total_losses[opt.rk_type] = rl_r + cl_h
            total_losses['ModReduce'] = total_losses['ModReduce'] + rl_r

        # Do feature
        if(opt.fk_type != None):    
            optimizer_f.zero_grad()
            # f_inputs = transform(inputs)
            feat_s, s_predictions= s_model(inputs, is_feat = True)

            if(opt.fk_type == 'semckd'):
                s_value, f_target, weight = self_attention(feat_s[1:-1], feat_t[1:-1])
                fl_f = FEATURE_LOSS(s_value, f_target, weight) * w_f_fl

            total_losses['ModReduce'] = total_losses['ModReduce'] + fl_f
            total_losses[opt.fk_type] = fl_f + cl_h
     
            
        
        postfix_value = {}

        if(opt.use_hinton == True):
            total_losses['ModReduce'].backward()
            optimizer.step()
            step_training_loss = np.append(step_training_loss, total_losses['ModReduce'].item())
            
            # Calculate Accuracy on training
            accurate_preds = accuracy(s_predictions, labels)
            top1_h.update(accurate_preds[0].item(), inputs.size(0))
            postfix_value['loss_h = '] = total_losses['hinton'].item()
        

        if(opt.rk_type != None):
            total_losses[opt.rk_type].backward()
            optimizer_r.step()
    
        


        if(opt.fk_type != None):
            total_losses[opt.fk_type].backward()
            optimizer_f.step()
       
        tepoch.set_postfix(postfix_value)
        


    if(opt.use_hinton == True):
        # Training Accuracy
        train_metric = top1_h.avg
        training_accuracy.append(train_metric)
        epoch_training_loss.append(np.mean(step_training_loss))
        print(f"Accuracy on training set for epoch {epoch}: {train_metric:.3f}")
        
        # Evaluation Accuracy
        eval_metric, eval_loss = eval_model(s_model, evalloader, device)
        print(f"Accuracy on evaluation set for s_model_h :{eval_metric:.3f}")
        if(eval_metric> max_valid_acc):
            max_valid_acc = eval_metric
            torch.save(s_model.state_dict(), PATH_s)

        # Save Checkpoint
        if (epoch+1)%ckp_frequency==0:
            ckp_state = {
                'epoch': epoch+1,
                'state_dict': s_model.state_dict(),
                'acc': eval_metric,
                'best_acc': max_valid_acc,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(ckp_state, ckp_PATH)

        validation_accuracy.append(eval_metric)
        validation_loss.append(eval_loss)
        


# create training and validation accuracy plot
print(f"This training online status was Peer Collaborative Learning")

if(opt.use_hinton == True):
    # Accuracy plot
    plt.figure()
    plt.plot(range(1, len(training_accuracy)+1), training_accuracy, 'r--')
    plt.plot(range(1, len(validation_accuracy)+1), validation_accuracy, 'b-')
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Response Student')
    plt.savefig('./plots/'+str(run_index)+'_PCL_'+str(t_model_type)+'_'+str(s_model_type)+'-Accuracy.png')

    # Loss Plot
    plt.figure()
    plt.plot(range(1, len(epoch_training_loss)+1), epoch_training_loss, 'r--')
    plt.plot(range(1, len(validation_loss)+1), validation_loss, 'b-')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Response Student')
    plt.savefig('./plots/'+str(run_index)+'_PCL_'+str(t_model_type)+'_'+str(s_model_type)+'-Loss.png')    
    
    # Evaluate
    s_model.load_state_dict(torch.load(PATH_s))
    eval_metric, test_loss = eval_model(s_model, testloader, device)
    print(f"Best model accuracey on test set:{eval_metric:.3f}")



sys.stdout.close()