# libraries import
import torch
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
from models.resnet import resnet32x4, resnet8x4, resnet32, resnet20, resnet110, resnet56
from models.resnetv2_org import resnet50
from models.vgg import vgg13, vgg8, vgg8_bn, vgg13_bn
from models.wrn import wrn_40_2, wrn_40_1, wrn_16_2
from models.ShuffleNetv2 import ShuffleV2
from models.ShuffleNetv1 import ShuffleV1
from models.mobilenetv2 import mobile_half
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import contextlib
import numpy as np
from torch.utils.data import Dataset
import dataset.cifar100 as cifar100
import os
import errno
from util.ramps import sigmoid_rampup
import argparse
from util.loss import CRDLoss

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img



def create_model(use_all_gpus = 0, device = torch.device("cuda:0"), model_type = "resnet32x4", use_cpu = 0, file_path = "model_summary.txt", pretrained = False, weights_path = "", num_classes =100):
    file_path = "./models/models_summary/" + model_type + "_model_summary.txt"
    
    # define model
    if(model_type == "resnet32x4"):
        model = resnet32x4(num_classes = num_classes)
    elif(model_type == "vgg13"):
        model = vgg13_bn(num_classes = num_classes)
    elif(model_type == "wrn_40_2"):
        model =  wrn_40_2(num_classes = num_classes)
    elif(model_type == "wrn_40_1"):
        model =  wrn_40_1(num_classes = num_classes)
    elif(model_type == "vgg8"):
        model = vgg8_bn(num_classes = num_classes)
    elif(model_type == "ShuffleV2"):
        model = ShuffleV2(num_classes = num_classes)
    elif(model_type == "mobilenetv2"):
        model = mobile_half(num_classes = num_classes)
    elif(model_type == "resnet8x4"):
        model = resnet8x4(num_classes = num_classes)
    elif (model_type == "wrn_40_1"):
        model = wrn_40_1(num_classes= num_classes)
    elif(model_type == "wrn_16_2"):
        model = wrn_16_2(num_classes = num_classes)
    elif(model_type == "resnet20"):
        model = resnet20(num_classes = num_classes)
    elif(model_type == "resnet32"):
        model = resnet32(num_classes = num_classes)
    elif(model_type == "resnet50"):
        model = resnet50(num_classes = num_classes)
    elif(model_type == "resnet56"):
        model = resnet56(num_classes = num_classes)
    elif(model_type == "resnet110"):
        model = resnet110(num_classes = num_classes)
    elif(model_type == "ShuffleV1"):
        model = ShuffleV1(num_classes = num_classes)


    if(pretrained == True):
        model.load_state_dict(torch.load(weights_path)['model'])
    

    if(use_cpu == 0):
        model = model.to(device)
    
    # produce error with cpu use
    with open(file_path, "w") as o:
        with contextlib.redirect_stdout(o):
            if(str(device) == "cuda:0"):
                summary(model, (3,32,32))


    if(use_all_gpus == 1):
        model = nn.DataParallel(model)
        print("using all gpus!")
       
    return model


# Data download and preparation
def create_data_loaders(batch_size = 128, mean = None, std = None, augment_train = False, opt= None):
    torch.manual_seed(0)
    np.random.seed(0)

    if(mean == None):
        mean = (0.5071, 0.4867, 0.4408)
    if(std == None):
        std = (0.2675, 0.2565, 0.2761)
        
    if(augment_train == True):
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            # Cutout(n_holes = 1, length=5)    
            ])
    else:
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean, std)])
    

    dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)


    #trainset = dataset
    if (opt.rk_type == 'crd'):
        trainloader, testloader, n_data = cifar100.get_cifar100_dataloaders_sample(batch_size=batch_size, num_workers=8, k = opt.nce_k, mode = opt.mode)
    else:
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last = (opt.fk_type == 'semckd'))
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean, std)])
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=int(batch_size/2), shuffle=False, num_workers=4)
        n_data = -1


    return trainloader, testloader, testloader, n_data


# evaluate model
def eval_model(model, dataloader, device):
    model.eval()
    accurate = 0
    num_elems = 0
    top1 = AverageMeter()
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            inputs, labels = batch[0].to(device), batch[1].to(device)
            # with torch.no_grad():
            #     outputs = model(inputs)
            outputs = model(inputs)

            # _, predictions = torch.max(outputs.data, 1)
            # accurate_preds = predictions == labels
            # num_elems += accurate_preds.shape[0]
            # accurate += accurate_preds.long().sum()
            metrics = accuracy(outputs, labels)
            top1.update(metrics[0].item(), inputs.size(0))

    # eval_metric = accurate.item() / num_elems
    eval_metric = top1.avg
    loss = 1 - eval_metric
    return eval_metric, loss


class ConvReg(nn.Module):
    """Convolutional regression for FitNet (feature map layer)"""
    def __init__(self, s_shape, t_shape, use_relu=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu
        s_N, s_C, s_H, s_W = s_shape
        t_N, t_C, t_H, t_W = t_shape
        self.s_H = s_H
        self.t_H = t_H
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1+s_H-t_H, 1+s_W-t_W))
        else:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, padding=1, stride=1)
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, t):
        if self.s_H == 2 * self.t_H or self.s_H * 2 == self.t_H or self.s_H >= self.t_H:
            x = self.conv(x)
            if self.use_relu:
                return self.relu(self.bn(x)), t
            else:
                return self.bn(x), t
        else:
            x = self.conv(x)
            if self.use_relu:
                return self.relu(self.bn(x)), F.adaptive_avg_pool2d(t, (self.s_H, self.s_H))
            else:
                return self.bn(x), F.adaptive_avg_pool2d(t, (self.s_H, self.s_H))

class Regress(nn.Module):
    """Simple Linear Regression for FitNet (feature vector layer)"""
    def __init__(self, dim_in=1024, dim_out=1024):
        super(Regress, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.relu(x)
        return x


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


# SemCKD learning rate decay
def adjust_learning_rate(epoch, learning_rate, lr_decay_rate, lr_decay_epochs, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(lr_decay_epochs))
    if steps > 0:
        new_lr = learning_rate * (lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class SelfA(nn.Module):
    """Cross layer Self Attention"""
    def __init__(self, s_len, t_len, input_channel, s_n, s_t, factor=4): 
        super(SelfA, self).__init__()
          
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        for i in range(t_len):
            setattr(self, 'key_weight'+str(i), MLPEmbed(input_channel, input_channel//factor))
        for i in range(s_len):
            setattr(self, 'query_weight'+str(i), MLPEmbed(input_channel, input_channel//factor))
        
        for i in range(s_len):
            for j in range(t_len):
                setattr(self, 'regressor'+str(i)+str(j), AAEmbed(s_n[i], s_t[j]))
               
    def forward(self, feat_s, feat_t):
        
        sim_t = list(range(len(feat_t)))
        sim_s = list(range(len(feat_s)))
        bsz = feat_s[0].shape[0]
        # similarity matrix
        for i in range(len(feat_t)):
            sim_temp = feat_t[i].reshape(bsz, -1)
            sim_t[i] = torch.matmul(sim_temp, sim_temp.t())
        for i in range(len(feat_s)):
            sim_temp = feat_s[i].reshape(bsz, -1)
            sim_s[i] = torch.matmul(sim_temp, sim_temp.t())
        
        # key of target layers    
        proj_key = self.key_weight0(sim_t[0])
        proj_key = proj_key[:, :, None]
        
        for i in range(1, len(sim_t)):
            temp_proj_key = getattr(self, 'key_weight'+str(i))(sim_t[i])
            proj_key =  torch.cat([proj_key, temp_proj_key[:, :, None]], 2)
        
        # query of source layers   
        proj_query = self.query_weight0(sim_s[0])
        proj_query = proj_query[:, None, :]
        for i in range(1, len(sim_s)):
            temp_proj_query = getattr(self, 'query_weight'+str(i))(sim_s[i])
            proj_query = torch.cat([proj_query, temp_proj_query[:, None, :]], 1)
        
        # attention weight
        energy = torch.bmm(proj_query, proj_key) # batch_size X No.stu feature X No.tea feature
        attention = F.softmax(energy, dim = -1)
        
        # feature space alignment
        proj_value_stu = []
        value_tea = []
        for i in range(len(sim_s)):
            proj_value_stu.append([])
            value_tea.append([])
            for j in range(len(sim_t)):            
                s_H, t_H = feat_s[i].shape[2], feat_t[j].shape[2]
                if s_H > t_H:
                    input = F.adaptive_avg_pool2d(feat_s[i], (t_H, t_H))
                    proj_value_stu[i].append(getattr(self, 'regressor'+str(i)+str(j))(input))
                    value_tea[i].append(feat_t[j])
                elif s_H < t_H or s_H == t_H:
                    target = F.adaptive_avg_pool2d(feat_t[j], (s_H, s_H))
                    proj_value_stu[i].append(getattr(self, 'regressor'+str(i)+str(j))(feat_s[i]))
                    value_tea[i].append(target)
                
        return proj_value_stu, value_tea, attention

class AAEmbed(nn.Module):
    """non-linear embed by MLP"""
    def __init__(self, num_input_channels=1024, num_target_channels=128):
        super(AAEmbed, self).__init__()
        self.num_mid_channel = 2 * num_target_channels
        
        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)
        def conv3x3(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        
        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, self.num_mid_channel),
            nn.BatchNorm2d(self.num_mid_channel),
            nn.ReLU(inplace=True),
            conv3x3(self.num_mid_channel, self.num_mid_channel),
            nn.BatchNorm2d(self.num_mid_channel),
            nn.ReLU(inplace=True),
            conv1x1(self.num_mid_channel, num_target_channels),
        )

    def forward(self, x):
        x = self.regressor(x)
        return x
class MLPEmbed(nn.Module):
    """non-linear embed by MLP"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(MLPEmbed, self).__init__()
        self.linear1 = nn.Linear(dim_in, 2 * dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(2 * dim_out, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.l2norm(self.linear2(x))
        return x
class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class LinearWeightedAvg(nn.Module):
    def __init__(self, n_students=3):
        super(LinearWeightedAvg, self).__init__()
        self.n_students = n_students
        if(n_students == 3):
            self.weights = nn.Parameter(torch.tensor([0.33, 0.33, 0.33]), requires_grad=True)
        else:
            self.weights = nn.Parameter(torch.tensor([0.5, 0.5]), requires_grad=True)
        
    def forward(self, predictions_arr):

        prediction_1 = None
        prediction_2 = None
        prediction_3 = None

        if(len(predictions_arr) == 3):
            prediction_1 = predictions_arr[0]
            prediction_2 = predictions_arr[1]
            prediction_3 = predictions_arr[2]
        elif(len(predictions_arr) == 2):
            prediction_1 = predictions_arr[0]
            prediction_2 = predictions_arr[1]

        self.weights.data = F.softmax(self.weights, dim=0)
        if(self.n_students == 3):
            group_out = self.weights[0].expand_as(prediction_1)*prediction_1 \
                        + self.weights[1].expand_as(prediction_2)*prediction_2 \
                        + self.weights[2].expand_as(prediction_3)*prediction_3
            return group_out, [self.weights[0], self.weights[1], self.weights[2]]
        else:
            group_out =  self.weights[0].expand_as(prediction_1)*prediction_1 \
                        + self.weights[1].expand_as(prediction_2)*prediction_2
            return group_out, [self.weights[0], self.weights[1]]

class One_FC_StudentEnsembler(nn.Module):
    def __init__(self, n_students=3):
        super(One_FC_StudentEnsembler, self).__init__()
        self.avgpool_weights = nn.AvgPool2d(16)
        if(n_students == 2):
            self.FC = nn.Linear(in_features=12, out_features=2)
        else:
            self.FC = nn.Linear(in_features=12, out_features=3)
        self.n_students = n_students

    def forward(self, inputs, predictions_arr):

        prediction_1 = None
        prediction_2 = None
        prediction_3 = None

        if(len(predictions_arr) == 3):
            prediction_1 = predictions_arr[0]
            prediction_2 = predictions_arr[1]
            prediction_3 = predictions_arr[2]
        elif(len(predictions_arr) == 2):
            prediction_1 = predictions_arr[0]
            prediction_2 = predictions_arr[1]

        weights = self.avgpool_weights(inputs)
        weights = weights.view(weights.size(0), -1)
        weights = self.FC(weights)
        weights = F.relu(weights)
        weights = F.softmax(weights, dim=1)
        if(self.n_students == 3):
            cur_st1_w = weights[:,0].mean()
            cur_st2_w = weights[:,1].mean()
            cur_st3_w = weights[:,2].mean()
            w_st1 = weights[:,0].repeat(prediction_1.size()[1], 1).transpose(0, 1)
            w_st2 = weights[:,1].repeat(prediction_2.size()[1], 1).transpose(0, 1)
            w_st3 = weights[:,2].repeat(prediction_3.size()[1], 1).transpose(0, 1)
            group_out = w_st1*prediction_1 + w_st2*prediction_2 + w_st3*prediction_3
            
            return group_out, [cur_st1_w, cur_st2_w, cur_st3_w]
        else:
            cur_st1_w = weights[:,0].mean()
            cur_st2_w = weights[:,1].mean()
            w_st1 = weights[:,0].repeat(prediction_1.size()[1], 1).transpose(0, 1)
            w_st2 = weights[:,1].repeat(prediction_2.size()[1], 1).transpose(0, 1)
            group_out = w_st1*prediction_1 + w_st2*prediction_2
            
            return group_out, [cur_st1_w, cur_st2_w]


class FC_StudentEnsembler(nn.Module):
    def __init__(self, num_classes=100, n_students=3):
        super(FC_StudentEnsembler, self).__init__()
        self.n_students = n_students
        self.FC = nn.Linear(in_features=num_classes*n_students, out_features=num_classes)

    def forward(self, predictions_arr):

        prediction_1 = None
        prediction_2 = None
        prediction_3 = None
        if(len(predictions_arr) == 3):
            prediction_1 = predictions_arr[0]
            prediction_2 = predictions_arr[1]
            prediction_3 = predictions_arr[2]
        elif(len(predictions_arr) == 2):
            prediction_1 = predictions_arr[0]
            prediction_2 = predictions_arr[1]

        if(self.n_students == 3):
            predictions = torch.cat((prediction_1, prediction_2, prediction_3), dim=1)
            group_out = self.FC(predictions)
            return group_out
        else:
            predictions = torch.cat((prediction_1, prediction_2), dim=1)
            group_out = self.FC(predictions)
            return group_out

def get_current_consistency_weight(epoch, epoch_threshold):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return sigmoid_rampup(epoch, epoch_threshold)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim = 1, largest = True, sorted = True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def update_ema_variables(model, ema_model, global_step, alpha=0.999):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def get_parser_options():
    ## adding arguments 
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--index', type=int, default=0, help='index indicating current run')

    parser.add_argument('--tm', type=str, default="resnet32x4", help='teacher model type',
                        choices=['resnet8', 'resnet20', 'resnet32', 'resnet56', 'resnet110','resnet8x4', 
                                    'resnet32x4', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2','vgg8', 'vgg13',
                                    'resnet50', 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])

    parser.add_argument('--use_hinton', type=int, default=1, help="calcualte hinton loss or not?")
    parser.add_argument('--rk_type', default=None, type=str, choices = ['rkd', 'crd'])
    parser.add_argument('--fk_type', default=None, type=str, choices=['fitnets', 'semckd'])

    parser.add_argument('--n_students', default=3, type=int, choices=[1, 2, 3])
    parser.add_argument('--sm', type=str, default="resnet8x4", 
                        help='student model type, if training same architecture for all students')
    parser.add_argument('--sm_h', type=str, default=None, help='hinton student model type')
    parser.add_argument('--sm_r', type=str, default=None, help='relational student model type')
    parser.add_argument('--sm_f', type=str, default=None, help='feature student model type')

    parser.add_argument('--gpu_id', type=str, default='0', help='Id for the preferred GPU')
    parser.add_argument('--multi_gpus', type=int, default=0, help='use multiple GPUs (1) or only one (0)')

    parser.add_argument('--batch_size', default=64, type=int, help='Determines the batch size.')
    parser.add_argument('--num_epochs', default=240, type=int, help='Determines the number of epochs.')

    parser.add_argument('--use_online', type=int, default=1, help='decide whether to use online learning or not.')
    parser.add_argument('--ol', default=None, type=str, choices=['One', 'FC', 'W_Avg', 'pcl'], 
                        help='online learning method. i.e. ONE (One), FC, LinearWeightedAvg (W_Avg), Peer Collaborative Learning (pcl)')
    parser.add_argument('--pcl_t', type=int, default=3, choices=[3, 4],
                        help='T to use with Peer Collaborative Learning. 3 (used in PCL paper) or 4 (used in SemCKD)')

    parser.add_argument('--cutoff_epoch', default=0, type=int, help='Cutoff epoch at which online learning starts. A value of 0 indicates either a synchronous online-offline training or fully offline training based on use_online argument value. Case of fully online training is unaccounted for.')

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
    return opt


