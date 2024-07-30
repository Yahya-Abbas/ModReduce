from util._util import *
import sys
from src.feature_model import feature_model
from src.relational_model import relational_model
from src.hinton_model import hinton_model
from util.loss import KLD
import time

class ModReduce_trainer():
    def __init__(self, opt):


        # Initiate the device to use
        self.device = torch.device("cuda:"+opt.gpu_id if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(self.device) 
        torch.backends.cudnn.benchmark = True

        # Setting Random Seeds
        torch.manual_seed(12345)
        np.random.seed(12345)

        self.make_directories()

        # Defining different training parameters with those shared
        # with CRD and semCKD are kept the same as their repos
        self.run_index = opt.index
        self.batch_size = opt.batch_size
        self.num_epochs = opt.num_epochs
        self.start_epoch = opt.start_epoch
        self.ckp_frequency = 20
        self.num_classes = 100
        self.warmup_ratio  = 0.3
        self.anneal_start = int(self.warmup_ratio * self.num_epochs)
        self.anneal_ratio = 1- self.warmup_ratio
        self.anneal_data_iterations = int(self.num_epochs * self.anneal_ratio)
        self.lr_decay_rate = 0.1
        self.lr_decay_epochs = [150, 180, 210]
        self.resume = opt.resume
        self.n_students = opt.n_students
        

        # Defining different learning types for the trainer class
        self.sm_h = opt.sm_h
        self.sm_r = opt.sm_r
        self.sm_f = opt.sm_f
        self.sm = opt.sm
        self.tm = opt.tm
        self.hk_type = "hinton" if(opt.use_hinton == 1) else None
        self.rk_type = opt.rk_type
        self.fk_type = opt.fk_type
        self.ol_type = opt.ol
        self.tm_type = opt.tm
        self.multi_gpus = opt.multi_gpus
        self.opt = opt # keeping opt which is a bit redundant

        # ERROR, I did not implement right naming for different
        # student architectures. Note that you have to edit it in
        # create_common() function in the base_model class below  
        # and self.ckp_PATH_online too.

        # Create data loader and init teacher model
        self.trainloader, self.evalloader, self.testloader, self.n_data = create_data_loaders\
            (batch_size= self.batch_size, augment_train=True, opt= opt)
        self.tm_weights_path = "models/"+str(self.tm)+".pth" 
        self.t_model = create_model(use_all_gpus = opt.multi_gpus, device=self.device,\
            model_type= self.tm, pretrained=True, weights_path=self.tm_weights_path)


        
        self.init_online_learning(opt)
        self.create_models()
        self.create_run_log_file()

        # verbose the opt of the training
        print("This training was run with the following options:")
        print(opt)




    def create_run_log_file(self): 
        sys.stdout = open(f"./logs/runIdentifiers_index_{self.run_index}_onlineType_{self.ol_type}_tmType_{self.tm_type}_smType_{self.sm}_knoweldgeTypes_{self.hk_type}_{self.rk_type}_{self.fk_type}.log","w")
        


    def train(self):
        # WARNING: Late online starting had a logical error in the github code.
        # FIXED: Using another flag to avoid never entering the if statement 
        self.epoch = 0
        
        for epoch in range(self.start_epoch, self.num_epochs):

            if epoch == 0:
                start = time.time()
            # epoch identifier to be used all across
            self.epoch = epoch

            # prepare everything needed before looping over batches
            self.epoch_start()

            # Now we start iterating over batches of data
            self.batches_loop()

            # validation and logging post completing epoch of batches 
            self.epoch_end()

            if epoch == 0:
                end = time.time()
                print(f"Time taken by 1 epoch is {(end - start)/60:.3f} minutes\n")
        
        # Create plots post training 
        for model in self.student_models_arr:
            model.create_plots(self)


     

    def epoch_start(self):
            # Encapsulating the train loader in tqdm to provide progress
            self.tepoch = tqdm(self.trainloader, unit="batch")

            # Determine whether online learning is synchronous or 
            # added at a cut off point.
            if(self.opt.use_online == 1):
                if(self.opt.cutoff_epoch == 0):
                    #synchronous offline online training with growing weight of online loss factor 
                    if(self.epoch % self.ol_value_growth_frequency ==0):
                        self.w_ol= self.w_ol_growth_rate * self.w_ol
                else:
                    # online training is off until cutoff epoch is reached
                    if(self.epoch >= self.opt.cutoff_epoch):
                        # Using another flag to avoid never entering this if statement  
                        self.train_online = True
                    else:
                        self.train_online = False

            # Put teacher model in eval mode
            self.t_model.eval()

            # Calculate current epoch consistency weight for pcl
            if(self.train_online == True and self.ol_type == 'pcl'):
                self.consistency_weight = get_current_consistency_weight(self.epoch, self.ramp_up_epoch)

            # For each model, do what is needed pre each epoch
            for model in self.student_models_arr:
                model.forward_epoch_start(self)


            

    def batches_loop(self):
            with torch.autograd.set_detect_anomaly(True):
                for step, self.batch in enumerate(self.tepoch):
                    # Set the description of the progress bar with the Number 
                    # of the current epoch
                    self.tepoch.set_description(f"Epoch {self.epoch}")
                
                    # If statement to resolve crd-specific dataloader
                    if(self.rk_type == 'crd'):
                        self.inputs, self.labels, self.index, self.contrast_idx = self.batch[0].to(self.device), self.batch[1].to(self.device), \
                            self.batch[2].to(self.device), self.batch[3].to(self.device) 
                    else:
                        self.inputs, self.labels = self.batch[0].to(self.device), self.batch[1].to(self.device)

                    # If statement to ignore the last batch that does not have 
                    # a size equal to the self.batch_size in case of semckd as
                    # semckd is not batch_size agnostic   
                    if(self.inputs.shape[0] != self.batch_size and self.fk_type == 'semckd'): continue

                    # Get the teacher predictions to be used in calculating different loss functions.
                    with torch.no_grad():
                        self.feat_t, self.t_predictions = self.t_model(self.inputs, is_feat = True)
                        self.feat_t   = [f.detach() for f in self.feat_t]
                    
                    # Calculate offline loss per each model
                    for model in self.student_models_arr:
                        model.calc_offline_loss(self)

                    # Calculate online loss for each model according
                    # to the type of online knoweldge used. 
                    if(self.train_online == True):
                        self.calc_online_loss()
                    
                    # backpropagate the losses of different models
                    self.postfix = {} # Varible to set the progress bar of tqdm
                    for model in self.student_models_arr:
                        model.back_propagate(self)
                    
                    # set postfix for progress bar
                    self.tepoch.set_postfix(self.postfix)

    def epoch_end(self):
        for model in self.student_models_arr:
            model.post_epoch_log(self)
        
        if(self.train_online == True):
            # Update Peer Mean Teachers if using Peer Collaborative Learning
            if(self.ol_type == 'pcl'):
                for model in self.student_models_arr:
                    model.update_ema_variables(self)
                    
            # Printing student weights for ensemblers 
            if(self.ol_type == 'One' or self.ol_type == 'W_Avg'):
                for i in range(len(self.student_models_arr)):
                    print(f"s_model_{self.student_models_arr[i].identifier} = {self.curr_ensembler_weights_arr[i]:.2f},", end='\n')
            
            # WARNING: CHECKPOINTING IS NOT IMPLEMENTED HERE
            # if(opt.ol!='pcl' and (epoch+1)%ckp_frequency==0):
            #     # Save Checkpoint
            #     ckp_state = {
            #         'epoch': epoch+1,
            #         'state_dict': student_ensembler.state_dict(),
            #         'optimizer': student_weights_optimizer.state_dict(),
            #     }
            #     torch.save(ckp_state, ckp_PATH_online)     

    def calc_online_loss(self):
        if(self.ol_type == 'pcl'):
            # Calculate ema_logit for all available models
            for model in self.student_models_arr:
                model.calc_ema_logit(self)
            # Caclulate total loss with pcl the online method
            for model in self.student_models_arr:
                model.calc_total_loss_pcl(self) 
        else:
            self.ol_ensembler_optimizer.zero_grad()
            # Create a list out of predictions for modularity
            predictions_list = []
            for model in self.student_models_arr:
                predictions_list.append(model.predictions.detach())
            
            # NOTE: I edtied the student ensemblers in _util
            if(self.ol_type == 'FC'):
                self.group_out = self.student_ensembler(predictions_list)
            elif(self.ol_type == 'W_Avg'):
                self.group_out, self.curr_ensembler_weights_arr = self.student_ensembler(predictions_list)
            elif(self.ol_type == 'One'):
                self.group_out, self.curr_ensembler_weights_arr = self.student_ensembler(self.inputs, predictions_list)

            # Compute Group Output loss
            student_weights_loss = self.online_ensembler_CL(self.group_out, self.labels)
            self.group_out = self.group_out.detach()   
            # ERROR: detaching after calculating loss, am not sure this is right?
            # Yahya & Sam: Seems to make sense to detach the group out from the computation graph as we only use the group output to calculate OL loss for students
            
            student_weights_loss.backward()
            self.ol_ensembler_optimizer.step()
            for model in self.student_models_arr:
                model.calc_total_loss_no_pcl(self)


                                
                    

    def make_directories(self):
        # Make directories if not exists
        if not os.path.isdir('checkpoints'):
            mkdir_p('checkpoints')
        if not os.path.isdir('logs'):
            mkdir_p('logs')
        if not os.path.isdir('plots'):
            mkdir_p('plots')
        if not os.path.isdir('student_models'):
            mkdir_p('student_models')
        if not os.path.isdir('models/models_summary'):
            mkdir_p('models/models_summary')


    def create_models(self):
        # This dictionary makes it easier to access different learning types
        # with a string as a key so as to avoid having variables with 
        # different names.
        self.student_models_arr = []
        if(self.hk_type != None):
            self.student_models_arr.append(hinton_model(self))
        if(self.rk_type != None):
            self.student_models_arr.append(relational_model(self))
        if(self.fk_type != None):
            self.student_models_arr.append(feature_model(self))   


    def init_online_learning(self, opt):
        # Specific online learning related update_ema_variables
        if(opt.use_online == True):
            
            ##################################################################### Sanity check for online learning 
            actual_num_students = (1 if (opt.use_hinton == 1) else 0) + (1 if (self.rk_type != None) else 0) \
                + (1 if (self.fk_type != None) else 0) 
            if(actual_num_students != self.n_students or self.n_students < 2 or self.ol_type == None):
                sys.exit("Please provide n_students parameter equivalent to provided student models\
                    \n, also don't run with use_online 1 with less than two models,\
                    \n and please provide an online knoweldge type.")
            ######################################################################################################

            
            
            if(opt.cutoff_epoch == 0):
                self.w_ol = 0.01
                self.max_ol_value = 0.7
                self.ol_value_growth_frequency = 10 
                self.w_ol_growth_rate = (self.max_ol_value/self.w_ol) ** (1/int(self.num_epochs/self.ol_value_growth_frequency))
                self.train_online = True
            else:
                self.w_ol=0.7  
                self.train_online = False 



            # Create Student Ensembler if not pcl
            if(self.ol_type == 'pcl'):
                # If pcl initiate pcl related components
                self.ramp_up_epoch = 80
                self.PCL_KL = KLD(temperature = opt.pcl_t)
            else:
                # Not pcl so we have to create an ensempler for the other online techniques
                # ERROR: in naming this path I don't consider if we have student models of different models 
                self.ckp_PATH_online = './checkpoints/online_ensembler_runIdentifiers_' + str(self.run_index) + '_' + \
                str(self.tm_type) + '_' + str(self.sm) + '_' + str(self.hk_type) + '_' + str(self.rk_type) + '_' + \
                str(self.fk_type) + '_' + str(self.ol_type) + '.pth'
                
                if(self.ol_type == 'One'):
                    self.student_ensembler = One_FC_StudentEnsembler(n_students=self.n_students).to(self.device)
                elif(self.ol_type == 'FC'):
                    self.student_ensembler = FC_StudentEnsembler(n_students=self.n_students).to(self.device)
                else:
                    self.student_ensembler = LinearWeightedAvg(n_students=self.n_students).to(self.device)

                # WARNING: changed the name of student_weights_optimizer to ol_ensembler_optimizer
                # student_weights_optimizer = torch.optim.SGD(student_ensembler.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

                self.ol_ensembler_optimizer = torch.optim.SGD(self.student_ensembler.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
                self.online_ensembler_CL = nn.CrossEntropyLoss()
                if(opt.resume):
                    state = torch.load(self.ckp_PATH_online)
                    self.student_ensembler.load_state_dict(state['state_dict'])
                    self.ol_ensembler_optimizer.load_state_dict(state['optimizer'])
        else:
            # Turn off the flag used with all online learning algorithms
            self.train_online = False


