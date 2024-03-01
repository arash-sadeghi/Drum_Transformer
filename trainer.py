from transformers import AdamW, get_linear_schedule_with_warmup
from torch import nn
import torch
from tqdm import tqdm
import wandb
import os
from time import time,ctime

def create_path_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Trainer:
    EPOCHS = 1000
    SAVE_INTERVAL = 100
    CODE_RUN_TIME = ctime(time()).replace(':','_').replace(' ','_')
    STATE_SAVE_PATH = os.path.join('data','state',CODE_RUN_TIME)

    def __init__(self,model,d_train_loader,d_val_loader,device):
            

        self.model = model
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, correct_bias=False)
        
        self.d_train_loader = d_train_loader
        self.d_val_loader = d_val_loader

        total_steps = len(d_train_loader) * self.EPOCHS

        self.scheduler = get_linear_schedule_with_warmup(
        self.optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
        )

        self.device = device
        self.loss_fn = nn.MSELoss().to(self.device)
        self.train_losses = []
        wandb.init(project="Music-Transformer")
        wandb.watch(self.model)

        self.generate_required_paths()
    
    def generate_required_paths(self):

        if not os.path.exists(Trainer.STATE_SAVE_PATH):
            create_path_if_not_exists(Trainer.STATE_SAVE_PATH)
            print(f"[+] Path '{Trainer.STATE_SAVE_PATH}' didn't exist and has been created.")
        else:
            print(f"[+] Path '{Trainer.STATE_SAVE_PATH}' already exists.")


    def train(self):
        self.progress_bar = tqdm(total=Trainer.EPOCHS, initial=0, ncols=100, mininterval=1)
        self.step = 0

        self.initial_weights = {}
        for name, param in self.model.named_parameters():
            self.initial_weights[name] = param.clone().detach()

        for epoch in range(Trainer.EPOCHS):
            self.train_epoch()
            self.progress_bar.update(1)

    def save_model_weigthts(self):
        torch.save( self.model.state_dict() , os.path.join(Trainer.STATE_SAVE_PATH,f'weights_{self.step}.pts') )
    
    def train_epoch(self):
        self.model = self.model.train()
        for d in self.d_train_loader:
            self.optimizer.zero_grad()
            l1 = self.model.l1

            input_ids = d[:][0].to(self.device)
            attention_mask = d[:][2].to(self.device)
            targets = d[:][1].to(self.device)

            outputs = self.model(input_ids=input_ids,attention_mask=attention_mask)
            # outputs = torch.stack(outputs)
            # outputs = outputs.transpose(1,0).squeeze()
            outputs_masked = (targets>0)*outputs

            loss = self.loss_fn(outputs_masked, targets)

            self.train_losses.append(loss.item())

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.progress_bar.set_description_str("(train_loss={: 8.6f})".format(self.train_losses[-1]))
            self.step +=1
            wandb.log({"trainLoss": float(loss.item()*100)},step=self.step)

            if self.step % Trainer.SAVE_INTERVAL == 0:
                self.save_model_weigthts()
                self.eval_model()

            
            # print("l1:",torch.all(self.model.l1.weight == l1.weight))

            # for block in range(len(l1)):
            #     if not torch.all(self.model.l1[block].weight == l1[block].weight):
            #         print("!!!detected an update")
            #         break
            # print("update_check over")

            # # Check which weights have changed
            # for name, param in self.model.named_parameters():
            #     if torch.equal(param, self.initial_weights[name]):
            #         # print(f"{name}: Unchanged")
            #         pass
            #     else:
            #         print(f"{name}: Changed")

            # Access gradients for each layer's parameters
            # print("Gradient of loss with respect to parameters:")
            # for name, param in self.model.named_parameters():
            #     if param.grad is not None:
            #         print(name, param.grad)


    def eval_model(self):
        self.model = self.model.eval()
        for d in self.d_val_loader:
            input_ids = d[:][0].to(self.device)
            attention_mask = d[:][2].to(self.device)
            targets = d[:][1].to(self.device)

            outputs = self.model(input_ids=input_ids,attention_mask=attention_mask)
            outputs_masked = (targets>0)*outputs

            loss = self.loss_fn(outputs_masked, targets)

            wandb.log({"valLoss": float(loss.item()*100)},step=self.step)

            torch.save( d , os.path.join(Trainer.STATE_SAVE_PATH,f'input_{self.step}.pts') )
            torch.save( outputs , os.path.join(Trainer.STATE_SAVE_PATH,f'ouput_{self.step}.pts') )


