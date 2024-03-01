from transformers import AdamW, get_linear_schedule_with_warmup
from torch import nn
import torch
from tqdm import tqdm
import wandb

class Trainer:
    EPOCHS = 100
    def __init__(self,model,d_train_loader,device):
            

        self.model = model
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, correct_bias=False)
        
        self.d_train_loader = d_train_loader
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

    def train(self):
        self.progress_bar = tqdm(total=Trainer.EPOCHS, initial=0, ncols=100, mininterval=1)
        self.step = 0

        self.initial_weights = {}
        for name, param in self.model.named_parameters():
            self.initial_weights[name] = param.clone().detach()

        for epoch in range(Trainer.EPOCHS):
            self.train_epoch(epoch)
            self.progress_bar.update(1)

    
    def train_epoch(self,epoch):
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
            wandb.log({"x": float(self.train_losses[-1]*100)},step=self.step)
            
            print("l1:",
                  torch.all(self.model.l1.weight == l1.weight)
                  
                  )

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


    # def eval_model(model, data_loader, loss_fn, device, n_examples):
    #     model = model.eval()

    #     losses = []
    #     correct_predictions = 0

    #     with torch.no_grad():
    #         for d in data_loader:

    #         input_ids = d[:][0].to(device)
    #         attention_mask = d[:][2].to(device)
    #         targets = d[:][1].to(device)

    #         outputs = model(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask
    #         )
    #         _, preds = torch.max(outputs, dim=1)

    #         loss = loss_fn(outputs, targets)

    #         correct_predictions += torch.sum(preds == targets)
    #         losses.append(loss.item())

    #     return correct_predictions.double() / n_examples, np.mean(losses)