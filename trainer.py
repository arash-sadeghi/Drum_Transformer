from transformers import AdamW, get_linear_schedule_with_warmup
from torch import nn
import torch
class Trainer:
    def __init__(self,model,d_train_loader,device):
            
        self.EPOCHS = 100

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
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)

    def train_epoch(self):
        self.model = self.model.train()

        losses = []
        correct_predictions = 0

        for d in self.d_train_loader:
            input_ids = d[:][0].to(self.device)
            attention_mask = d[:][2].to(self.device)
            targets = d[:][1].to(self.device)

            outputs = self.model(input_ids=input_ids,attention_mask=attention_mask)

            #TODO
            _, preds = torch.max(outputs, dim=1)
            loss = self.loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return None

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