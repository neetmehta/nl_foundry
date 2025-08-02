import tqdm
import torch
import wandb

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, schedular, metrics_fn=None, grad_accumulation_steps=1, wandb_run=None, device="cpu"):
        self.model = model
        self.optimizer = optimizer
        self.schedular = schedular
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.model.to(device)
        self.optimizer.zero_grad()
        self.metrics_fn = metrics_fn
        self.steps = 0
        self.grad_accumulation_steps = grad_accumulation_steps
        self.wandb_run = wandb_run
        
    def train_step(self, batch):
        
        outputs = self.model(batch)
        
        loss = outputs['loss']
        loss = loss / self.grad_accumulation_steps
        loss.backward()
        if (self.steps+1) % self.grad_accumulation_steps == 0:
            self.optimizer.step()
            self.schedular.step()
            self.optimizer.zero_grad()
            
        self.steps += 1
        
        final_loss = loss.item() * self.grad_accumulation_steps
        
        if self.wandb_run:
            self.wandb_run.log({"loss": final_loss, "step": self.steps})
        
        return final_loss, outputs
        
    def train_epoch(self):
        
        self.model.train()
        progress_bar = tqdm.tqdm(enumerate(self.train_dataloader), desc="Training", leave=False)
        total_loss = 0.0
        
        for _, batch in progress_bar:
            loss = self.train_step(batch)
            progress_bar.set_postfix(loss = loss)
            total_loss += loss
            
        avg_loss = total_loss / len(self.train_dataloader.dataset)
        print(f"Average Training Loss: {avg_loss}")
                
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        total_metrics = {}
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                outputs = self.model(batch)
                loss = outputs['loss']
                total_loss += loss.item()
                
                if self.metrics_fn:
                    metrics = self.metrics_fn(outputs, batch)
                    for key, value in metrics.items():
                        if key not in total_metrics:
                            total_metrics[key] = 0.0
                        total_metrics[key] += value.item()
                        
        avg_loss = total_loss / len(self.val_dataloader.dataset)
        avg_metrics = {key: value / len(self.val_dataloader.dataset) for key, value in total_metrics.items()}
        
        return avg_loss, avg_metrics

    def train_and_validate(self, num_epochs):
        for epoch in range(num_epochs):
            self.train_epoch()
            val_loss, val_metrics = self.validate()
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss}, Metrics: {val_metrics}")
            
    def save_checkpoint(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.schedular.state_dict(),
            'step': self.steps
        }, path)
