import tqdm

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, metrics_fn=None, grad_accumulation_steps=1, device="cpu"):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.model.to(device)
        self.optimizer.zero_grad()
        self.metrics_fn = metrics_fn
        self.steps = 0
        self.grad_accumulation_steps = grad_accumulation_steps
        
    def train_step(self, batch):
        
        outputs = self.model(batch)
        
        loss = outputs['loss']
        loss = loss / self.grad_accumulation_steps
        loss.backward()
        if (self.steps+1) % self.grad_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            
        self.steps += 1
        
        return loss.item() * self.grad_accumulation_steps, outputs
        
    def train(self, num_epochs):
        
        self.model.train()
        progress_bar = tqdm.tqdm(enumerate(self.train_dataloader), desc="Training", leave=False)
        for epoch in range(num_epochs):
            for _, batch in progress_bar:
                loss = self.train_step(batch)
                progress_bar.set_postfix(loss = loss)