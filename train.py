import torch
import torch.nn as F
from dataset import train_dataset, val_dataset, test_dataset
from metrics import AccuracyTrack, LossTrack
from bert import BERT
from torch.utils.data import DataLoader
import random
import numpy as np 

def set_seed(seed):
    """ Set all seeds to make results reproducible (deterministic mode).
        When seed is a false-y value or not supplied, disables deterministic mode. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def push_batch_to_device(data, device):
    data['inp'] = data['inp'].to(device)
    data['label'] = data['label'].to(device)
@torch.no_grad()
def evaluate(model, dataloader, PATH, mode):
    average_loss = LossTrack(F.CrossEntropyLoss)
    accuracy_calc = AccuracyTrack()
    model.load_state_dict(torch.load(PATH))
    model.eval() 
    # with torch.no_grad():
    for data in dataloader:
        push_batch_to_device(data, device)
        out = model(data['inp'])
        accuracy_calc.update(out, data['label'])
        average_loss.update(out, data['label'])
    print(f"{mode}_Accuracy : {round(accuracy_calc.item(), 3)}%, {mode}_Loss : {round(average_loss.item(), 3)}")
    
@torch.no_grad()
def validate(model, dataloader):
    average_loss = LossTrack(F.CrossEntropyLoss)
    accuracy_calc = AccuracyTrack()
    model.eval()
    # with torch.no_grad():
    for data in dataloader:
        push_batch_to_device(data, device)
        out = model(data['inp'])
        accuracy_calc.update(out, data['label'])
        average_loss.update(out, data['label'])
    return accuracy_calc.item(), average_loss.item()

def train(model, epochs, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_func = F.CrossEntropyLoss()
    max_acc, best_epoch = 0, 0
    train_acc, train_loss = AccuracyTrack(), LossTrack(F.CrossEntropyLoss)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.99, patience = 10
    )
    for epoch in range(epochs):
        # print(f"##############################  Epoch : {epoch}  ###################################")
        model.train()
        for data in train_loader:
            push_batch_to_device(data, device)
            out = model(data['inp'])
            # Reset the accumulated gradients obtained from previous step to 0 & calculate new gradients
            loss = loss_func(out, data['label'])
            optimizer.zero_grad()
            loss.backward()
            # Apply the gradients to update weights
            optimizer.step()
            # Calculate Accuracy and Loss for training data
            train_acc.update(out, data['label'])
            train_loss.update(out, data['label'])
        
        val_acc, val_loss = validate(model, val_loader)
        scheduler.step(val_loss)
        print(f"Epoch : {epoch}, Train_Accuracy : {round(train_acc.item(), 3)}%, Train_Loss : {round(train_loss.item(), 3)}, Val_Accuracy : {round(val_acc, 3)}%, Val_Loss : {round(val_loss, 3)}, LR : {scheduler.optimizer.param_groups[0]['lr']}")
        
        if val_acc > max_acc:
            max_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), './transformer.pth.tar')
        train_acc.reset()
        train_loss.reset()
    print(f"Maximum validation accuracy is : {max_acc}, epoch : {best_epoch}")
set_seed(0)
LEARNING_RATE = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_OF_WORKERS = 16
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=NUM_OF_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=NUM_OF_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=NUM_OF_WORKERS)

model = BERT(262)
model.to(device)
train(model, 300, train_loader, val_loader)
evaluate(model, test_loader, './transformer.pth.tar', 'test')


