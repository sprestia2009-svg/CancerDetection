## Gets prior Data

import torch
import os
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from math import floor
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryPrecision, BinaryF1Score, BinaryAUROC, BinaryAveragePrecision, BinarySpecificity
from torchvision import transforms
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import time
from torch import nn

## GETTING DATA (I removed the actual file strings)
path_to_cancerous_data = r"..."

meta = []
img = []

## Gets the images, which are from ISIC, Kaggle, and the Ham10000 datasets

for i in os.listdir(path_to_cancerous_data):
    p = os.path.join(path_to_cancerous_data,i)
    if i == "ham10000":
        meta.append(pd.read_csv(os.path.join(p,"HAM10000_metadata.csv")))
        img.extend(p + "\\" + "HAM10000_images_part_1" + "\\" + i for i in os.listdir(os.path.join(p,"HAM10000_images_part_1")))
        img.extend(p + "\\" + "HAM10000_images_part_2" + "\\" + i for i in os.listdir(os.path.join(p,"HAM10000_images_part_2")))

    if i == "ISIC2020":
        meta.append(pd.read_csv(os.path.join(p,"PatientMetaData.csv")))
        img.extend(p + "\\" + "ISIC_2020_Training_JPEG" + "\\" + i for i in os.listdir(os.path.join(p,"ISIC_2020_Training_JPEG")))

    if i == "KaggleData":
        img.extend(p + "\\" + "benign" + "\\" + i for i in os.listdir(os.path.join(p,"benign")))
        meta.extend(np.zeros(len(os.listdir(os.path.join(p,"benign")))))
        img.extend(p + "\\" + "malignant" + "\\" + i for i in os.listdir(os.path.join(p,"malignant")))
        meta.extend(np.ones(len(os.listdir(os.path.join(p,"malignant")))))

path_to_cancerous_data = r"..."


for i in range(len(os.listdir(path_to_cancerous_data))):
    if i != 1:
        i = os.listdir(path_to_cancerous_data)[i]
        p = os.path.join(path_to_cancerous_data,i)
        idx = 0
        if os.listdir(p)[0].endswith(".csv"):
            idx = 1
        img_p = os.path.join(p,os.listdir(p)[idx])
        img_p = os.path.join(img_p,os.listdir(img_p)[0])
        img.extend(img_p + "//" + i for i in os.listdir(img_p) if not i.endswith(".txt"))
        meta.append(pd.read_csv(os.path.join(p,os.path.join(p,os.listdir(p)[1 - idx]))))
    
    else:
        i = os.listdir(path_to_cancerous_data)[i]
        p = os.path.join(path_to_cancerous_data,i)
        idx = 0
        if os.listdir(p)[0].endswith(".csv"):
            idx = 1
        img_p = os.path.join(p,os.listdir(p)[idx])
        img_p = os.path.join(img_p,os.listdir(img_p)[0])
        
        iter_list = [img_p + "//" + i for i in os.listdir(img_p) if not i.endswith(".txt")]

        img.extend(np.array(iter_list)[::2]) ## removes superpixel annotation duplicates, only selecting the actual mole images
        meta.append(pd.read_csv(os.path.join(p,os.path.join(p,os.listdir(p)[1 - idx]))))



img = np.array(img)

## for predicting cancerous moles

targets = meta.copy()

benign = ["nv", "bkl", "df", "vasc"]

targets[0] = (1 - targets[0].dx.isin(benign).astype(int)).values
targets[1] = targets[1].target.values

#targets = #np.concatenate([targets[0], targets[1],targets[2:-4],1 - np.array(targets[-4].benign == "benign").astype(int), targets[-3]["melanoma"].values, targets[-2]["MEL"].values,targets[-1]["MEL"].values])
targets = np.concatenate([targets[0], targets[1],targets[2:-4], (targets[-4].benign == "malignant").astype(int).values, (targets[-3].melanoma).astype(int).values, targets[-2]["MEL"].values,targets[-1]["MEL"].values])


targets = list(targets)
img = list(img)

## Gets New ISIC 2024 data

## Gets New ISIC 2024 data

path_to_cancerous_data = r"..."

idx = 0
if os.listdir(path_to_cancerous_data)[idx].endswith(".csv"):
    idx = 1

targets.extend(pd.read_csv(os.path.join(path_to_cancerous_data,os.listdir(path_to_cancerous_data)[1 - idx])).malignant.astype(int).values)
img.extend([os.path.join(path_to_cancerous_data, os.path.join(os.listdir(path_to_cancerous_data)[idx], i)) for i in os.listdir(os.path.join(path_to_cancerous_data,os.listdir(path_to_cancerous_data)[idx])) if not (i.endswith(".txt") or i.endswith(".csv"))]) ## Metadata stored as the last files here



## FINISHED DATA

## Now we Select Data

torch.manual_seed(2009)

img = np.array(img)
targets = np.array(targets)

idx = torch.randperm(len(img))

X_tr = img[idx[:int(0.8*len(idx))]]
X_ts = img[idx[int(0.8*len(idx)):]]

y_tr = targets[idx[:int(0.8*len(idx))]]
y_ts = targets[idx[int(0.8*len(idx)):]]





## MODEL & Loaders Setup

class MoleDataset(Dataset):
    def __init__(self, X, y, transform):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,idx):
        return self.transform(Image.open(self.X[idx])), torch.from_numpy(np.array([self.y[idx]])).float()
    
if __name__ == "__main__":
    ## transforms

    tr_transforms = transforms.Compose([
        transforms.RandomResizedCrop((256,384), scale = (0.6,0.8)), 
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomRotation(30),
        transforms.RandomPerspective(distortion_scale= 0.75, p = 0.5),
        transforms.ElasticTransform(alpha = 20.0, sigma= 10.0),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05, hue=0.05),
        transforms.ToTensor()
    ])

    model = timm.create_model(
        "efficientnet_b3",
        pretrained=True,
        num_classes = 1
    )

    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    prev_t = transform.transforms[:2]
    prev_t.extend(tr_transforms.transforms[1:-1])
    prev_t.extend(transform.transforms[2:])

    tr_transforms = transforms.Compose(prev_t)
    ts_transforms = transform

    tr_dataset = MoleDataset(X_tr, y_tr, tr_transforms)
    ts_dataset = MoleDataset(X_ts, y_ts, ts_transforms)


    print("Finished Loaders")

    ## Metrics

    from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryPrecision, BinaryF1Score, BinaryAUROC, BinaryAveragePrecision, BinarySpecificity

    metrics = {
        'specificity': BinarySpecificity().to("cuda"),
        'acc': BinaryAccuracy().to("cuda"),
        'recall': BinaryRecall().to("cuda"),
        'precision': BinaryPrecision().to("cuda"),
        'F1': BinaryF1Score().to("cuda"),
        'ap': BinaryAveragePrecision().to("cuda"),
        'auroc': BinaryAUROC().to("cuda"),
    }

    ## Loss Setup

    def FocalLoss(log, y, alpha, gamma = 2, eps = 1e-3):
            probs = torch.sigmoid(log.clamp(-50,100)).clamp(eps, 1 - eps)
            fp = -(alpha[1] * (1 - probs) ** gamma * torch.log(probs) * y).mean() ## for y = 1
            sp = -(alpha[0] * probs ** gamma * torch.log(1 - probs) * (1-y)).mean() ## for y = 0
            
            return (fp + sp) / 2
            
    def SCE(log, y, alpha=0.1, beta=3, A=-1, eps=1e-3, weight=[0.56, 5]):
        probs = torch.sigmoid(log).clamp(eps, 1 - eps)
        weights = torch.where(y > 0, weight[1], weight[0])
        fp = -alpha * ((y * torch.log(probs) + (1 - y) * torch.log(1 - probs)) * weights)
        y1 = torch.where(y > 0, y, torch.tensor(0.5, device=y.device))
        y1 = torch.where(y1 > 0.5, torch.log(y1), torch.tensor(float(A), device=y.device))
        y2 = torch.where(y > 0, torch.tensor(0.5, device=y.device), torch.tensor(1.0, device=y.device))
        y2 = torch.where(y2 > 0.5, torch.log(y2), torch.tensor(float(A), device=y.device))
        sp = -beta * (((probs * y1) + (1 - probs) * y2) * weights)

        return (fp + sp).mean()


    ## Loss
    loss_fn = lambda log, y, alpha = [0.035, 0.965], gamma = 2, eps = 1e-3: FocalLoss(log, y, alpha, gamma, eps) 

    '''
    loss_fn = lambda log, y, alpha = 1, beta = 3, A = -5, eps = 1e-2, weight = [0.56,5]: SCE(log, y, alpha, beta, A, eps, weight) #lambda log, y, gamma = 2, eps = 1e-8: FocalLoss(log, y, lamb, eps)
    eval_loss_fn = lambda log, y, alpha = 1, beta = 3, A = -5, eps = 1e-2, weight = [0.56,5]: SCE(log, y, alpha, beta, A, eps, weight) #lambda log, y, gamma = 2, eps = 1e-8: FocalLoss(log, y, lamb, eps)
    '''


    ## Final Setup

    model.classifier = nn.Sequential(
        nn.Linear(1536,512),
        nn.LeakyReLU(0.05),
        nn.Dropout(p=0.3),
        nn.Linear(512,128),
        nn.LeakyReLU(0.05),
        nn.Linear(128,1)
        )
    model = model.to("cuda")
    #model.load_state_dict(torch.load("model_state_dict.pt")) 


    ## Scalers
    from torch.amp import autocast, GradScaler

    scaler = GradScaler("cuda")


    workers = 12
    batch_size = 96
    fetch = 2
    tr_loader = DataLoader(tr_dataset, batch_size= batch_size, shuffle = True, pin_memory= True, num_workers= workers, persistent_workers= True, prefetch_factor=fetch) #, sampler = sampler)
    ts_loader = DataLoader(ts_dataset, batch_size= batch_size, shuffle= False, pin_memory= True, num_workers= workers, persistent_workers= True, prefetch_factor=fetch) #, sampler= sampler)






    lr = 3e-4
    lr_mult = 0.1

    max_epochs_per_stage = [10,30,40,50,60,80,100, 100]
    warmup_epochs_per_stage = [0, 3, 3,3,3,3, 3,3]
    patience_epochs_per_stage = [3,4,6,6,7,7,7,7] ## len should be len(model.blocks) + 1 = 8 because first stage only impacts the head

    for p in model.parameters():
        p.requires_grad = False

    for p in model.get_classifier().parameters():
        p.requires_grad = True

    trained_params = list(model.get_classifier().parameters())
    optimizer = torch.optim.AdamW(trained_params, lr=lr, weight_decay=1e-4)

    print("----------------------------------  Training Has Begun ----------------------------------")

    loaded_stage = 7
    #model.load_state_dict(torch.load(f"stage{loaded_stage}_best.pt"))

    model.to("cuda")

    for stage in range(len(model.blocks) + 1):
        if stage < loaded_stage:
            continue

        if stage > 0:
            model.load_state_dict(torch.load(f"stage{stage - 1}_best.pt")) ## Loads previous stage best part

            block_idx = len(model.blocks) - stage
            block = model.blocks[block_idx]

            for p in block.parameters():
                p.requires_grad = True

            optimizer.add_param_group({"params": list(block.parameters()), "lr": lr * (lr_mult ** stage)})

        for group in range(len(optimizer.param_groups)):
            optimizer.param_groups[group]["lr"] = lr * (lr_mult ** group) ## reinitialize learning rate after each scheduler step

        warmup_epochs = warmup_epochs_per_stage[stage]
        
        if warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs_per_stage[stage] - warmup_epochs, eta_min=1e-6)
            sched = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        else:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs_per_stage[stage], eta_min=1e-6)

        patience = patience_epochs_per_stage[stage]

        best_val_loss = float("inf")

        patience_counter = 0
        for epoch in range(max_epochs_per_stage[stage]):
            if epoch % 10 == 0:
                print(epoch)
            ep_time = time.time()
            model.train()
            for i, (X,y) in enumerate(tr_loader):
                X = X.to("cuda")
                y = y.to("cuda")
                with autocast("cuda", dtype=torch.bfloat16):
                    log = model(X)
                tr_loss = loss_fn(log.float(),y)
                if torch.isnan(tr_loss):
                    print(f"Train NaN: {i}")
                optimizer.zero_grad()

                scaler.scale(tr_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            tr_time = time.time()

            ts_time = time.time()

            model.eval()

            ## What Follows is just General Code
            total_loss = 0

            for m in metrics.values():
                m.reset()


            with torch.inference_mode():
                for i, (X,y) in enumerate(ts_loader):
                    X = X.to("cuda")
                    y = y.to("cuda")
                    with autocast("cuda", dtype=torch.bfloat16):
                        log = model(X)
                    ts_loss = loss_fn(log.float(),y)

                    if torch.isnan(ts_loss):
                        print("Test NaN")
                        #torch.save(X, "X.pth")
                        #torch.save(y, "y.pth")
                        #torch.save(model.state_dict(), f"model.pt")

                    total_loss += ts_loss.item()

                    for m in metrics.values():
                        m.update(torch.sigmoid(log), y.int())
            total_loss /= (i + 1)
    
            print(f"Stage: {stage} Epoch: {epoch}  Loss: {total_loss}     AP: {metrics['ap'].compute():.4f}     AUROC: {metrics['auroc'].compute():.4f}     F1: {metrics['F1'].compute():.4f}     Precision: {metrics['precision'].compute()}     Recall: {metrics['recall'].compute()}     Accuracy: {metrics['acc'].compute()}     Specificity: {metrics['specificity'].compute()}")
            print(f"Total Time: {((time.time() - ep_time) / 60):.2f}    Train Time: {((tr_time - ep_time) / 60):.2f}    Test Time: {((time.time() - ts_time) / 60):.2f}")

            if total_loss < best_val_loss:
                best_val_loss = total_loss
                torch.save(model.state_dict(), f"stage{stage}_best.pt")
            else:
                if epoch >= warmup_epochs:
                    patience_counter += 1

            if patience_counter >= patience:
                break

            sched.step()

    