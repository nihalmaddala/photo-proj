#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T

ISO_STOPS = [50, 64, 80, 100, 125, 160, 200, 250, 320, 400, 500, 640, 800, 1000, 1250, 1600, 2000, 2500, 3200, 6400, 12800]
F_STOPS = [1.2, 1.4, 1.8, 2.0, 2.2, 2.8, 3.5, 4.0, 5.6, 8.0, 11.0, 16.0, 22.0]
SHUTTER_STOPS = ['1/8000','1/4000','1/2000','1/1000','1/500','1/250','1/125','1/60','1/30','1/15','1/8','1/4','1/2','1"','2"','4"','8"','15"','30"']

class PhotoDataset(Dataset):
    def __init__(self, csv_path, images_root, split='train', val_ratio=0.1, transforms=None, seed=42):
        df = pd.read_csv(csv_path)
        # deterministic split
        np.random.seed(seed)
        perm = np.random.permutation(len(df))
        val_size = int(len(df) * val_ratio)
        val_idx = set(perm[:val_size])
        if split == 'train':
            self.df = df.iloc[[i for i in range(len(df)) if i not in val_idx]].reset_index(drop=True)
        else:
            self.df = df.iloc[[i for i in range(len(df)) if i in val_idx]].reset_index(drop=True)
        self.images_root = Path(images_root)
        self.transforms = transforms or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        img = self.transforms(img)
        return img, {
            'iso': int(row['iso_cls']),
            'aperture': int(row['aperture_cls']),
            'shutter': int(row['shutter_cls'])
        }

class MultiHead(nn.Module):
    def __init__(self, backbone_out=1000, iso_classes=len(ISO_STOPS), ap_classes=len(F_STOPS), sh_classes=len(SHUTTER_STOPS)):
        super().__init__()
        self.iso_head = nn.Linear(backbone_out, iso_classes)
        self.ap_head = nn.Linear(backbone_out, ap_classes)
        self.sh_head = nn.Linear(backbone_out, sh_classes)

    def forward(self, x):
        return self.iso_head(x), self.ap_head(x), self.sh_head(x)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.heads = MultiHead(backbone_out=512)

    def forward(self, x):
        feats = self.backbone(x)
        return self.heads(feats)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    total = 0
    losses = []
    for images, targets in loader:
        images = images.to(device)
        iso = targets['iso'].to(device)
        ap = targets['aperture'].to(device)
        sh = targets['shutter'].to(device)
        optimizer.zero_grad()
        out_iso, out_ap, out_sh = model(images)
        loss = ce(out_iso, iso) + ce(out_ap, ap) + ce(out_sh, sh)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        total += images.size(0)
    return float(np.mean(losses))


def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction='sum')
    total = 0
    loss_sum = 0.0
    correct = np.zeros(3)
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            iso = targets['iso'].to(device)
            ap = targets['aperture'].to(device)
            sh = targets['shutter'].to(device)
            out_iso, out_ap, out_sh = model(images)
            loss = ce(out_iso, iso) + ce(out_ap, ap) + ce(out_sh, sh)
            loss_sum += loss.item()
            total += images.size(0)
            correct[0] += (out_iso.argmax(1) == iso).sum().item()
            correct[1] += (out_ap.argmax(1) == ap).sum().item()
            correct[2] += (out_sh.argmax(1) == sh).sum().item()
    return loss_sum/total, (correct/total).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--images_root', required=True)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--model_out', required=True)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = PhotoDataset(args.csv, args.images_root, split='train')
    val_ds = PhotoDataset(args.csv, args.images_root, split='val')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val = 1e9
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    for epoch in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, accs = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} acc_iso={accs[0]:.3f} acc_ap={accs[1]:.3f} acc_sh={accs[2]:.3f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.model_out)
            # Save label map for inference
            label_map = {
                'iso': ISO_STOPS,
                'aperture': F_STOPS,
                'shutter': SHUTTER_STOPS
            }
            with open(Path(args.model_out).with_suffix('.label_map.json'), 'w') as f:
                json.dump(label_map, f)
            print(f"Saved best model → {args.model_out}")

if __name__ == '__main__':
    main()
