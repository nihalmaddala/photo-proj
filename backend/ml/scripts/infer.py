#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn

class MultiHead(nn.Module):
    def __init__(self, backbone_out, iso_classes, ap_classes, sh_classes):
        super().__init__()
        self.iso_head = nn.Linear(backbone_out, iso_classes)
        self.ap_head = nn.Linear(backbone_out, ap_classes)
        self.sh_head = nn.Linear(backbone_out, sh_classes)
    def forward(self, x):
        return self.iso_head(x), self.ap_head(x), self.sh_head(x)

class Model(nn.Module):
    def __init__(self, iso_classes, ap_classes, sh_classes):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.heads = MultiHead(512, iso_classes, ap_classes, sh_classes)
    def forward(self, x):
        feats = self.backbone(x)
        return self.heads(feats)

TRANSFORM = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--label_map', required=False)
    ap.add_argument('--image', required=True)
    args = ap.parse_args()

    label_map_path = args.label_map or Path(args.model).with_suffix('.label_map.json')
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)

    model = Model(len(label_map['iso']), len(label_map['aperture']), len(label_map['shutter']))
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.eval()

    img = Image.open(args.image).convert('RGB')
    x = TRANSFORM(img).unsqueeze(0)

    with torch.no_grad():
        out_iso, out_ap, out_sh = model(x)
        iso_idx = out_iso.argmax(1).item()
        ap_idx = out_ap.argmax(1).item()
        sh_idx = out_sh.argmax(1).item()

    print(json.dumps({
        'iso': label_map['iso'][iso_idx],
        'aperture': f"f/{label_map['aperture'][ap_idx]}",
        'shutter': label_map['shutter'][sh_idx]
    }))

if __name__ == '__main__':
    main()
