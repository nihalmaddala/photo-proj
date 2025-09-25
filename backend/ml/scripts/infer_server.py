#!/usr/bin/env python3
import argparse
import io
import json
from pathlib import Path

from PIL import Image
from flask import Flask, request, jsonify
import torch
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn

# Model definitions (mirrors infer.py)
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

app = Flask(__name__)

class Predictor:
    def __init__(self, model_path: str):
        model_path = Path(model_path)
        label_map_path = model_path.with_suffix('.label_map.json')
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)
        self.label_map = label_map
        self.model = Model(len(label_map['iso']), len(label_map['aperture']), len(label_map['shutter']))
        self.model.load_state_dict(torch.load(str(model_path), map_location='cpu'))
        self.model.eval()

    def predict(self, image: Image.Image):
        x = TRANSFORM(image.convert('RGB')).unsqueeze(0)
        with torch.no_grad():
            out_iso, out_ap, out_sh = self.model(x)
            iso_idx = out_iso.argmax(1).item()
            ap_idx = out_ap.argmax(1).item()
            sh_idx = out_sh.argmax(1).item()
        return {
            'iso': int(self.label_map['iso'][iso_idx]),
            'aperture': f"f/{self.label_map['aperture'][ap_idx]}",
            'shutter': self.label_map['shutter'][sh_idx]
        }

predictor: Predictor = None  # set in main

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    file = request.files['image']
    try:
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes))
        result = predictor.predict(img)
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='/Users/nihalmaddala/photo-proj/backend/ml/models/baseline.pt')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5001)
    args = parser.parse_args()

    global predictor
    # Fallback to baseline_200.pt if main model not present
    model_path = Path(args.model)
    if not model_path.exists():
        alt = Path('/Users/nihalmaddala/photo-proj/backend/ml/models/baseline_200.pt')
        if alt.exists():
            model_path = alt
    predictor = Predictor(str(model_path))

    app.run(host=args.host, port=args.port)

if __name__ == '__main__':
    main() 