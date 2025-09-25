#!/usr/bin/env python3
import io
import json
import os
from pathlib import Path
from typing import Optional

from flask import Flask, request, jsonify
from PIL import Image

# Optional heavy imports are deferred until first use
_torch = None
_torchvision_models = None
_torch_nn = None
_T = None

app = Flask(__name__)

DEFAULT_MODEL = Path(__file__).parent / 'ml' / 'models' / 'baseline_200.pt'

class _Predictor:
	def __init__(self, model_path: Path):
		global _torch, _torchvision_models, _torch_nn, _T
		if _torch is None:
			import torch as _torch  # type: ignore
			import torchvision.transforms as _T  # type: ignore
			import torchvision.models as _torchvision_models  # type: ignore
			import torch.nn as _torch_nn  # type: ignore
			# Keep CPU usage modest
			_torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS', '1')))

		self.model_path = Path(model_path)
		label_map_path = self.model_path.with_suffix('.label_map.json')
		with open(label_map_path, 'r') as f:
			self.label_map = json.load(f)

		class MultiHead(_torch_nn.Module):
			def __init__(self, backbone_out, iso_classes, ap_classes, sh_classes):
				super().__init__()
				self.iso_head = _torch_nn.Linear(backbone_out, iso_classes)
				self.ap_head = _torch_nn.Linear(backbone_out, ap_classes)
				self.sh_head = _torch_nn.Linear(backbone_out, sh_classes)
			def forward(self, x):
				return self.iso_head(x), self.ap_head(x), self.sh_head(x)

		class Model(_torch_nn.Module):
			def __init__(self, iso_classes, ap_classes, sh_classes):
				super().__init__()
				self.backbone = _torchvision_models.resnet18(weights=_torchvision_models.ResNet18_Weights.DEFAULT)
				self.backbone.fc = _torch_nn.Identity()
				self.heads = MultiHead(512, iso_classes, ap_classes, sh_classes)
			def forward(self, x):
				feats = self.backbone(x)
				return self.heads(feats)

		self.TRANSFORM = _T.Compose([
			_T.Resize((224, 224)),
			_T.ToTensor(),
			_T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
		])

		self.model = Model(
			len(self.label_map['iso']),
			len(self.label_map['aperture']),
			len(self.label_map['shutter'])
		)
		state = _torch.load(str(self.model_path), map_location='cpu')
		self.model.load_state_dict(state)
		self.model.eval()

	def predict(self, image: Image.Image):
		global _torch
		x = self.TRANSFORM(image.convert('RGB')).unsqueeze(0)
		with _torch.no_grad():
			out_iso, out_ap, out_sh = self.model(x)
			iso_idx = out_iso.argmax(1).item()
			ap_idx = out_ap.argmax(1).item()
			sh_idx = out_sh.argmax(1).item()
		return {
			'iso': int(self.label_map['iso'][iso_idx]),
			'aperture': f"f/{self.label_map['aperture'][ap_idx]}",
			'shutterSpeed': self.label_map['shutter'][sh_idx]
		}

_predictor: Optional[_Predictor] = None
_model_path: Optional[Path] = None


def _get_predictor() -> _Predictor:
	global _predictor, _model_path
	if _predictor is None:
		# Resolve model path with fallback
		candidate = os.environ.get('MODEL_PATH')
		if candidate:
			mpath = Path(candidate)
		else:
			mpath = DEFAULT_MODEL
		if not mpath.exists():
			alt = Path(__file__).parent / 'ml' / 'models' / 'baseline.pt'
			if alt.exists():
				mpath = alt
		_model_path = mpath
		_predictor = _Predictor(mpath)
	return _predictor


@app.get('/api/health')
def health():
	loaded = _predictor is not None
	return jsonify({
		'status': 'healthy',
		'aiConfigured': True,
		'mlModelLoaded': loaded,
		'modelPath': str(_model_path) if _model_path else None
	})


@app.post('/api/analyze-image')
def analyze_image():
	try:
		if 'image' not in request.files:
			return jsonify({'error': 'No image file provided'}), 400
		file = request.files['image']
		img = Image.open(io.BytesIO(file.read()))
		result = _get_predictor().predict(img)
		resp = {
			'iso': result['iso'],
			'aperture': result['aperture'],
			'shutterSpeed': result['shutterSpeed'],
			'explanations': {
				'iso': 'Predicted by local model.',
				'aperture': 'Predicted by local model.',
				'shutterSpeed': 'Predicted by local model.'
			},
			'reasoning': 'Settings inferred by a small ResNet18-based classifier.',
			'tip': None,
			'confidence': 0.7,
			'photographyType': 'other',
			'lightingCondition': 'mixed'
		}
		return jsonify({'success': True, 'data': resp})
	except Exception as e:
		return jsonify({'success': False, 'error': str(e)}), 500


@app.post('/api/refine-settings')
def refine_settings():
	try:
		data = request.get_json(silent=True) or {}
		user_input = (data.get('userInput') or '').lower()
		current = data.get('currentSettings') or {}
		iso = int(current.get('iso', 100))
		aperture = str(current.get('aperture', 'f/4.0'))
		shutter = str(current.get('shutterSpeed', '1/250s'))

		if 'blur' in user_input or 'bokeh' in user_input:
			aperture = 'f/1.8'
			shutter = '1/1000s'
		elif 'sharp' in user_input or 'clear' in user_input or 'landscape' in user_input:
			aperture = 'f/8.0'
			shutter = '1/125s'

		resp = {
			'iso': iso,
			'aperture': aperture,
			'shutterSpeed': shutter,
			'explanations': {
				'iso': 'Kept ISO stable for image quality.',
				'aperture': 'Adjusted to match the desired depth of field.',
				'shutterSpeed': 'Adjusted to balance exposure and motion.'
			},
			'reasoning': 'Lightweight on-device refinement without calling external APIs.',
			'tip': None,
			'confidence': 0.6
		}
		return jsonify({'success': True, 'data': resp})
	except Exception as e:
		return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
	port = int(os.environ.get('PORT', '3001'))
	# Avoid Flask debug reloader (spawns extra process) to reduce CPU
	app.run(host='0.0.0.0', port=port, debug=False, threaded=True) 