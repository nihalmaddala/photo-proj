# DSLR Settings ML Pipeline (Local Prototype)

Goal: train a lightweight model that maps an input image to 3 targets:
- ISO (classification over common ISO stops)
- Aperture (classification over common f-stops)
- Shutter speed (classification over common shutter stops)

This is a pragmatic baseline that can run locally on CPU and can be improved later.

## 1) Create dataset from images with EXIF

Put raw images under `backend/ml/data/raw/` (any nested folders allowed).
Run the extractor to build a CSV with labels and a resized dataset for training.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install pillow exifread pandas numpy scikit-learn torch torchvision tqdm

python backend/ml/scripts/extract_exif.py \
  --input_dir backend/ml/data/raw \
  --output_csv backend/ml/data/dataset.csv \
  --output_images backend/ml/data/processed \
  --max_per_class 500
```

The CSV will contain columns: `path,iso,aperture,shutter,iso_cls,aperture_cls,shutter_cls`.

## 2) Train baseline model

```bash
python backend/ml/scripts/train_baseline.py \
  --csv backend/ml/data/dataset.csv \
  --images_root backend/ml/data/processed \
  --epochs 10 \
  --batch_size 32 \
  --lr 1e-3 \
  --model_out backend/ml/models/baseline.pt
```

## 3) Export and integrate

The training script saves a Torch model and a JSON label map. At inference time in the Node backend, use a Python microservice (or a CLI call) to load the model and return predictions.

Prototype CLI inference:

```bash
python backend/ml/scripts/infer.py \
  --model backend/ml/models/baseline.pt \
  --label_map backend/ml/models/label_map.json \
  --image /path/to/image.jpg
```

## Notes
- This is a baseline; accuracy depends on data quality/coverage.
- We discretize targets into common photographic stops to reduce label sparsity.
- Later, we can add regression heads, data augmentation, and curriculum per genre (portrait, landscape, action).
- If EXIF is missing, sample can be skipped.
