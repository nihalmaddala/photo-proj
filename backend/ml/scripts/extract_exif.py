#!/usr/bin/env python3
import argparse
import os
import io
from pathlib import Path
import json
import math
import pandas as pd
from PIL import Image, ExifTags
from tqdm import tqdm

COMMON_ISO = [50, 64, 80, 100, 125, 160, 200, 250, 320, 400, 500, 640, 800, 1000, 1250, 1600, 2000, 2500, 3200, 6400, 12800]
COMMON_F = [1.2, 1.4, 1.8, 2.0, 2.2, 2.8, 3.5, 4.0, 5.6, 8.0, 11.0, 16.0, 22.0]
COMMON_SHUTTER = [
    '1/8000','1/4000','1/2000','1/1000','1/500','1/250','1/125','1/60','1/30','1/15','1/8','1/4','1/2','1"','2"','4"','8"','15"','30"'
]

# Helpers

def nearest_class(value, classes):
    # For floats (aperture), choose nearest by absolute distance
    if isinstance(value, (int, float)):
        return min(range(len(classes)), key=lambda i: abs(classes[i] - float(value)))
    raise ValueError('Unsupported value type for nearest_class')


def rational_to_float(x):
    try:
        if isinstance(x, tuple) and len(x) == 2:
            n, d = x
            return float(n) / float(d) if d else None
        return float(x)
    except Exception:
        return None


def exif_to_labels(exif):
    iso = exif.get('ISOSpeedRatings') or exif.get('PhotographicSensitivity')
    fnum = exif.get('FNumber') or exif.get('ApertureValue')
    exposure = exif.get('ExposureTime')

    iso = int(iso) if iso else None
    fnum = rational_to_float(fnum)
    exposure = rational_to_float(exposure)

    # Map exposure seconds to class string in COMMON_SHUTTER
    shutter_str = None
    if exposure:
        if exposure >= 1:
            # seconds format like 1", 2", etc.
            shutter_str = f"{round(exposure)}\""
        else:
            denom = round(1.0 / exposure)
            shutter_str = f"1/{denom}"
    return iso, fnum, shutter_str


def load_exif(image_path):
    try:
        with Image.open(image_path) as img:
            exif_data = {}
            raw = img._getexif() or {}
            for k, v in raw.items():
                tag = ExifTags.TAGS.get(k, k)
                exif_data[tag] = v
            return exif_data
    except Exception:
        return {}


def resize_and_save(src, dst, size=512):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        with Image.open(src) as img:
            img = img.convert('RGB')
            img.thumbnail((size, size))
            img.save(dst, format='JPEG', quality=90)
            return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--output_csv', required=True)
    ap.add_argument('--output_images', required=True)
    ap.add_argument('--max_per_class', type=int, default=1000)
    args = ap.parse_args()

    rows = []
    counts = {}

    input_dir = Path(args.input_dir)
    for p in tqdm(list(input_dir.rglob('*'))):
        if p.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue
        exif = load_exif(p)
        iso, fnum, shutter = exif_to_labels(exif)
        if not iso or not fnum or not shutter:
            continue

        # Guard classes
        if iso not in COMMON_ISO:
            # snap iso to nearest known stop
            iso = min(COMMON_ISO, key=lambda x: abs(x - iso))
        if shutter not in COMMON_SHUTTER:
            # approximate to nearest common shutter by value distance
            def shutter_val(s):
                if '"' in s:
                    return float(s.replace('"',''))
                if '/' in s:
                    return 1.0 / float(s.split('/')[-1])
                return None
            sv = shutter_val(shutter)
            shutter = min(COMMON_SHUTTER, key=lambda s: abs(shutter_val(s) - sv))

        iso_cls = COMMON_ISO.index(iso)
        aperture_cls = nearest_class(fnum, COMMON_F)
        shutter_cls = COMMON_SHUTTER.index(shutter)

        rel = p.relative_to(input_dir)
        out_path = Path(args.output_images) / rel.with_suffix('.jpg')
        if not resize_and_save(p, out_path):
            continue

        key = (iso_cls, aperture_cls, shutter_cls)
        counts[key] = counts.get(key, 0) + 1
        if counts[key] > args.max_per_class:
            continue

        rows.append({
            'path': str(out_path),
            'iso': iso,
            'aperture': f"f/{COMMON_F[aperture_cls]}",
            'shutter': shutter + ('s' if '/' in shutter else ''),
            'iso_cls': iso_cls,
            'aperture_cls': aperture_cls,
            'shutter_cls': shutter_cls,
        })

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")

if __name__ == '__main__':
    main()
