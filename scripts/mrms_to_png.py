#!/usr/bin/env python3
"""
scripts/mrms_to_png.py
Phase 3B MVP — MRMS GRIB2 → Transparent PNG → Cloudflare R2

Steps:
  1. Find latest MRMS_MultiSensor_QPE_01H_Pass2 file from NOAA directory
  2. Download + decompress .grib2.gz → .grib2
  3. Convert .grib2 → GeoTIFF (EPSG:4326) using gdal_translate + gdalwarp
  4. Read GeoTIFF with GDAL Python bindings → numpy array
  5. Apply rainfall colormap → transparent RGBA PNG
  6. Upload mrms_latest.png to Cloudflare R2

Required env vars:
  R2_ACCESS_KEY_ID
  R2_SECRET_ACCESS_KEY
  R2_BUCKET_NAME
  R2_PUBLIC_BASE_URL    e.g. https://pub-xxx.r2.dev
  R2_ENDPOINT_URL       e.g. https://<account_id>.r2.cloudflarestorage.com
"""

import os
import re
import gzip
import json
import tempfile
import datetime
import subprocess
import urllib.request

import numpy as np
from PIL import Image
import boto3
from botocore.config import Config

# ---------------------------------------------------------------------------
# MRMS source
# ---------------------------------------------------------------------------
MRMS_BASE_URL = "https://mrms.ncep.noaa.gov/2D/MultiSensor_QPE_01H_Pass2/"
MRMS_PRODUCT  = "MultiSensor_QPE_01H_Pass2"

# R2 config from environment
R2_ACCESS_KEY_ID      = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET_NAME       = os.environ["R2_BUCKET_NAME"]
R2_PUBLIC_BASE_URL   = os.environ["R2_PUBLIC_BASE_URL"].rstrip("/")
R2_ENDPOINT_URL      = os.environ["R2_ENDPOINT_URL"].rstrip("/")

OUTPUT_KEY = "mrms_latest.png"

# ---------------------------------------------------------------------------
# Rainfall colormap (mm input thresholds → RGBA)
# Zero/nodata = fully transparent
# ---------------------------------------------------------------------------
COLORMAP_MM = [
    (0.0,   (0,   0,   0,   0)),     # transparent (no rain)
    (0.254, (100, 200, 255, 160)),   # trace  (~0.01 in)
    (6.35,  (50,  150, 255, 190)),   # 0.25 in
    (12.7,  (30,  80,  220, 210)),   # 0.50 in
    (25.4,  (80,  30,  200, 220)),   # 1.00 in
    (38.1,  (160, 20,  180, 230)),   # 1.50 in
    (63.5,  (220, 30,  80,  235)),   # 2.50 in
    (101.6, (180, 10,  10,  245)),   # 4.00 in
    (float("inf"), (100, 0, 0, 255)),
]


def mm_to_rgba(data_mm: np.ndarray) -> np.ndarray:
    h, w = data_mm.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    thresholds = [t for t, _ in COLORMAP_MM]
    colors     = [c for _, c in COLORMAP_MM]
    for i in range(len(thresholds) - 1):
        lo, hi = thresholds[i], thresholds[i + 1]
        mask = (data_mm >= lo) & (data_mm < hi)
        rgba[mask] = colors[i]
    return rgba


# ---------------------------------------------------------------------------
# Step 1: Find latest filename
# ---------------------------------------------------------------------------
def find_latest_filename() -> str:
    print("Fetching MRMS directory listing…")
    with urllib.request.urlopen(MRMS_BASE_URL, timeout=30) as resp:
        html = resp.read().decode()

    pattern = re.compile(
        r'href__="(MRMS_' + re.escape(MRMS_PRODUCT) + r'_\d{2}\.\d{2}_\d{8}-\d{6}\.grib2\.gz)"'
    )
    matches = pattern.findall(html)
    if not matches:
        raise RuntimeError("No .grib2.gz files found in MRMS directory listing.")

    latest = sorted(matches)[-1]
    print(f"  Latest: {latest}")
    return latest


# ---------------------------------------------------------------------------
# Step 2: Download + decompress
# ---------------------------------------------------------------------------
def download_and_decompress(filename: str, tmpdir: str) -> str:
    url       = MRMS_BASE_URL + filename
    gz_path   = os.path.join(tmpdir, filename)
    grib2_path = gz_path[:-3]  # strip .gz

    print(f"Downloading {url} …")
    with urllib.request.urlopen(url, timeout=120) as resp:
        with open(gz_path, "wb") as f:
            f.write(resp.read())
    print(f"  {os.path.getsize(gz_path):,} bytes (.gz)")

    print("Decompressing…")
    with gzip.open(gz_path, "rb") as gz_in, open(grib2_path, "wb") as out:
        out.write(gz_in.read())
    print(f"  {os.path.getsize(grib2_path):,} bytes (.grib2)")

    os.remove(gz_path)
    return grib2_path


# ---------------------------------------------------------------------------
# Step 3: GRIB2 → GeoTIFF via GDAL CLI (most reliable on CI)
# ---------------------------------------------------------------------------
def grib2_to_geotiff(grib2_path: str, tmpdir: str) -> str:
    tiff_path = os.path.join(tmpdir, "mrms.tif")

    # gdal_translate: extract band 1, output GeoTIFF
    cmd = [
        "gdal_translate",
        "-of", "GTiff",
        "-b", "1",
        grib2_path,
        tiff_path,
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-500:] if result.stdout else "")
    if result.returncode != 0:
        print("STDERR:", result.stderr[-1000:])
        raise RuntimeError(f"gdal_translate failed (exit {result.returncode})")

    print(f"  GeoTIFF: {os.path.getsize(tiff_path):,} bytes")
    return tiff_path


# ---------------------------------------------------------------------------
# Step 4: Read GeoTIFF → numpy array (mm, nodata masked to 0)
# ---------------------------------------------------------------------------
def geotiff_to_array(tiff_path: str):
    from osgeo import gdal
    gdal.UseExceptions()

    ds = gdal.Open(tiff_path)
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray().astype(np.float32)
    nodata = band.GetNoDataValue()

    gt = ds.GetGeoTransform()
    # gt: (west, xres, 0, north, 0, -yres)
    west  = gt[0]
    north = gt[3]
    xres  = gt[1]
    yres  = abs(gt[5])
    nrows, ncols = data.shape
    south = north - nrows * yres
    east  = west  + ncols * xres

    ds = None

    # Mask nodata
    if nodata is not None:
        data[data == nodata] = 0.0
    data[data < 0] = 0.0

    print(f"  Array {nrows}×{ncols}, bounds W={west:.3f} E={east:.3f} S={south:.3f} N={north:.3f}")
    print(f"  mm range: {data.min():.3f} – {data.max():.3f}, nonzero={(data > 0.1).sum():,}")

    bounds_leaflet = [[south, west], [north, east]]
    return data, bounds_leaflet


# ---------------------------------------------------------------------------
# Step 5: numpy (mm) → transparent RGBA PNG
# ---------------------------------------------------------------------------
def array_to_png(data_mm: np.ndarray, png_path: str):
    print("Rendering PNG…")
    rgba = mm_to_rgba(data_mm)

    img = Image.fromarray(rgba, mode="RGBA")

    # Downsample for reasonable file size (MRMS CONUS grid is ~3500×7000)
    w, h = img.size
    if w > 2000:
        img = img.resize((w // 4, h // 4), Image.LANCZOS)
        print(f"  Downsampled to {img.size}")

    img.save(png_path, "PNG", optimize=True)
    print(f"  Saved: {png_path} ({os.path.getsize(png_path):,} bytes)")


# ---------------------------------------------------------------------------
# Step 6: Upload to Cloudflare R2
# ---------------------------------------------------------------------------
def upload_to_r2(png_path: str) -> str:
    s3 = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )

    print(f"Uploading to R2 bucket '{R2_BUCKET_NAME}' as '{OUTPUT_KEY}'…")
    s3.upload_file(
        png_path,
        R2_BUCKET_NAME,
        OUTPUT_KEY,
        ExtraArgs={
            "ContentType": "image/png",
            "CacheControl": "public, max-age=3600",
        },
    )

    public_url = f"{R2_PUBLIC_BASE_URL}/{OUTPUT_KEY}"
    print(f"  Public URL: {public_url}")
    return public_url


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        gz_filename = find_latest_filename()
        grib2_path  = download_and_decompress(gz_filename, tmpdir)
        tiff_path   = grib2_to_geotiff(grib2_path, tmpdir)
        data_mm, bounds = geotiff_to_array(tiff_path)

        png_path = os.path.join(tmpdir, "mrms_latest.png")
        array_to_png(data_mm, png_path)

        public_url = upload_to_r2(png_path)

    print("\n=== Phase 3B MVP complete ===")
    print(f"  PNG URL : {public_url}")
    print(f"  Bounds  : {bounds}")


if __name__ == "__main__":
    main()
