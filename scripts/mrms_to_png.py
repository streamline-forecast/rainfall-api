#!/usr/bin/env python3
"""
scripts/mrms_to_png.py
Rolling 24-hour MRMS archive — GRIB2 → GeoTIFF + PNG → Cloudflare R2

For each of the latest 24 hourly MRMS files:
  1. Download .grib2.gz from NOAA
  2. Decompress → .grib2
  3. Convert → GeoTIFF (EPSG:4326)
  4. Render → transparent RGBA PNG
  5. Upload PNG  → mrms/hourly/png/mrms_NN.png   (NN = sequence 00-23)
  6. Upload TIF  → mrms/hourly/geotiff/mrms_NN.tif
  7. Upload GRIB2→ mrms/hourly/grib2/mrms_NN.grib2

After all files, build and upload:
  mrms/hourly/index.json

Also keeps the legacy:
  mrms_latest.png   (most-recent hour, unchanged key)

Required env vars:
  R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME,
  R2_PUBLIC_BASE_URL, R2_ENDPOINT_URL
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
# Config
# ---------------------------------------------------------------------------
MRMS_BASE_URL = "https://mrms.ncep.noaa.gov/2D/MultiSensor_QPE_01H_Pass2/"
MRMS_PRODUCT  = "MultiSensor_QPE_01H_Pass2"
NUM_HOURS     = 24   # rolling window to keep

R2_ACCESS_KEY_ID     = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET_NAME       = os.environ["R2_BUCKET_NAME"]
R2_PUBLIC_BASE_URL   = os.environ["R2_PUBLIC_BASE_URL"].rstrip("/")
R2_ENDPOINT_URL      = os.environ["R2_ENDPOINT_URL"].rstrip("/")

BOUNDS_LEAFLET = [[20.0, -130.0], [55.0, -60.0]]   # fallback; overridden per file

# ---------------------------------------------------------------------------
# Rainfall colormap (mm → RGBA)
# ---------------------------------------------------------------------------
COLORMAP_MM = [
    (0.0,          (0,   0,   0,   0)),
    (0.254,        (100, 200, 255, 160)),
    (6.35,         (50,  150, 255, 190)),
    (12.7,         (30,  80,  220, 210)),
    (25.4,         (80,  30,  200, 220)),
    (38.1,         (160, 20,  180, 230)),
    (63.5,         (220, 30,  80,  235)),
    (101.6,        (180, 10,  10,  245)),
    (float("inf"), (100, 0,   0,   255)),
]

def mm_to_rgba(data_mm: np.ndarray) -> np.ndarray:
    h, w = data_mm.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    thresholds = [t for t, _ in COLORMAP_MM]
    colors     = [c for _, c in COLORMAP_MM]
    for i in range(len(thresholds) - 1):
        lo, hi = thresholds[i], thresholds[i + 1]
        rgba[(data_mm >= lo) & (data_mm < hi)] = colors[i]
    return rgba


# ---------------------------------------------------------------------------
# R2 client (shared)
# ---------------------------------------------------------------------------
def make_s3():
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def upload_bytes(s3, key: str, data: bytes, content_type: str):
    s3.put_object(
        Bucket=R2_BUCKET_NAME,
        Key=key,
        Body=data,
        ContentType=content_type,
        CacheControl="public, max-age=3600",
    )
    return f"{R2_PUBLIC_BASE_URL}/{key}"


def upload_file(s3, local_path: str, key: str, content_type: str) -> str:
    s3.upload_file(
        local_path,
        R2_BUCKET_NAME,
        key,
        ExtraArgs={"ContentType": content_type, "CacheControl": "public, max-age=3600"},
    )
    return f"{R2_PUBLIC_BASE_URL}/{key}"


# ---------------------------------------------------------------------------
# Step 1: Fetch directory listing → sorted list of up-to-24 filenames
# ---------------------------------------------------------------------------
def find_latest_filenames(n: int = NUM_HOURS) -> list:
    print(f"Fetching MRMS directory listing…")
    with urllib.request.urlopen(MRMS_BASE_URL, timeout=30) as resp:
        html = resp.read().decode()

    pattern = re.compile(
        r'href__=\"(MRMS_' + re.escape(MRMS_PRODUCT)
        + r'_\\d{2}\\.\\d{2}_(\\d{8}-\\d{6})\\.grib2\\.gz)\"'
    )
    matches = pattern.findall(html)   # list of (filename, timestamp_str)
    if not matches:
        raise RuntimeError("No .grib2.gz files found in MRMS directory listing.")

    # Sort by embedded timestamp (already lexicographically sortable)
    matches.sort(key=lambda x: x[1])
    latest = matches[-n:]             # newest N
    filenames = [m[0] for m in latest]
    print(f"  Found {len(matches)} files total, using latest {len(filenames)}")
    for f in filenames:
        print(f"    {f}")
    return filenames


# ---------------------------------------------------------------------------
# Step 2: Download + decompress
# ---------------------------------------------------------------------------
def download_and_decompress(filename: str, tmpdir: str) -> str:
    url        = MRMS_BASE_URL + filename
    gz_path    = os.path.join(tmpdir, filename)
    grib2_path = gz_path[:-3]

    print(f"  Downloading {filename} …")
    with urllib.request.urlopen(url, timeout=120) as resp:
        with open(gz_path, "wb") as f:
            f.write(resp.read())

    with gzip.open(gz_path, "rb") as gz_in, open(grib2_path, "wb") as out:
        out.write(gz_in.read())

    os.remove(gz_path)
    print(f"    {os.path.getsize(grib2_path):,} bytes (.grib2)")
    return grib2_path


# ---------------------------------------------------------------------------
# Step 3: GRIB2 → GeoTIFF
# ---------------------------------------------------------------------------
def grib2_to_geotiff(grib2_path: str, tiff_path: str):
    cmd = ["gdal_translate", "-of", "GTiff", "-b", "1", grib2_path, tiff_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gdal_translate failed:\\n{result.stderr[-800:]}")
    return tiff_path


# ---------------------------------------------------------------------------
# Step 4: GeoTIFF → numpy array + bounds
# ---------------------------------------------------------------------------
def geotiff_to_array(tiff_path: str):
    from osgeo import gdal
    gdal.UseExceptions()

    ds   = gdal.Open(tiff_path)
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray().astype(np.float32)
    nodata = band.GetNoDataValue()

    gt    = ds.GetGeoTransform()
    west  = gt[0];  north = gt[3]
    xres  = gt[1];  yres  = abs(gt[5])
    nrows, ncols = data.shape
    south = north - nrows * yres
    east  = west  + ncols * xres
    ds    = None

    if nodata is not None:
        data[data == nodata] = 0.0
    data[data < 0] = 0.0

    bounds = [[south, west], [north, east]]
    return data, bounds


# ---------------------------------------------------------------------------
# Step 5: numpy → PNG bytes
# ---------------------------------------------------------------------------
def array_to_png_bytes(data_mm: np.ndarray) -> bytes:
    rgba = mm_to_rgba(data_mm)
    img  = Image.fromarray(rgba, mode="RGBA")
    w, h = img.size
    if w > 2000:
        img = img.resize((w // 4, h // 4), Image.LANCZOS)
    import io
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Parse UTC timestamp from MRMS filename
# ---------------------------------------------------------------------------
def parse_timestamp(filename: str) -> str:
    """Extract YYYYMMDD-HHMMSS and return ISO-8601 UTC string."""
    m = re.search(r'(\d{8})-(\d{6})', filename)
    if not m:
        return ""
    d, t = m.group(1), m.group(2)
    dt = datetime.datetime(
        int(d[:4]), int(d[4:6]), int(d[6:8]),
        int(t[:2]), int(t[2:4]), int(t[4:6]),
        tzinfo=datetime.timezone.utc
    )
    return dt.isoformat()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    s3        = make_s3()
    filenames = find_latest_filenames(NUM_HOURS)
    index     = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for seq, filename in enumerate(filenames):
            tag = f"{seq:02d}"
            print(f"\n=== Processing [{seq+1}/{len(filenames)}] seq={tag}  {filename} ===")

            # Download + decompress
            grib2_path = download_and_decompress(filename, tmpdir)

            # GRIB2 → GeoTIFF
            tiff_path = os.path.join(tmpdir, f"mrms_{tag}.tif")
            grib2_to_geotiff(grib2_path, tiff_path)

            # GeoTIFF → array + bounds
            data_mm, bounds = geotiff_to_array(tiff_path)

            # Render PNG bytes
            png_bytes = array_to_png_bytes(data_mm)

            # Read GeoTIFF bytes
            with open(tiff_path, "rb") as f:
                tiff_bytes = f.read()

            # Read GRIB2 bytes
            with open(grib2_path, "rb") as f:
                grib2_bytes = f.read()

            # Upload to R2
            png_key    = f"mrms/hourly/png/mrms_{tag}.png"
            tif_key    = f"mrms/hourly/geotiff/mrms_{tag}.tif"
            grib2_key  = f"mrms/hourly/grib2/mrms_{tag}.grib2"

            png_url   = upload_bytes(s3, png_key,   png_bytes,  "image/png")
            tif_url   = upload_bytes(s3, tif_key,   tiff_bytes, "image/tiff")
            grib2_url = upload_bytes(s3, grib2_key, grib2_bytes,"application/octet-stream")

            ts = parse_timestamp(filename)
            print(f"    PNG  → {png_url}")
            print(f"    TIF  → {tif_url}")
            print(f"    GRB2 → {grib2_url}")

            index.append({
                "sequence_order":    seq,
                "timestamp_utc":     ts,
                "filename_source":   filename,
                "image_url":         png_url,
                "geotiff_url":       tif_url,
                "grib2_url":         grib2_url,
                "bounds":            bounds,
                "accumulation_hours": 1,
                "product":           f"MRMS_{MRMS_PRODUCT}",
                "units":             "inches",
            })

            # Keep legacy mrms_latest.png pointing to the most recent file
            if seq == len(filenames) - 1:
                print("  Updating legacy mrms_latest.png …")
                upload_bytes(s3, "mrms_latest.png", png_bytes, "image/png")

    # Upload index.json
    index_key = "mrms/hourly/index.json"
    index_payload = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "num_hours":     len(index),
        "files":         index,
    }
    index_url = upload_bytes(
        s3, index_key,
        json.dumps(index_payload, indent=2).encode(),
        "application/json",
    )

    print("\n=== 24-hour MRMS archive complete ===")
    print(f"  Index URL  : {index_url}")
    print(f"  Latest PNG : {R2_PUBLIC_BASE_URL}/mrms_latest.png")
    print(f"  Hours      : {len(index)}")


if __name__ == "__main__":
    main()
