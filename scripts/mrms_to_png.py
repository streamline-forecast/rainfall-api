#!/usr/bin/env python3

import os
import re
import gzip
import json
import tempfile
import datetime
import subprocess
import urllib.request
import io

import numpy as np
from PIL import Image
import boto3
from botocore.config import Config

MRMS_BASE_URL = "https://mrms.ncep.noaa.gov/2D/MultiSensor_QPE_01H_Pass2/"
MRMS_PRODUCT = "MultiSensor_QPE_01H_Pass2"
NUM_HOURS = 24
ACCUM_DURATIONS = [1, 3, 6, 12, 24]

R2_ACCESS_KEY_ID = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET_NAME = os.environ["R2_BUCKET_NAME"]
R2_PUBLIC_BASE_URL = os.environ["R2_PUBLIC_BASE_URL"].rstrip("/")
R2_ENDPOINT_URL = os.environ["R2_ENDPOINT_URL"].rstrip("/")

COLORMAP_MM = [
    (0.0, (0, 0, 0, 0)),
    (0.254, (100, 200, 255, 160)),
    (6.35, (50, 150, 255, 190)),
    (12.7, (30, 80, 220, 210)),
    (25.4, (80, 30, 200, 220)),
    (38.1, (160, 20, 180, 230)),
    (63.5, (220, 30, 80, 235)),
    (101.6, (180, 10, 10, 245)),
    (float("inf"), (100, 0, 0, 255)),
]


def mm_to_rgba(data_mm):
    h, w = data_mm.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for i in range(len(COLORMAP_MM) - 1):
        lo, color = COLORMAP_MM[i]
        hi, _ = COLORMAP_MM[i + 1]
        rgba[(data_mm >= lo) & (data_mm < hi)] = color
    return rgba


def make_s3():
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def upload_bytes(s3, key, data, content_type):
    s3.put_object(
        Bucket=R2_BUCKET_NAME,
        Key=key,
        Body=data,
        ContentType=content_type,
        CacheControl="public, max-age=3600",
    )
    return f"{R2_PUBLIC_BASE_URL}/{key}"


def find_latest_filenames(n=NUM_HOURS):
    print("Fetching MRMS directory listing…")
    with urllib.request.urlopen(MRMS_BASE_URL, timeout=30) as resp:
        html = resp.read().decode()

    pattern = re.compile(
        r"MRMS_MultiSensor_QPE_01H_Pass2_00\.00_\d{8}-\d{6}\.grib2\.gz"
    )

    matches = sorted(set(pattern.findall(html)))

    if not matches:
        raise RuntimeError("No .grib2.gz files found in MRMS directory listing.")

    latest = matches[-n:]

    print(f"Found {len(latest)} MRMS files")
    for f in latest:
        print(f"  {f}")

    return latest


def download_and_decompress(filename, tmpdir):
    url = MRMS_BASE_URL + filename
    gz_path = os.path.join(tmpdir, filename)
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


def grib2_to_geotiff(grib2_path, tiff_path):
    cmd = ["gdal_translate", "-of", "GTiff", "-b", "1", grib2_path, tiff_path]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"gdal_translate failed:\n{result.stderr[-1200:]}")

    return tiff_path


def geotiff_to_array(tiff_path):
    from osgeo import gdal

    gdal.UseExceptions()

    ds = gdal.Open(tiff_path)
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray().astype(np.float32)
    nodata = band.GetNoDataValue()
    gt = ds.GetGeoTransform()
    projection = ds.GetProjection()

    west = gt[0]
    north = gt[3]
    xres = gt[1]
    yres = abs(gt[5])
    nrows, ncols = data.shape
    south = north - nrows * yres
    east = west + ncols * xres

    ds = None

    if nodata is not None:
        data[data == nodata] = 0.0

    data[np.isnan(data)] = 0.0
    data[data < 0] = 0.0

    bounds = [[south, west], [north, east]]
    return data, bounds, gt, projection


def write_array_to_geotiff(array_mm, output_path, geotransform, projection):
    from osgeo import gdal

    rows, cols = array_mm.shape
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32)

    ds.SetGeoTransform(geotransform)
    ds.SetProjection(projection)

    band = ds.GetRasterBand(1)
    band.WriteArray(array_mm.astype(np.float32))
    band.SetNoDataValue(0.0)
    band.FlushCache()

    ds.FlushCache()
    ds = None

    return output_path


def array_to_png_bytes(data_mm):
    rgba = mm_to_rgba(data_mm)
    img = Image.fromarray(rgba, mode="RGBA")

    w, h = img.size
    if w > 2000:
        img = img.resize((w // 4, h // 4), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def parse_timestamp(filename):
    m = re.search(r"(\d{8})-(\d{6})", filename)
    if not m:
        return ""

    d, t = m.group(1), m.group(2)
    dt = datetime.datetime(
        int(d[:4]),
        int(d[4:6]),
        int(d[6:8]),
        int(t[:2]),
        int(t[2:4]),
        int(t[4:6]),
        tzinfo=datetime.timezone.utc,
    )
    return dt.isoformat()


def build_accumulations(s3, hourly_arrays, hourly_index, geotransform, projection, bounds, tmpdir):
    accum_index = []

    for duration in ACCUM_DURATIONS:
        if len(hourly_arrays) < duration:
            print(f"Skipping {duration}h accumulation; not enough hourly rasters.")
            continue

        selected_arrays = hourly_arrays[-duration:]
        selected_records = hourly_index[-duration:]

        accum_mm = np.sum(selected_arrays, axis=0).astype(np.float32)

        duration_tag = f"{duration:02d}h"
        png_key = f"mrms/accum/png/accum_{duration_tag}.png"
        tif_key = f"mrms/accum/geotiff/accum_{duration_tag}.tif"

        png_bytes = array_to_png_bytes(accum_mm)

        accum_tif_path = os.path.join(tmpdir, f"accum_{duration_tag}.tif")
        write_array_to_geotiff(accum_mm, accum_tif_path, geotransform, projection)

        with open(accum_tif_path, "rb") as f:
            tif_bytes = f.read()

        png_url = upload_bytes(s3, png_key, png_bytes, "image/png")
        tif_url = upload_bytes(s3, tif_key, tif_bytes, "image/tiff")

        max_mm = float(np.nanmax(accum_mm))
        max_inches = max_mm / 25.4

        record = {
            "duration_hours": duration,
            "start_time_utc": selected_records[0]["timestamp_utc"],
            "end_time_utc": selected_records[-1]["timestamp_utc"],
            "image_url": png_url,
            "geotiff_url": tif_url,
            "bounds": bounds,
            "units": "mm",
            "max_rainfall_mm": round(max_mm, 3),
            "max_rainfall_inches": round(max_inches, 3),
            "product": f"MRMS_{MRMS_PRODUCT}",
        }

        accum_index.append(record)

        print(f"  Accum {duration_tag} PNG → {png_url}")
        print(f"  Accum {duration_tag} TIF → {tif_url}")

    payload = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "durations": ACCUM_DURATIONS,
        "files": accum_index,
    }

    index_url = upload_bytes(
        s3,
        "mrms/accum/index.json",
        json.dumps(payload, indent=2).encode(),
        "application/json",
    )

    print(f"  Accum index → {index_url}")
    return index_url


def main():
    s3 = make_s3()
    filenames = find_latest_filenames(NUM_HOURS)

    hourly_index = []
    hourly_arrays = []
    latest_bounds = None
    latest_geotransform = None
    latest_projection = None

    with tempfile.TemporaryDirectory() as tmpdir:
        for seq, filename in enumerate(filenames):
            tag = f"{seq:02d}"
            print(f"\n=== Processing [{seq + 1}/{len(filenames)}] seq={tag} {filename} ===")

            grib2_path = download_and_decompress(filename, tmpdir)

            tiff_path = os.path.join(tmpdir, f"mrms_{tag}.tif")
            grib2_to_geotiff(grib2_path, tiff_path)

            data_mm, bounds, geotransform, projection = geotiff_to_array(tiff_path)

            hourly_arrays.append(data_mm)
            latest_bounds = bounds
            latest_geotransform = geotransform
            latest_projection = projection

            png_bytes = array_to_png_bytes(data_mm)

            with open(tiff_path, "rb") as f:
                tiff_bytes = f.read()

            with open(grib2_path, "rb") as f:
                grib2_bytes = f.read()

            png_key = f"mrms/hourly/png/mrms_{tag}.png"
            tif_key = f"mrms/hourly/geotiff/mrms_{tag}.tif"
            grib2_key = f"mrms/hourly/grib2/mrms_{tag}.grib2"

            png_url = upload_bytes(s3, png_key, png_bytes, "image/png")
            tif_url = upload_bytes(s3, tif_key, tiff_bytes, "image/tiff")
            grib2_url = upload_bytes(s3, grib2_key, grib2_bytes, "application/octet-stream")

            ts = parse_timestamp(filename)

            record = {
                "sequence_order": seq,
                "timestamp_utc": ts,
                "filename_source": filename,
                "image_url": png_url,
                "geotiff_url": tif_url,
                "grib2_url": grib2_url,
                "bounds": bounds,
                "accumulation_hours": 1,
                "product": f"MRMS_{MRMS_PRODUCT}",
                "units": "mm",
            }

            hourly_index.append(record)

            print(f"    PNG  → {png_url}")
            print(f"    TIF  → {tif_url}")
            print(f"    GRB2 → {grib2_url}")

            if seq == len(filenames) - 1:
                print("  Updating legacy mrms_latest.png …")
                upload_bytes(s3, "mrms_latest.png", png_bytes, "image/png")

        hourly_payload = {
            "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "num_hours": len(hourly_index),
            "files": hourly_index,
        }

        hourly_index_url = upload_bytes(
            s3,
            "mrms/hourly/index.json",
            json.dumps(hourly_payload, indent=2).encode(),
            "application/json",
        )

        print("\n=== Building accumulation rasters ===")
        build_accumulations(
            s3=s3,
            hourly_arrays=hourly_arrays,
            hourly_index=hourly_index,
            geotransform=latest_geotransform,
            projection=latest_projection,
            bounds=latest_bounds,
            tmpdir=tmpdir,
        )

    print("\n=== MRMS archive complete ===")
    print(f"  Hourly index: {hourly_index_url}")
    print(f"  Accum index : {R2_PUBLIC_BASE_URL}/mrms/accum/index.json")
    print(f"  Latest PNG  : {R2_PUBLIC_BASE_URL}/mrms_latest.png")
    print(f"  Hours       : {len(hourly_index)}")


if __name__ == "__main__":
    main()
