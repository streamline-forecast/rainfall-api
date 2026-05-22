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

RUN_VERSION = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S")

COLORMAP_MM = [
    (0.0, (0, 0, 0, 0)),              # No rain / transparent
    (0.254, (186, 224, 255, 160)),    # 0.01–0.25 in Light Total
    (6.35, (120, 170, 255, 175)),     # 0.25–0.50 in Minor Total
    (12.7, (70, 95, 220, 190)),       # 0.50–1.00 in Moderate Total
    (25.4, (125, 70, 210, 205)),      # 1.00–2.00 in Heavy Total
    (50.8, (205, 70, 190, 220)),      # 2.00–4.00 in Very Heavy Total
    (101.6, (210, 60, 85, 235)),      # 4.00–8.00 in Extreme Total
    (203.2, (100, 0, 0, 255)),        # >8.00 in Exceptional Total
    (float("inf"), (100, 0, 0, 255)),
]


def version_url(url):
    return f"{url}?v={RUN_VERSION}"


def mm_to_rgba(data_mm):
    h, w = data_mm.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    valid = np.isfinite(data_mm) & (data_mm >= 0) & (data_mm < 2000)

    for i in range(len(COLORMAP_MM) - 1):
        lo, color = COLORMAP_MM[i]
        hi, _ = COLORMAP_MM[i + 1]
        mask = valid & (data_mm >= lo) & (data_mm < hi)
        rgba[mask] = color

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
        CacheControl="no-cache, no-store, must-revalidate",
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


def download_and_decompress(filename, tmpdir, max_attempts=5):
    url = MRMS_BASE_URL + filename
    gz_path = os.path.join(tmpdir, filename)
    grib2_path = gz_path[:-3]

    for attempt in range(1, max_attempts + 1):
        try:
            print(f"  Downloading {filename} attempt {attempt}/{max_attempts} …")

            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 GitHubActions-RainfallPipeline/1.0",
                    "Accept": "*/*",
                },
            )

            with urllib.request.urlopen(req, timeout=180) as resp:
                with open(gz_path, "wb") as f:
                    f.write(resp.read())

            with gzip.open(gz_path, "rb") as gz_in, open(grib2_path, "wb") as out:
                out.write(gz_in.read())

            if os.path.exists(gz_path):
                os.remove(gz_path)

            print(f"    {os.path.getsize(grib2_path):,} bytes (.grib2)")
            return grib2_path

        except Exception as e:
            print(f"    WARNING: download failed for {filename}: {e}")

            if attempt == max_attempts:
                raise

            import time
            time.sleep(10 * attempt)


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
    from osgeo import osr

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())

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

    print("PNG native dimensions:", img.size)

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


def print_accum_debug(duration_tag, accum_mm):
    max_mm = float(np.nanmax(accum_mm))
    min_mm = float(np.nanmin(accum_mm))

    print(f"DEBUG accum_{duration_tag}")
    print("  shape:", accum_mm.shape)
    print("  min_mm:", min_mm)
    print("  max_mm:", max_mm)
    print("  max_inches:", max_mm / 25.4)


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
        png_key = f"mrms/accum/png/accum_{duration_tag}_{RUN_VERSION}.png"
        tif_key = f"mrms/accum/geotiff/accum_{duration_tag}_{RUN_VERSION}.tif"

        print_accum_debug(duration_tag, accum_mm)

        print("MRMS ACCUM projection:")
        print(projection)

        print("MRMS ACCUM geotransform:")
        print(geotransform)

        print("MRMS ACCUM bounds:")
        print(bounds)

        print("MRMS ACCUM shape:")
        print(accum_mm.shape)
    
        png_bytes = array_to_png_bytes(accum_mm)

        accum_tif_path = os.path.join(tmpdir, f"accum_{duration_tag}.tif")
        write_array_to_geotiff(accum_mm, accum_tif_path, geotransform, projection)

        with open(accum_tif_path, "rb") as f:
            tif_bytes = f.read()

        png_url_raw = upload_bytes(s3, png_key, png_bytes, "image/png")
        tif_url_raw = upload_bytes(s3, tif_key, tif_bytes, "image/tiff")

        png_url = version_url(png_url_raw)
        tif_url = version_url(tif_url_raw)

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
            "version": RUN_VERSION,
        }

        accum_index.append(record)

        print(f"  Accum {duration_tag} PNG → {png_url}")
        print(f"  Accum {duration_tag} TIF → {tif_url}")

    payload = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "version": RUN_VERSION,
        "durations": ACCUM_DURATIONS,
        "files": accum_index,
    }

    index_url_raw = upload_bytes(
        s3,
        "mrms/accum/index.json",
        json.dumps(payload, indent=2).encode(),
        "application/json",
    )

    index_url = version_url(index_url_raw)

    print(f"  Accum index → {index_url}")
    return index_url


def main():
    print("RUN_VERSION:", RUN_VERSION)

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

            print("MRMS projection:")
            print(projection)

            print("MRMS geotransform:")
            print(geotransform)

            print("MRMS bounds:")
            print(bounds)

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

            png_url_raw = upload_bytes(s3, png_key, png_bytes, "image/png")
            tif_url_raw = upload_bytes(s3, tif_key, tiff_bytes, "image/tiff")
            grib2_url_raw = upload_bytes(s3, grib2_key, grib2_bytes, "application/octet-stream")

            png_url = version_url(png_url_raw)
            tif_url = version_url(tif_url_raw)
            grib2_url = version_url(grib2_url_raw)

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
                "version": RUN_VERSION,
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
            "version": RUN_VERSION,
            "num_hours": len(hourly_index),
            "files": hourly_index,
        }

        hourly_index_url_raw = upload_bytes(
            s3,
            "mrms/hourly/index.json",
            json.dumps(hourly_payload, indent=2).encode(),
            "application/json",
        )

        hourly_index_url = version_url(hourly_index_url_raw)

        print("\n=== Building accumulation rasters ===")
        accum_index_url = build_accumulations(
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
    print(f"  Accum index : {accum_index_url}")
    print(f"  Latest PNG  : {R2_PUBLIC_BASE_URL}/mrms_latest.png?v={RUN_VERSION}")
    print(f"  Hours       : {len(hourly_index)}")


if __name__ == "__main__":
    main()
