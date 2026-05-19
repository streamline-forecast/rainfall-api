#!/usr/bin/env python3

import os
import gzip
import tempfile
import subprocess
import urllib.request

import numpy as np
from PIL import Image
import boto3
from botocore.config import Config


MRMS_BASE_URL = "https://mrms.ncep.noaa.gov/2D/MultiSensor_QPE_01H_Pass2/"
MRMS_PRODUCT = "MultiSensor_QPE_01H_Pass2"
MRMS_FILENAME = f"MRMS_{MRMS_PRODUCT}.latest.grib2.gz"

R2_ACCESS_KEY_ID = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET_NAME = os.environ["R2_BUCKET_NAME"]
R2_PUBLIC_BASE_URL = os.environ["R2_PUBLIC_BASE_URL"].rstrip("/")
R2_ENDPOINT_URL = os.environ["R2_ENDPOINT_URL"].rstrip("/")

OUTPUT_KEY = "mrms_latest.png"


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


def download_and_decompress(tmpdir: str) -> str:
    url = MRMS_BASE_URL + MRMS_FILENAME
    gz_path = os.path.join(tmpdir, MRMS_FILENAME)
    grib2_path = gz_path.replace(".gz", "")

    print(f"Downloading {url}")

    with urllib.request.urlopen(url, timeout=120) as response:
        with open(gz_path, "wb") as file:
            file.write(response.read())

    print(f"Downloaded .gz size: {os.path.getsize(gz_path):,} bytes")

    with gzip.open(gz_path, "rb") as gz_in:
        with open(grib2_path, "wb") as grib_out:
            grib_out.write(gz_in.read())

    print(f"Decompressed GRIB2 size: {os.path.getsize(grib2_path):,} bytes")

    os.remove(gz_path)

    return grib2_path


def grib2_to_geotiff(grib2_path: str, tmpdir: str) -> str:
    tiff_path = os.path.join(tmpdir, "mrms.tif")

    cmd = [
        "gdal_translate",
        "-of",
        "GTiff",
        "-b",
        "1",
        grib2_path,
        tiff_path,
    ]

    print("Running:", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout[-1000:])

    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise RuntimeError(f"gdal_translate failed with exit code {result.returncode}")

    print(f"GeoTIFF size: {os.path.getsize(tiff_path):,} bytes")

    return tiff_path


def geotiff_to_array(tiff_path: str):
    from osgeo import gdal

    gdal.UseExceptions()

    dataset = gdal.Open(tiff_path)
    if dataset is None:
        raise RuntimeError("GDAL could not open GeoTIFF.")

    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray().astype(np.float32)
    nodata = band.GetNoDataValue()

    transform = dataset.GetGeoTransform()

    west = transform[0]
    north = transform[3]
    xres = transform[1]
    yres = abs(transform[5])

    rows, cols = data.shape

    south = north - rows * yres
    east = west + cols * xres

    dataset = None

    if nodata is not None:
        data[data == nodata] = 0.0

    data[np.isnan(data)] = 0.0
    data[data < 0] = 0.0

    print(f"Raster size: {cols} x {rows}")
    print(f"Bounds: south={south}, west={west}, north={north}, east={east}")
    print(f"Rainfall range mm: min={data.min():.3f}, max={data.max():.3f}")

    bounds_leaflet = [[south, west], [north, east]]

    return data, bounds_leaflet


def mm_to_rgba(data_mm: np.ndarray) -> np.ndarray:
    height, width = data_mm.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)

    for index in range(len(COLORMAP_MM) - 1):
        low = COLORMAP_MM[index][0]
        high = COLORMAP_MM[index + 1][0]
        color = COLORMAP_MM[index][1]

        mask = (data_mm >= low) & (data_mm < high)
        rgba[mask] = color

    return rgba


def array_to_png(data_mm: np.ndarray, png_path: str):
    print("Rendering transparent rainfall PNG")

    rgba = mm_to_rgba(data_mm)
    image = Image.fromarray(rgba, mode="RGBA")

    width, height = image.size

    if width > 2200:
        new_width = width // 4
        new_height = height // 4
        image = image.resize((new_width, new_height), Image.LANCZOS)
        print(f"Downsampled PNG to {new_width} x {new_height}")

    image.save(png_path, "PNG", optimize=True)

    print(f"Saved PNG: {png_path}")
    print(f"PNG size: {os.path.getsize(png_path):,} bytes")


def upload_to_r2(png_path: str) -> str:
    s3 = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )

    print(f"Uploading {OUTPUT_KEY} to R2 bucket {R2_BUCKET_NAME}")

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

    print(f"Public URL: {public_url}")

    return public_url


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        grib2_path = download_and_decompress(tmpdir)
        tiff_path = grib2_to_geotiff(grib2_path, tmpdir)

        data_mm, bounds = geotiff_to_array(tiff_path)

        png_path = os.path.join(tmpdir, OUTPUT_KEY)
        array_to_png(data_mm, png_path)

        public_url = upload_to_r2(png_path)

    print("MRMS PNG pipeline complete")
    print(f"PNG URL: {public_url}")
    print(f"Leaflet bounds: {bounds}")


if __name__ == "__main__":
    main()
