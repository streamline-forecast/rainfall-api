#!/usr/bin/env python3

import os
import io
import json
import tempfile
import datetime
import urllib.request

import numpy as np
from PIL import Image
from osgeo import gdal
import boto3
from botocore.config import Config

gdal.UseExceptions()

PRODUCT_NAME = "42-Hour Flood Outlook Accumulation"
PRODUCT_ID = "combined_42h"

MRMS_INDEX_URL = "https://pub-74bd4bf29f244211bef1f4e1faf616d6.r2.dev/mrms/hourly/index.json"
HRRR_INDEX_URL = "https://pub-74bd4bf29f244211bef1f4e1faf616d6.r2.dev/hrrr/forecast/index.json"

R2_ACCESS_KEY_ID = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET_NAME = os.environ["R2_BUCKET_NAME"]
R2_PUBLIC_BASE_URL = os.environ["R2_PUBLIC_BASE_URL"].rstrip("/")
R2_ENDPOINT_URL = os.environ["R2_ENDPOINT_URL"].rstrip("/")

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

def fetch_json(url):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 GitHubActions-RainfallPipeline/1.0",
            "Accept": "application/json,text/plain,*/*",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def download_file(url, path):
    print(f"Downloading {url}")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 GitHubActions-RainfallPipeline/1.0",
            "Accept": "*/*",
        },
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        with open(path, "wb") as file:
            file.write(response.read())
    return path


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


def array_to_png_bytes(data_mm):
    rgba = mm_to_rgba(data_mm)
    img = Image.fromarray(rgba, mode="RGBA")

    print("PNG native dimensions:", img.size)

    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def read_raster_array(path):
    ds = gdal.Open(path)
    band = ds.GetRasterBand(1)

    arr = band.ReadAsArray().astype(np.float32)
    nodata = band.GetNoDataValue()

    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    width = ds.RasterXSize
    height = ds.RasterYSize

    west = gt[0]
    north = gt[3]
    xres = gt[1]
    yres = abs(gt[5])
    south = north - height * yres
    east = west + width * xres

    if nodata is not None:
        arr[arr == nodata] = 0.0

    arr[np.isnan(arr)] = 0.0
    arr[arr < 0] = 0.0

    bounds = [[south, west], [north, east]]

    ds = None
    return arr, gt, proj, width, height, bounds


def warp_to_web_mercator(src_path, dst_path):
    options = gdal.WarpOptions(
        format="GTiff",
        dstSRS="EPSG:3857",
        resampleAlg="near",
        errorThreshold=0.0,
        multithread=True,
        creationOptions=["COMPRESS=LZW", "TILED=YES"],
    )

    ds = gdal.Warp(dst_path, src_path, options=options)

    if ds is None:
        raise RuntimeError(
            f"Failed to warp raster to EPSG:3857: {src_path}"
        )

    ds.FlushCache()
    ds = None

    return dst_path

def reproject_to_reference(src_path, dst_path, ref_gt, ref_proj, ref_width, ref_height):
    minx = ref_gt[0]
    maxy = ref_gt[3]
    maxx = minx + ref_gt[1] * ref_width
    miny = maxy + ref_gt[5] * ref_height

    options = gdal.WarpOptions(
        format="GTiff",
        dstSRS=ref_proj,
        outputBounds=(minx, miny, maxx, maxy),
        width=ref_width,
        height=ref_height,
        resampleAlg="bilinear",
        srcNodata=0,
        dstNodata=0,
    )

    gdal.Warp(dst_path, src_path, options=options)
    return dst_path


def write_geotiff(path, array_mm, gt, proj):
    rows, cols = array_mm.shape
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(path, cols, rows, 1, gdal.GDT_Float32)

    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)

    band = ds.GetRasterBand(1)
    band.WriteArray(array_mm.astype(np.float32))
    band.SetNoDataValue(0.0)
    band.FlushCache()

    ds.FlushCache()
    ds = None

    return path


def print_waxahachie_debug(accumulator, ref_gt):
    lat = 32.36170
    lon = -96.89100

    west = ref_gt[0]
    north = ref_gt[3]
    xres = ref_gt[1]
    yres = abs(ref_gt[5])

    col = int((lon - west) / xres)
    row = int((north - lat) / yres)

    height, width = accumulator.shape

    print("WAXAHACHIE DEBUG")
    print("shape:", accumulator.shape)
    print("row:", row)
    print("col:", col)

    if row < 0 or row >= height or col < 0 or col >= width:
        print("Waxahachie point is outside raster bounds")
        return

    test_mm = float(accumulator[row, col])
    test_rgba = mm_to_rgba(np.array([[test_mm]], dtype=np.float32))[0, 0].tolist()

    print("mm:", test_mm)
    print("inches:", test_mm / 25.4)
    print("rgba:", test_rgba)


def main():
    print("Fetching indexes")
    mrms_index = fetch_json(MRMS_INDEX_URL)
    hrrr_index = fetch_json(HRRR_INDEX_URL)

    mrms_records = (mrms_index.get("files") or [])[-24:]
    hrrr_records = (hrrr_index.get("forecast_hours") or [])[:18]

    if len(mrms_records) < 1:
        raise RuntimeError("No MRMS records found.")

    if len(hrrr_records) < 1:
        raise RuntimeError("No HRRR records found.")

    print(f"MRMS records: {len(mrms_records)}")
    print(f"HRRR records: {len(hrrr_records)}")

    all_records = []

    for record in mrms_records:
        all_records.append({
            "source": "mrms",
            "time": record.get("timestamp_utc"),
            "geotiff_url": record.get("geotiff_url"),
        })

    for record in hrrr_records:
        all_records.append({
            "source": "hrrr",
            "time": record.get("valid_time_utc"),
            "geotiff_url": record.get("geotiff_url"),
        })

    if any(not r["geotiff_url"] for r in all_records):
        raise RuntimeError("One or more records is missing geotiff_url.")

    s3 = make_s3()

    with tempfile.TemporaryDirectory() as tmpdir:
        ref_path = os.path.join(tmpdir, "reference_mrms.tif")
        download_file(mrms_records[0]["geotiff_url"], ref_path)

        ref_arr, ref_gt, ref_proj, ref_width, ref_height, bounds = read_raster_array(ref_path)
        accumulator = np.zeros_like(ref_arr, dtype=np.float32)

        for i, record in enumerate(all_records):
            source = record["source"]
            url = record["geotiff_url"]

            raw_path = os.path.join(tmpdir, f"raw_{i:02d}_{source}.tif")
            aligned_path = os.path.join(tmpdir, f"aligned_{i:02d}_{source}.tif")

            download_file(url, raw_path)

            if i == 0:
                aligned_path = raw_path
            else:
                reproject_to_reference(
                    src_path=raw_path,
                    dst_path=aligned_path,
                    ref_gt=ref_gt,
                    ref_proj=ref_proj,
                    ref_width=ref_width,
                    ref_height=ref_height,
                )

            arr, _, _, _, _, _ = read_raster_array(aligned_path)
            accumulator += arr

            print(f"Added {i + 1}/{len(all_records)}: {source} {record['time']}")

        max_mm = float(np.nanmax(accumulator))
        max_inches = max_mm / 25.4

        print("Accumulator shape:", accumulator.shape)
        print("Accumulator min:", float(np.nanmin(accumulator)))
        print("Accumulator max:", max_mm)

        combined_tif_path = os.path.join(tmpdir, "combined_42h.tif")
        write_geotiff(combined_tif_path, accumulator, ref_gt, ref_proj)

        display_png_tif_path = os.path.join(tmpdir, "combined_42h_png3857.tif")

        warp_to_web_mercator(
            combined_tif_path,
            display_png_tif_path,
        )

        png_mm, png_gt, png_projection, png_width, png_height, png_bounds = read_raster_array(
            display_png_tif_path
        )

        png_bytes = array_to_png_bytes(png_mm)

        print_waxahachie_debug(accumulator, ref_gt)

        with open(combined_tif_path, "rb") as f:
            tif_bytes = f.read()

        image_url = upload_bytes(
            s3,
            "combined/accum/png/combined_42h.png",
            png_bytes,
            "image/png",
        )

        geotiff_url = upload_bytes(
            s3,
            "combined/accum/geotiff/combined_42h.tif",
            tif_bytes,
            "image/tiff",
        )

        start_time = mrms_records[0].get("timestamp_utc")
        transition_time = hrrr_index.get("cycle_time_utc") or mrms_records[-1].get("timestamp_utc")
        end_time = hrrr_records[-1].get("valid_time_utc")

        payload = {
            "product_name": PRODUCT_NAME,
            "product_id": PRODUCT_ID,
            "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "observed_hours": len(mrms_records),
            "forecast_hours": len(hrrr_records),
            "total_hours": len(mrms_records) + len(hrrr_records),
            "start_time_utc": start_time,
            "transition_time_utc": transition_time,
            "end_time_utc": end_time,
            "image_url": image_url,
            "geotiff_url": geotiff_url,
            "bounds": png_bounds,
            "sample_bounds": bounds,
            "units": "mm",
            "max_rainfall_mm": round(max_mm, 3),
            "max_rainfall_inches": round(max_inches, 3),
            "sources": ["MRMS observed", "HRRR forecast"],
            "note": "Combined accumulation uses latest 24 MRMS observed GeoTIFFs plus first 18 HRRR forecast GeoTIFFs.",
        }

        index_url = upload_bytes(
            s3,
            "combined/accum/index.json",
            json.dumps(payload, indent=2).encode("utf-8"),
            "application/json",
        )

        print("Combined 42-hour accumulation complete")
        print(f"Index URL: {index_url}")
        print(f"PNG URL: {image_url}")
        print(f"GeoTIFF URL: {geotiff_url}")
        print(f"Max rainfall: {max_mm:.3f} mm / {max_inches:.3f} in")


if __name__ == "__main__":
    main()
