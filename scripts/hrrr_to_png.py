#!/usr/bin/env python3

import os
import re
import io
import json
import tempfile
import datetime
import subprocess
import urllib.request
import urllib.error

import numpy as np
from PIL import Image
import boto3
from botocore.config import Config


HRRR_BUCKET_BASE = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
DOMAIN = "conus"
PRODUCT = "wrfsfcf"

FORECAST_HOURS = list(range(1, 19))
ACCUM_DURATIONS = [1, 3, 6, 12, 18]

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


def hrrr_urls(cycle_date, cycle_hour, fhour):
    ymd = cycle_date.strftime("%Y%m%d")
    hh = f"{cycle_hour:02d}"
    ff = f"{fhour:02d}"

    base = (
        f"{HRRR_BUCKET_BASE}/hrrr.{ymd}/{DOMAIN}/"
        f"hrrr.t{hh}z.{PRODUCT}{ff}.grib2"
    )

    return base, base + ".idx"


def url_exists(url):
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status == 200
    except Exception:
        return False


def choose_latest_cycle():
    now = datetime.datetime.now(datetime.timezone.utc)
    candidate = now - datetime.timedelta(minutes=90)
    candidate = candidate.replace(minute=0, second=0, microsecond=0)

    for back in range(0, 8):
        dt = candidate - datetime.timedelta(hours=back)
        cycle_date = dt.date()
        cycle_hour = dt.hour

        _, idx_url = hrrr_urls(cycle_date, cycle_hour, 1)

        print(f"Checking HRRR cycle {dt.isoformat()} → {idx_url}")

        if url_exists(idx_url):
            print(f"Selected HRRR cycle: {dt.isoformat()}")
            return dt

    raise RuntimeError("No available HRRR cycle found in the last 8 hours.")


def fetch_text(url):
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def parse_idx_for_apcp(idx_text):
    lines = [line.strip() for line in idx_text.splitlines() if line.strip()]
    records = []

    for line in lines:
        parts = line.split(":")
        if len(parts) < 5:
            continue

        try:
            record_number = int(parts[0])
            byte_start = int(parts[1])
        except ValueError:
            continue

        variable = parts[3] if len(parts) > 3 else ""
        level = parts[4] if len(parts) > 4 else ""
        timing = ":".join(parts[5:]) if len(parts) > 5 else ""

        records.append(
            {
                "record_number": record_number,
                "byte_start": byte_start,
                "variable": variable,
                "level": level,
                "timing": timing,
                "line": line,
            }
        )

    for i, rec in enumerate(records):
        rec["byte_end"] = records[i + 1]["byte_start"] - 1 if i + 1 < len(records) else None

    apcp_records = [
        r for r in records
        if r["variable"] == "APCP" and "surface" in r["level"].lower()
    ]

    if not apcp_records:
        raise RuntimeError("No APCP surface record found in HRRR idx file.")

    return apcp_records[-1]


def download_byte_range(url, byte_start, byte_end, output_path):
    headers = {}

    if byte_end is not None:
        headers["Range"] = f"bytes={byte_start}-{byte_end}"
    else:
        headers["Range"] = f"bytes={byte_start}-"

    req = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()

    with open(output_path, "wb") as f:
        f.write(data)

    print(f"Downloaded byte range to {output_path}: {len(data):,} bytes")
    return output_path


def grib2_to_geotiff(grib2_path, tiff_path):
    cmd = ["gdal_translate", "-of", "GTiff", "-b", "1", grib2_path, tiff_path]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"gdal_translate failed:\n{result.stderr[-2000:]}")

    return tiff_path


def geotiff_to_array(tiff_path):
    from osgeo import gdal

    gdal.UseExceptions()

    ds = gdal.Open(tiff_path)
    if ds is None:
        raise RuntimeError(f"Could not open GeoTIFF: {tiff_path}")

    band = ds.GetRasterBand(1)
    data = band.ReadAsArray().astype(np.float32)
    nodata = band.GetNoDataValue()
    gt = ds.GetGeoTransform()
    projection = ds.GetProjection()

    west = gt[0]
    north = gt[3]
    xres = gt[1]
    yres = abs(gt[5])

    rows, cols = data.shape
    south = north - rows * yres
    east = west + cols * xres

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
    
def warp_to_display_grid(src_path, dst_path):
    from osgeo import gdal

    options = gdal.WarpOptions(
        format="GTiff",
        dstSRS="EPSG:4326",
        outputBounds=(-134.0, 20.0, -60.0, 55.0),
        width=7000,
        height=3500,
        resampleAlg="bilinear",
        srcNodata=0,
        dstNodata=0,
    )

    ds = gdal.Warp(dst_path, src_path, options=options)

    if ds is None:
        raise RuntimeError(f"Failed to warp raster to display grid: {src_path}")

    ds = None
    return dst_path

def valid_time(cycle_dt, fhour):
    return cycle_dt + datetime.timedelta(hours=fhour)


def print_accum_debug(tag, accum_mm):
    max_mm = float(np.nanmax(accum_mm))
    min_mm = float(np.nanmin(accum_mm))

    print(f"DEBUG HRRR accum_{tag}")
    print("  shape:", accum_mm.shape)
    print("  min_mm:", min_mm)
    print("  max_mm:", max_mm)
    print("  max_inches:", max_mm / 25.4)


def process_hrrr_forecast(s3, cycle_dt, tmpdir):
    forecast_records = []
    hourly_arrays = []

    previous_cumulative = None
    latest_bounds = None
    latest_geotransform = None
    latest_projection = None

    cycle_date = cycle_dt.date()
    cycle_hour = cycle_dt.hour

    for fhour in FORECAST_HOURS:
        print(f"\n=== HRRR F{fhour:02d} ===")

        grib_url, idx_url = hrrr_urls(cycle_date, cycle_hour, fhour)

        try:
            idx_text = fetch_text(idx_url)
            apcp = parse_idx_for_apcp(idx_text)

            apcp_grib_path = os.path.join(tmpdir, f"hrrr_f{fhour:02d}_apcp.grib2")
            cumulative_tif_path = os.path.join(tmpdir, f"hrrr_f{fhour:02d}_cumulative.tif")
            hourly_tif_path = os.path.join(tmpdir, f"hrrr_f{fhour:02d}.tif")

            print(f"APCP idx line: {apcp['line']}")

            download_byte_range(
                grib_url,
                apcp["byte_start"],
                apcp["byte_end"],
                apcp_grib_path,
            )

            grib2_to_geotiff(apcp_grib_path, cumulative_tif_path)

            cumulative_mm, bounds, geotransform, projection = geotiff_to_array(cumulative_tif_path)

            print("HRRR projection:")
            print(projection)

            print("HRRR geotransform:")
            print(geotransform)

            print("HRRR bounds:")
            print(bounds)

            if previous_cumulative is None:
                hourly_mm = cumulative_mm
            else:
                hourly_mm = cumulative_mm - previous_cumulative
                hourly_mm[hourly_mm < 0] = 0.0
                hourly_mm[np.isnan(hourly_mm)] = 0.0

            previous_cumulative = cumulative_mm.copy()

            native_hourly_tif_path = os.path.join(tmpdir, f"hrrr_f{fhour:02d}_native.tif")
            display_hourly_tif_path = hourly_tif_path

            write_array_to_geotiff(hourly_mm, native_hourly_tif_path, geotransform, projection)

            print(f"Warping HRRR F{fhour:02d} hourly to EPSG:4326 display grid")
            warp_to_display_grid(native_hourly_tif_path, display_hourly_tif_path)

            display_hourly_mm, display_bounds, display_gt, display_projection = geotiff_to_array(display_hourly_tif_path)

            print("DISPLAY hourly projection:")
            print(display_projection)
            print("DISPLAY hourly bounds:")
            print(display_bounds)

            png_bytes = array_to_png_bytes(display_hourly_mm)

            with open(display_hourly_tif_path, "rb") as f:
                tif_bytes = f.read()

            with open(apcp_grib_path, "rb") as f:
                grib_bytes = f.read()

            png_key = f"hrrr/forecast/png/hrrr_f{fhour:02d}.png"
            tif_key = f"hrrr/forecast/geotiff/hrrr_f{fhour:02d}.tif"
            grib_key = f"hrrr/forecast/grib2/hrrr_f{fhour:02d}_apcp.grib2"

            png_url_raw = upload_bytes(s3, png_key, png_bytes, "image/png")
            tif_url_raw = upload_bytes(s3, tif_key, tif_bytes, "image/tiff")
            grib_url_raw = upload_bytes(s3, grib_key, grib_bytes, "application/octet-stream")

            png_url = version_url(png_url_raw)
            tif_url = version_url(tif_url_raw)
            grib_url_public = version_url(grib_url_raw)

            vmax_mm = float(np.nanmax(display_hourly_mm))
            vmax_inches = vmax_mm / 25.4

            record = {
                "fhour": fhour,
                "cycle_time_utc": cycle_dt.isoformat(),
                "valid_time_utc": valid_time(cycle_dt, fhour).isoformat(),
                "image_url": png_url,
                "geotiff_url": tif_url,
                "grib2_url": grib_url_public,
                "bounds": display_bounds,
                "units": "mm",
                "hourly_max_mm": round(vmax_mm, 3),
                "hourly_max_inches": round(vmax_inches, 3),
                "source_grib_url": grib_url,
                "source_idx_url": idx_url,
                "apcp_idx_line": apcp["line"],
                "version": RUN_VERSION,
            }

            forecast_records.append(record)
            hourly_arrays.append(display_hourly_mm)
            latest_bounds = display_bounds
            latest_geotransform = display_gt
            latest_projection = display_projection

            print(f"PNG → {png_url}")
            print(f"TIF → {tif_url}")
            print(f"GRIB2 → {grib_url_public}")

        except Exception as e:
            print(f"WARNING: Skipping F{fhour:02d}: {e}")

    if not forecast_records:
        raise RuntimeError("No HRRR forecast hours processed successfully.")

    return (
        forecast_records,
        hourly_arrays,
        latest_bounds,
        latest_geotransform,
        latest_projection,
    )


def build_accumulations(
    s3,
    hourly_arrays,
    forecast_records,
    bounds,
    geotransform,
    projection,
    tmpdir,
):
    accum_records = []

    for duration in ACCUM_DURATIONS:
        if len(hourly_arrays) < duration:
            print(f"Skipping {duration}h accumulation; not enough forecast hours.")
            continue

        selected_arrays = hourly_arrays[:duration]
        selected_records = forecast_records[:duration]

        accum_mm = np.sum(selected_arrays, axis=0).astype(np.float32)

        tag = f"{duration:02d}h"
        png_key = f"hrrr/forecast/accum/png/accum_{tag}.png"
        tif_key = f"hrrr/forecast/accum/geotiff/accum_{tag}.tif"

        print_accum_debug(tag, accum_mm)

        tif_path = os.path.join(tmpdir, f"hrrr_accum_{tag}.tif")

        write_array_to_geotiff(accum_mm, tif_path, geotransform, projection)

        # hourly_arrays are already warped to EPSG:4326, so accum_mm is already display-grid aligned.
        display_mm, display_bounds, display_gt, display_projection = geotiff_to_array(tif_path)

        print("DISPLAY accumulation projection:")
        print(display_projection)
        print("DISPLAY accumulation bounds:")
        print(display_bounds)

        png_bytes = array_to_png_bytes(display_mm)

        if tag == "18h":
            test_row = 2251
            test_col = 3284

            test_mm = float(display_mm[test_row, test_col])

            test_rgba = mm_to_rgba(
                np.array([[test_mm]], dtype=np.float32)
            )[0, 0].tolist()

            print("HRRR 18H PIXEL DEBUG")
            print("row/col:", test_row, test_col)
            print("display_mm:", test_mm)
            print("inches:", test_mm / 25.4)
            print("expected_rgba:", test_rgba)

        with open(tif_path, "rb") as f:
            tif_bytes = f.read()

        png_url_raw = upload_bytes(s3, png_key, png_bytes, "image/png")
        tif_url_raw = upload_bytes(s3, tif_key, tif_bytes, "image/tiff")

        png_url = version_url(png_url_raw)
        tif_url = version_url(tif_url_raw)

        max_mm = float(np.nanmax(display_mm))
        max_inches = max_mm / 25.4

        record = {
            "duration_hours": duration,
            "start_valid_time_utc": selected_records[0]["valid_time_utc"],
            "end_valid_time_utc": selected_records[-1]["valid_time_utc"],
            "image_url": png_url,
            "geotiff_url": tif_url,
            "bounds": display_bounds,
            "units": "mm",
            "max_rainfall_mm": round(max_mm, 3),
            "max_rainfall_inches": round(max_inches, 3),
            "version": RUN_VERSION,
        }

        accum_records.append(record)

        print(f"Accum {tag} PNG → {png_url}")
        print(f"Accum {tag} TIF → {tif_url}")

    return accum_records


def main():
    print("RUN_VERSION:", RUN_VERSION)

    s3 = make_s3()
    cycle_dt = choose_latest_cycle()

    with tempfile.TemporaryDirectory() as tmpdir:
        (
            forecast_records,
            hourly_arrays,
            bounds,
            geotransform,
            projection,
        ) = process_hrrr_forecast(s3, cycle_dt, tmpdir)

        accum_records = build_accumulations(
            s3=s3,
            hourly_arrays=hourly_arrays,
            forecast_records=forecast_records,
            bounds=bounds,
            geotransform=geotransform,
            projection=projection,
            tmpdir=tmpdir,
        )

    index_payload = {
        "cycle_time_utc": cycle_dt.isoformat(),
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "version": RUN_VERSION,
        "source": "NOAA HRRR AWS Open Data",
        "forecast_hours": forecast_records,
        "accumulations": accum_records,
    }

    index_url_raw = upload_bytes(
        s3,
        "hrrr/forecast/index.json",
        json.dumps(index_payload, indent=2).encode(),
        "application/json",
    )

    index_url = version_url(index_url_raw)

    print("\n=== HRRR forecast pipeline complete ===")
    print(f"Index URL: {index_url}")
    print(f"Forecast hours processed: {len(forecast_records)}")
    print(f"Accumulations processed: {len(accum_records)}")


if __name__ == "__main__":
    main()
