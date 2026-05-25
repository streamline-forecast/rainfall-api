#!/usr/bin/env python3

import os
import io
import json
import tempfile
import datetime
import urllib.request
import subprocess

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
    (0.0, (0, 0, 0, 0)),
    (0.254, (186, 224, 255, 160)),
    (6.35, (120, 170, 255, 175)),
    (12.7, (70, 95, 220, 190)),
    (25.4, (125, 70, 210, 205)),
    (50.8, (205, 70, 190, 220)),
    (101.6, (210, 60, 85, 235)),
    (203.2, (100, 0, 0, 255)),
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
        _, idx_url = hrrr_urls(dt.date(), dt.hour, 1)

        print(f"Checking HRRR cycle {dt.isoformat()} → {idx_url}")

        if url_exists(idx_url):
            print(f"Selected HRRR cycle: {dt.isoformat()}")
            return dt

    raise RuntimeError("No available HRRR cycle found in the last 8 hours.")


def fetch_text(url, max_attempts=5):
    import time

    for attempt in range(1, max_attempts + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 GitHubActions-HRRR-Pipeline/1.0",
                    "Accept": "text/plain,*/*",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read().decode("utf-8", errors="ignore")
        except Exception as e:
            print(f"WARNING: fetch_text failed attempt {attempt}/{max_attempts}: {e}")
            if attempt == max_attempts:
                raise
            time.sleep(10 * attempt)


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

        records.append({
            "record_number": record_number,
            "byte_start": byte_start,
            "variable": parts[3] if len(parts) > 3 else "",
            "level": parts[4] if len(parts) > 4 else "",
            "timing": ":".join(parts[5:]) if len(parts) > 5 else "",
            "line": line,
        })

    for i, rec in enumerate(records):
        rec["byte_end"] = records[i + 1]["byte_start"] - 1 if i + 1 < len(records) else None

    apcp_records = [
        r for r in records
        if r["variable"] == "APCP" and "surface" in r["level"].lower()
    ]

    if not apcp_records:
        raise RuntimeError("No APCP surface record found in HRRR idx file.")

    print("APCP RECORDS FOUND:")
    for r in apcp_records:
        print(r["line"])

    return apcp_records[-1]

def download_byte_range(url, byte_start, byte_end, output_path, max_attempts=5):
    import time

    for attempt in range(1, max_attempts + 1):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 GitHubActions-HRRR-Pipeline/1.0",
                "Accept": "application/octet-stream,*/*",
                "Range": f"bytes={byte_start}-{byte_end}" if byte_end is not None else f"bytes={byte_start}-",
            }

            req = urllib.request.Request(url, headers=headers)

            with urllib.request.urlopen(req, timeout=180) as resp:
                data = resp.read()

            with open(output_path, "wb") as f:
                f.write(data)

            print(f"Downloaded byte range to {output_path}: {len(data):,} bytes")
            return output_path

        except Exception as e:
            print(f"WARNING: download_byte_range failed attempt {attempt}/{max_attempts}: {e}")
            if attempt == max_attempts:
                raise
            time.sleep(10 * attempt)


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
    band.SetNoDataValue(-9999.0)
    band.FlushCache()

    ds.FlushCache()
    ds = None

    return output_path
    
def write_array_to_geotiff_from_bounds(array_mm, output_path, bounds):
    from osgeo import gdal, osr

    rows, cols = array_mm.shape
    south, west = bounds[0]
    north, east = bounds[1]

    xres = (east - west) / cols
    yres = (north - south) / rows

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(
        output_path,
        cols,
        rows,
        1,
        gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES"],
    )

    ds.SetGeoTransform([west, xres, 0, north, 0, -yres])

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ds.SetProjection(srs.ExportToWkt(["FORMAT=WKT1_GDAL"]))

    band = ds.GetRasterBand(1)
    band.WriteArray(array_mm.astype(np.float32))
    band.SetNoDataValue(-9999.0)
    band.FlushCache()

    ds.FlushCache()
    ds = None
def warp_to_epsg4326(src_path, dst_path):
    from osgeo import gdal

    options = gdal.WarpOptions(
        format="GTiff",
        dstSRS="EPSG:4326",
        resampleAlg="near",
        errorThreshold=0.0,
        multithread=True,
        creationOptions=["COMPRESS=LZW", "TILED=YES"],
    )

    ds = gdal.Warp(dst_path, src_path, options=options)

    if ds is None:
        raise RuntimeError(f"Failed to warp raster to EPSG:4326: {src_path}")

    ds.FlushCache()
    ds = None
    return dst_path

def warp_to_web_mercator(src_path, dst_path):
    from osgeo import gdal

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
        raise RuntimeError(f"Failed to warp raster to EPSG:3857: {src_path}")

    ds.FlushCache()
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
            cumulative_tif_path = os.path.join(tmpdir, f"hrrr_f{fhour:02d}_cumulative_native.tif")
            native_hourly_tif_path = os.path.join(tmpdir, f"hrrr_f{fhour:02d}_hourly_native.tif")
            display_hourly_tif_path = os.path.join(tmpdir, f"hrrr_f{fhour:02d}.tif")

            print(f"APCP idx line: {apcp['line']}")

            download_byte_range(
                grib_url,
                apcp["byte_start"],
                apcp["byte_end"],
                apcp_grib_path,
            )

            grib2_to_geotiff(apcp_grib_path, cumulative_tif_path)
            if fhour == 6:
                with open(cumulative_tif_path, "rb") as f:
                    native_cumulative_bytes = f.read()

                native_key = (
                    f"hrrr/debug/native/"
                    f"hrrr_f{fhour:02d}_cumulative_native_{RUN_VERSION}.tif"
                )

                native_url = upload_bytes(
                    s3,
                    native_key,
                    native_cumulative_bytes,
                    "image/tiff",
                )

                print(f"NATIVE CUMULATIVE DEBUG TIF → {native_url}")

            cumulative_mm, native_bounds, native_gt, native_projection = geotiff_to_array(cumulative_tif_path)

            if previous_cumulative is None:
                hourly_native_mm = cumulative_mm
            else:
                hourly_native_mm = cumulative_mm - previous_cumulative
                hourly_native_mm[hourly_native_mm < 0] = 0.0
                hourly_native_mm[np.isnan(hourly_native_mm)] = 0.0

            previous_cumulative = cumulative_mm.copy()

            write_array_to_geotiff(
                hourly_native_mm,
                native_hourly_tif_path,
                native_gt,
                native_projection,
            )

            extractor_hourly_tif_path = os.path.join(
                tmpdir,
                f"hrrr_f{fhour:02d}_extractor_4326.tif"
            )

            corrected_hourly_tif_path = os.path.join(
                tmpdir,
                f"hrrr_f{fhour:02d}_corrected_extract.tif"
            )

            png_display_tif_path = os.path.join(
                tmpdir,
                f"hrrr_f{fhour:02d}_png3857.tif"
            )

            print(f"Warping HRRR F{fhour:02d} hourly to EPSG:4326")
            warp_to_epsg4326(native_hourly_tif_path, extractor_hourly_tif_path)

            display_hourly_mm, display_bounds, display_gt, display_projection = geotiff_to_array(
                extractor_hourly_tif_path
            )

            write_array_to_geotiff_from_bounds(
                display_hourly_mm,
                corrected_hourly_tif_path,
                display_bounds,
            )

            corrected_mm, corrected_bounds, corrected_gt, corrected_projection = geotiff_to_array(
                corrected_hourly_tif_path
            )

            print("HRRR hourly corrected extractor bounds:")
            print(corrected_bounds)
            print("HRRR hourly corrected min/max mm:")
            print(float(np.nanmin(corrected_mm)), float(np.nanmax(corrected_mm)))

            print(f"Warping HRRR F{fhour:02d} corrected hourly to EPSG:3857 for PNG")
            # Build corrected extractor raster
            write_array_to_geotiff_from_bounds(
                display_hourly_mm,
                corrected_hourly_tif_path,
                display_bounds,
            )

            corrected_mm, corrected_bounds, corrected_gt, corrected_projection = geotiff_to_array(
                corrected_hourly_tif_path
            )

            # Build display PNG from WebMerc version of corrected raster
            warp_to_web_mercator(
                corrected_hourly_tif_path,
                png_display_tif_path
            )

            png_mm, png_bounds, png_gt, png_projection = geotiff_to_array(
                png_display_tif_path
            )

            png_bytes = array_to_png_bytes(png_mm)

            # Upload extractor GeoTIFF
            with open(corrected_hourly_tif_path, "rb") as f:
                tif_bytes = f.read()

            # Upload GRIB debug file
            with open(apcp_grib_path, "rb") as f:
                grib_bytes = f.read()


            png_key = f"hrrr/forecast/png/hrrr_f{fhour:02d}_{RUN_VERSION}.png"
            tif_key = f"hrrr/forecast/geotiff/hrrr_f{fhour:02d}_{RUN_VERSION}.tif"
            grib_key = f"hrrr/forecast/grib2/hrrr_f{fhour:02d}_apcp_{RUN_VERSION}.grib2"

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
                "bounds": png_bounds,              # Leaflet display PNG bounds
                "geotiff_bounds": corrected_bounds, # extractor GeoTIFF bounds
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
        png_key = f"hrrr/forecast/accum/png/accum_{tag}_{RUN_VERSION}.png"
        tif_key = f"hrrr/forecast/accum/geotiff/accum_{tag}_{RUN_VERSION}.tif"

        print_accum_debug(tag, accum_mm)

        tif_path = os.path.join(tmpdir, f"hrrr_accum_{tag}.tif")

        write_array_to_geotiff(
            accum_mm,
            tif_path,
            geotransform,
            projection,
        )

        display_mm, display_bounds, display_gt, display_projection = geotiff_to_array(tif_path)

        print("DISPLAY accumulation bounds:")
        print(display_bounds)

        # Build PNG from Web Mercator version for Leaflet display
        webmerc_tif_path = os.path.join(tmpdir, f"hrrr_accum_{tag}_3857.tif")
        warp_to_web_mercator(tif_path, webmerc_tif_path)

        webmerc_mm, webmerc_bounds, webmerc_gt, webmerc_projection = geotiff_to_array(webmerc_tif_path)

        print("WEB MERCATOR accumulation bounds:")
        print(webmerc_bounds)

        png_bytes = array_to_png_bytes(webmerc_mm)

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
        json.dumps(index_payload, indent=2).encode("utf-8"),
        "application/json",
    )

    index_url = version_url(index_url_raw)

    print("\n=== HRRR forecast pipeline complete ===")
    print(f"Index URL: {index_url}")
    print(f"Forecast hours processed: {len(forecast_records)}")
    print(f"Accumulations processed: {len(accum_records)}")


if __name__ == "__main__":
    main()
