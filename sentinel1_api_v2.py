"""Robust Sentinel-1 daily AOI downloader.

Designed for long multi-year pulls and easy reuse: change bbox + output_dir only.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator
import math
import re
import time

from affine import Affine
import numpy as np
import pandas as pd
import planetary_computer
from PIL import Image
from pystac import Item
from pystac_client import Client
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import transform_bounds, transform_geom
from rasterio.windows import Window, from_bounds
from shapely.geometry import box, shape


PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


@dataclass(frozen=True)
class CollectConfig:
    bbox: tuple[float, float, float, float]
    start_date: str
    end_date: str
    geometry: dict | None = None

    output_dir: str = "S1_Images"
    metadata_csv: str = "S1_Images/metadata.csv"

    collections: tuple[str, ...] = ("sentinel-1-grd",)
    one_best_per_day: bool = True
    min_overlap_fraction: float = 0.2

    preferred_asset: str = "vv"
    asset_allowlist: tuple[str, ...] = ("vv", "vh", "VV", "VH", "visual", "preview")

    save_geotiff: bool = False
    save_jpg: bool = True

    jpg_max_side_px: int | None = 1400
    jpg_stretch: str = "percentile"  # "percentile" or "fixed"
    jpg_db_min: float = -25.0
    jpg_db_max: float = 5.0
    percentile_low: float = 2.0
    percentile_high: float = 98.0

    bbox_padding_deg: float = 0.0
    max_workers: int = 4
    overwrite: bool = False
    limit_days: int | None = None
    chunk_by: str = "year"  # "year" or "month"

    on_missing_crs: str = "full"  # "full" or "skip"
    max_retries: int = 3
    retry_backoff: float = 1.5
    min_valid_fraction: float = 0.02

    # Optional STAC property filters
    orbit_direction: str | None = None   # "ascending" / "descending"
    instrument_mode: str | None = None   # e.g. "IW"
    polarizations: tuple[str, ...] | None = None
    normalize_jpg_direction: bool = False
    reference_orbit_direction: str = "ascending"   # "ascending" / "descending"
    flip_axis: str = "horizontal"                  # "horizontal" / "vertical"
    mask_outside_geometry: bool = False


def collect_daily(config: CollectConfig) -> pd.DataFrame:
    _validate_bbox(config.bbox)
    _validate_geometry(config.geometry, config.bbox)
    _validate_fraction(config.min_overlap_fraction, "min_overlap_fraction")
    _validate_fraction(config.min_valid_fraction, "min_valid_fraction")
    _validate_choice(config.jpg_stretch, {"percentile", "fixed"}, "jpg_stretch")
    _validate_choice(config.on_missing_crs, {"full", "skip"}, "on_missing_crs")
    _validate_choice(config.chunk_by, {"year", "month"}, "chunk_by")
    _validate_choice(config.reference_orbit_direction, {"ascending", "descending"}, "reference_orbit_direction")
    _validate_choice(config.flip_axis, {"horizontal", "vertical"}, "flip_axis")
    if config.max_workers < 1:
        raise ValueError("max_workers must be >= 1")

    padded_bbox = _expand_bbox(config.bbox, config.bbox_padding_deg)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    for start_dt, end_dt in _iter_date_ranges(config.start_date, config.end_date, config.chunk_by):
        items = _search_items(
            bbox=padded_bbox,
            start_date=start_dt,
            end_date=end_dt,
            collections=config.collections,
            orbit_direction=config.orbit_direction,
            instrument_mode=config.instrument_mode,
            polarizations=config.polarizations,
        )
        if not items:
            continue

        selected = _select_items(
            items,
            padded_bbox,
            config.geometry,
            one_best_per_day=config.one_best_per_day,
            min_overlap_fraction=config.min_overlap_fraction,
        )
        if config.limit_days is not None:
            selected = selected.head(config.limit_days)
        if selected.empty:
            continue

        work = [(row["item"], float(row["overlap_fraction"])) for _, row in selected.iterrows()]
        worker_count = min(max(1, int(config.max_workers)), len(work))
        records: list[dict] = []

        if worker_count == 1:
            for item, overlap in work:
                records.append(
                    _download_item(
                        item=item,
                        bbox=padded_bbox,
                        geometry=config.geometry,
                        output_dir=out_dir,
                        preferred_asset=config.preferred_asset,
                        asset_allowlist=config.asset_allowlist,
                        save_geotiff=config.save_geotiff,
                        save_jpg=config.save_jpg,
                        jpg_db_min=config.jpg_db_min,
                        jpg_db_max=config.jpg_db_max,
                        jpg_max_side_px=config.jpg_max_side_px,
                        jpg_stretch=config.jpg_stretch,
                        percentile_low=config.percentile_low,
                        percentile_high=config.percentile_high,
                        overwrite=config.overwrite,
                        overlap_fraction=overlap,
                        on_missing_crs=config.on_missing_crs,
                        max_retries=config.max_retries,
                        retry_backoff=config.retry_backoff,
                        min_valid_fraction=config.min_valid_fraction,
                        normalize_jpg_direction=config.normalize_jpg_direction,
                        reference_orbit_direction=config.reference_orbit_direction,
                        flip_axis=config.flip_axis,
                        mask_outside_geometry=config.mask_outside_geometry,
                    )
                )
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(
                        _download_item,
                        item=item,
                        bbox=padded_bbox,
                        geometry=config.geometry,
                        output_dir=out_dir,
                        preferred_asset=config.preferred_asset,
                        asset_allowlist=config.asset_allowlist,
                        save_geotiff=config.save_geotiff,
                        save_jpg=config.save_jpg,
                        jpg_db_min=config.jpg_db_min,
                        jpg_db_max=config.jpg_db_max,
                        jpg_max_side_px=config.jpg_max_side_px,
                        jpg_stretch=config.jpg_stretch,
                        percentile_low=config.percentile_low,
                        percentile_high=config.percentile_high,
                        overwrite=config.overwrite,
                        overlap_fraction=overlap,
                        on_missing_crs=config.on_missing_crs,
                        max_retries=config.max_retries,
                        retry_backoff=config.retry_backoff,
                        min_valid_fraction=config.min_valid_fraction,
                        normalize_jpg_direction=config.normalize_jpg_direction,
                        reference_orbit_direction=config.reference_orbit_direction,
                        flip_axis=config.flip_axis,
                        mask_outside_geometry=config.mask_outside_geometry,
                    )
                    for item, overlap in work
                ]
                for fut in as_completed(futures):
                    records.append(fut.result())

        if records:
            _append_csv(records, Path(config.metadata_csv))
            all_records.extend(records)

    if not all_records:
        return _empty_result_df()
    df = pd.DataFrame.from_records(all_records)
    df.sort_values(["date", "datetime", "item_id"], inplace=True, ignore_index=True)
    return df


def preview_search(config: CollectConfig, sample_size: int = 20) -> dict[str, object]:
    """Inspect what the API can see before downloading anything."""
    _validate_bbox(config.bbox)
    _validate_geometry(config.geometry, config.bbox)
    _validate_fraction(config.min_overlap_fraction, "min_overlap_fraction")
    padded_bbox = _expand_bbox(config.bbox, config.bbox_padding_deg)
    aoi_geom = _aoi_geometry(padded_bbox, config.geometry)

    raw_rows: list[dict] = []
    for start_dt, end_dt in _iter_date_ranges(config.start_date, config.end_date, config.chunk_by):
        items = _search_items(
            bbox=padded_bbox,
            start_date=start_dt,
            end_date=end_dt,
            collections=config.collections,
            orbit_direction=config.orbit_direction,
            instrument_mode=config.instrument_mode,
            polarizations=config.polarizations,
        )
        for item in items:
            dt = item.datetime
            if dt is None or item.geometry is None:
                continue
            geom = shape(item.geometry)
            if geom.is_empty:
                continue
            overlap_fraction = _overlap_fraction(aoi_geom, geom)
            asset_keys = ",".join(item.assets.keys())
            raw_rows.append(
                {
                    "date": dt.date().isoformat(),
                    "datetime": dt.isoformat(),
                    "item_id": item.id,
                    "collection": item.collection_id,
                    "overlap_fraction": overlap_fraction,
                    "orbit_direction": _item_orbit_direction(item),
                    "asset_keys": asset_keys,
                }
            )

    raw_df = pd.DataFrame(raw_rows)
    if raw_df.empty:
        return {
            "items_found": 0,
            "items_after_overlap": 0,
            "days_found": 0,
            "days_after_overlap": 0,
            "sample": pd.DataFrame(),
        }

    overlap_df = raw_df[raw_df["overlap_fraction"] >= config.min_overlap_fraction].copy()
    overlap_df.sort_values(
        ["date", "overlap_fraction", "datetime", "item_id"],
        ascending=[True, False, True, True],
        inplace=True,
    )

    return {
        "items_found": int(len(raw_df)),
        "items_after_overlap": int(len(overlap_df)),
        "days_found": int(raw_df["date"].nunique()),
        "days_after_overlap": int(overlap_df["date"].nunique()),
        "sample": overlap_df.head(sample_size).reset_index(drop=True),
    }


def _append_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return
    write_header = not path.exists()
    df.to_csv(path, index=False, mode="a", header=write_header)


def _empty_result_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "datetime",
            "item_id",
            "collection",
            "asset",
            "overlap_fraction",
            "status",
            "jpg_path",
            "tif_path",
            "reason",
            "orbit_direction",
            "flip_applied",
        ]
    )


def _search_items(
    bbox: tuple[float, float, float, float],
    start_date: date,
    end_date: date,
    collections: Iterable[str],
    orbit_direction: str | None,
    instrument_mode: str | None,
    polarizations: tuple[str, ...] | None,
) -> list[Item]:
    client = Client.open(PC_STAC_URL, modifier=planetary_computer.sign_inplace)
    timerange = f"{start_date.isoformat()}T00:00:00Z/{end_date.isoformat()}T23:59:59Z"

    query: dict = {}
    if orbit_direction:
        query["sat:orbit_state"] = {"eq": orbit_direction.lower()}
    if instrument_mode:
        query["sar:instrument_mode"] = {"eq": instrument_mode}
    if polarizations:
        query["sar:polarizations"] = {"in": list(polarizations)}

    search = client.search(
        collections=list(collections),
        bbox=list(bbox),
        datetime=timerange,
        query=query if query else None,
    )
    return list(search.items())


def _select_items(
    items: list[Item],
    bbox: tuple[float, float, float, float],
    geometry: dict | None,
    one_best_per_day: bool,
    min_overlap_fraction: float,
) -> pd.DataFrame:
    aoi = _aoi_geometry(bbox, geometry)
    rows: list[dict] = []
    for item in items:
        dt = item.datetime
        if dt is None or item.geometry is None:
            continue
        geom = shape(item.geometry)
        if geom.is_empty:
            continue
        overlap_fraction = _overlap_fraction(aoi, geom)
        rows.append(
            {
                "date": dt.date().isoformat(),
                "datetime": dt.isoformat(),
                "item_id": item.id,
                "collection": item.collection_id,
                "overlap_fraction": overlap_fraction,
                "item": item,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["date", "datetime", "item_id", "collection", "overlap_fraction", "item"]
        )

    df = pd.DataFrame(rows)
    df = df[df["overlap_fraction"] >= min_overlap_fraction].copy()
    if df.empty:
        return pd.DataFrame(
            columns=["date", "datetime", "item_id", "collection", "overlap_fraction", "item"]
        )

    df.sort_values(
        by=["date", "overlap_fraction", "datetime", "item_id"],
        ascending=[True, False, True, True],
        inplace=True,
    )
    if one_best_per_day:
        df = df.drop_duplicates(subset=["date"], keep="first")
    return df.reset_index(drop=True)


def _download_item(
    item: Item,
    bbox: tuple[float, float, float, float],
    geometry: dict | None,
    output_dir: Path,
    preferred_asset: str,
    asset_allowlist: tuple[str, ...],
    save_geotiff: bool,
    save_jpg: bool,
    jpg_db_min: float,
    jpg_db_max: float,
    jpg_max_side_px: int | None,
    jpg_stretch: str,
    percentile_low: float,
    percentile_high: float,
    overwrite: bool,
    overlap_fraction: float,
    on_missing_crs: str,
    max_retries: int,
    retry_backoff: float,
    min_valid_fraction: float,
    normalize_jpg_direction: bool,
    reference_orbit_direction: str,
    flip_axis: str,
    mask_outside_geometry: bool,
) -> dict:
    dt = item.datetime
    if dt is None:
        return _failed_record(item, overlap_fraction, "missing_datetime")
    orbit_direction = _item_orbit_direction(item)

    signed_item = planetary_computer.sign(item)
    asset_key = _pick_asset_key(signed_item, preferred_asset, asset_allowlist)
    if asset_key is None:
        return _failed_record(item, overlap_fraction, "no_supported_asset", orbit_direction=orbit_direction)

    asset_href = signed_item.assets[asset_key].href
    timestamp = dt.strftime("%Y%m%dT%H%M%SZ")
    stem = _safe_name(f"{dt.date().isoformat()}_{item.collection_id}_{asset_key}_{timestamp}_{item.id}")
    tif_path = output_dir / f"{stem}.tif"
    jpg_path = output_dir / f"{stem}.jpg"

    if not overwrite and _already_done(save_geotiff, save_jpg, tif_path, jpg_path):
        return {
            "date": dt.date().isoformat(),
            "datetime": dt.isoformat(),
            "item_id": item.id,
            "collection": item.collection_id,
            "asset": asset_key,
            "overlap_fraction": overlap_fraction,
            "status": "skipped_exists",
            "jpg_path": str(jpg_path) if save_jpg else "",
            "tif_path": str(tif_path) if save_geotiff else "",
            "reason": "",
            "orbit_direction": orbit_direction,
            "flip_applied": False,
        }

    for attempt in range(max_retries + 1):
        try:
            with rasterio.open(asset_href) as src:
                window = _window_from_wgs84_bbox(src, bbox, on_missing_crs)
                if window is None:
                    return _failed_record(
                        item, overlap_fraction, "missing_crs_skip", asset_key, orbit_direction
                    )

                # Downsample read when only JPG is needed.
                read_kwargs = {}
                if not save_geotiff and jpg_max_side_px:
                    out_h, out_w = _scaled_shape(src, window, jpg_max_side_px)
                    if out_h and out_w:
                        read_kwargs["out_shape"] = (1, out_h, out_w)
                        read_kwargs["resampling"] = Resampling.bilinear

                data = src.read(1, window=window, boundless=True, **read_kwargs).astype(np.float32)
                transform = src.window_transform(window)
                if "out_shape" in read_kwargs:
                    out_h = read_kwargs["out_shape"][1]
                    out_w = read_kwargs["out_shape"][2]
                    scale_x = window.width / out_w
                    scale_y = window.height / out_h
                    transform = transform * Affine.scale(scale_x, scale_y)
                out_crs = src.crs
                nodata = src.nodata

            if mask_outside_geometry and geometry is not None:
                if out_crs is None:
                    return _failed_record(
                        item,
                        overlap_fraction,
                        "geometry_mask_requires_crs",
                        asset_key,
                        orbit_direction,
                    )
                data = _mask_data_to_geometry(data, transform, out_crs, geometry)

            if nodata is not None:
                data = np.where(data == nodata, np.nan, data)

            finite = np.isfinite(data)
            valid_fraction = float(finite.sum() / max(1, data.size))
            if valid_fraction < min_valid_fraction:
                return _failed_record(item, overlap_fraction, "low_valid_fraction", asset_key, orbit_direction)

            if save_geotiff:
                if out_crs is None:
                    # Skip GeoTIFF if CRS missing to avoid invalid files.
                    pass
                else:
                    _write_tif(tif_path, data, transform, out_crs)
            if save_jpg:
                vis = _to_display_jpg_array(
                    data,
                    stretch=jpg_stretch,
                    db_min=jpg_db_min,
                    db_max=jpg_db_max,
                    p_low=percentile_low,
                    p_high=percentile_high,
                )
                flip_applied = False
                if normalize_jpg_direction:
                    vis, flip_applied = _normalize_jpg_direction(
                        vis, orbit_direction, reference_orbit_direction, flip_axis
                    )
                vis = _resize_max_side(vis, jpg_max_side_px)
                Image.fromarray(vis, mode="L").save(jpg_path, quality=95)
            else:
                flip_applied = False

            return {
                "date": dt.date().isoformat(),
                "datetime": dt.isoformat(),
                "item_id": item.id,
                "collection": item.collection_id,
                "asset": asset_key,
                "overlap_fraction": overlap_fraction,
                "status": "downloaded",
                "jpg_path": str(jpg_path) if save_jpg else "",
                "tif_path": str(tif_path) if save_geotiff and out_crs else "",
                "reason": "",
                "orbit_direction": orbit_direction,
                "flip_applied": flip_applied,
            }
        except Exception as exc:  # pragma: no cover - runtime/network dependent
            if attempt >= max_retries:
                return _failed_record(
                    item, overlap_fraction, f"read_error:{exc}", asset_key, orbit_direction
                )
            time.sleep(retry_backoff ** attempt)

    return _failed_record(item, overlap_fraction, "unknown_error", asset_key, orbit_direction)


def _scaled_shape(src: rasterio.DatasetReader, window: Window, max_side_px: int) -> tuple[int | None, int | None]:
    if max_side_px <= 0:
        return None, None
    width = int(window.width)
    height = int(window.height)
    if width <= 0 or height <= 0:
        return None, None
    current_max = max(width, height)
    if current_max <= max_side_px:
        return None, None
    scale = max_side_px / float(current_max)
    out_w = max(1, int(round(width * scale)))
    out_h = max(1, int(round(height * scale)))
    return out_h, out_w


def _pick_asset_key(item: Item, preferred_asset: str, allowlist: tuple[str, ...]) -> str | None:
    keys = list(item.assets.keys())
    if preferred_asset in item.assets:
        return preferred_asset
    for key in allowlist:
        if key in item.assets:
            return key
    return keys[0] if keys else None


def _item_orbit_direction(item: Item) -> str:
    value = item.properties.get("sat:orbit_state", "")
    if isinstance(value, str):
        return value.lower()
    return ""


def _window_from_wgs84_bbox(
    src: rasterio.DatasetReader,
    bbox: tuple[float, float, float, float],
    on_missing_crs: str,
) -> Window | None:
    if src.crs is None:
        if on_missing_crs == "skip":
            return None
        return Window(col_off=0, row_off=0, width=src.width, height=src.height)
    left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
    window = from_bounds(left, bottom, right, top, src.transform)
    return window.round_offsets().round_lengths()


def _mask_data_to_geometry(
    data: np.ndarray,
    transform,
    crs,
    geometry: dict,
) -> np.ndarray:
    projected_geom = transform_geom("EPSG:4326", crs, geometry)
    inside_mask = geometry_mask(
        [projected_geom],
        out_shape=data.shape,
        transform=transform,
        invert=True,
    )
    return np.where(inside_mask, data, np.nan)


def _overlap_fraction(aoi, geom) -> float:
    if aoi.is_empty:
        return 0.0
    inter = aoi.intersection(geom)
    if inter.is_empty:
        return 0.0
    return float(inter.area / aoi.area)


def _aoi_geometry(bbox: tuple[float, float, float, float], geometry: dict | None):
    if geometry is None:
        return box(*bbox)
    return shape(geometry)


def _to_display_jpg_array(
    data: np.ndarray,
    stretch: str,
    db_min: float,
    db_max: float,
    p_low: float,
    p_high: float,
) -> np.ndarray:
    arr = data.copy()
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)

    arr_finite = arr[finite]
    p5 = float(np.percentile(arr_finite, 5))
    p95 = float(np.percentile(arr_finite, 95))
    if p5 >= 0.0 and p95 <= 2.0:
        arr = 10.0 * np.log10(np.clip(arr, 1e-8, None))

    if stretch == "fixed":
        lo, hi = db_min, db_max
    else:
        lo = float(np.percentile(arr_finite, p_low))
        hi = float(np.percentile(arr_finite, p_high))
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            lo, hi = db_min, db_max

    arr = np.clip(arr, lo, hi)
    scaled = ((arr - lo) / (hi - lo)) * 255.0
    scaled[~finite] = 0.0
    return scaled.astype(np.uint8)


def _write_tif(path: Path, data: np.ndarray, transform, crs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nodata = -9999.0
    out = np.where(np.isfinite(data), data, nodata).astype(np.float32)
    with rasterio.open(
        path,
        mode="w",
        driver="GTiff",
        height=out.shape[0],
        width=out.shape[1],
        count=1,
        dtype=np.float32,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="deflate",
    ) as dst:
        dst.write(out, 1)


def _validate_bbox(bbox: tuple[float, float, float, float]) -> None:
    if len(bbox) != 4:
        raise ValueError("bbox must be (min_lon, min_lat, max_lon, max_lat)")
    min_lon, min_lat, max_lon, max_lat = bbox
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("bbox must satisfy min_lon < max_lon and min_lat < max_lat")
    if min_lon < -180 or max_lon > 180 or min_lat < -90 or max_lat > 90:
        raise ValueError("bbox coordinates are out of WGS84 bounds")


def _validate_geometry(geometry: dict | None, bbox: tuple[float, float, float, float]) -> None:
    if geometry is None:
        return
    geom = shape(geometry)
    if geom.is_empty:
        raise ValueError("geometry must not be empty")
    min_lon, min_lat, max_lon, max_lat = geom.bounds
    if min_lon < bbox[0] or min_lat < bbox[1] or max_lon > bbox[2] or max_lat > bbox[3]:
        raise ValueError("geometry must be inside bbox")


def _validate_fraction(value: float, name: str) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def _validate_choice(value: str, allowed: set[str], name: str) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}")


def _expand_bbox(
    bbox: tuple[float, float, float, float],
    padding_deg: float,
) -> tuple[float, float, float, float]:
    if padding_deg <= 0:
        return bbox
    min_lon, min_lat, max_lon, max_lat = bbox
    return (
        max(-180.0, min_lon - padding_deg),
        max(-90.0, min_lat - padding_deg),
        min(180.0, max_lon + padding_deg),
        min(90.0, max_lat + padding_deg),
    )


def _resize_max_side(arr: np.ndarray, max_side_px: int | None) -> np.ndarray:
    if max_side_px is None or max_side_px <= 0:
        return arr
    height, width = arr.shape[:2]
    current_max = max(height, width)
    if current_max <= max_side_px:
        return arr
    scale = max_side_px / float(current_max)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    return np.array(Image.fromarray(arr).resize((new_w, new_h), resample=resample))


def _normalize_jpg_direction(
    arr: np.ndarray,
    orbit_direction: str,
    reference_orbit_direction: str,
    flip_axis: str,
) -> tuple[np.ndarray, bool]:
    if orbit_direction not in {"ascending", "descending"}:
        return arr, False
    if orbit_direction == reference_orbit_direction:
        return arr, False
    if flip_axis == "horizontal":
        return np.fliplr(arr), True
    return np.flipud(arr), True


def _already_done(save_geotiff: bool, save_jpg: bool, tif_path: Path, jpg_path: Path) -> bool:
    tif_ok = (not save_geotiff) or tif_path.exists()
    jpg_ok = (not save_jpg) or jpg_path.exists()
    return tif_ok and jpg_ok


def _failed_record(
    item: Item,
    overlap_fraction: float,
    reason: str,
    asset_key: str = "",
    orbit_direction: str = "",
) -> dict:
    dt = item.datetime.isoformat() if item.datetime is not None else ""
    date_val = item.datetime.date().isoformat() if item.datetime is not None else ""
    return {
        "date": date_val,
        "datetime": dt,
        "item_id": item.id,
        "collection": item.collection_id,
        "asset": asset_key,
        "overlap_fraction": overlap_fraction,
        "status": "failed",
        "jpg_path": "",
        "tif_path": "",
        "reason": reason,
        "orbit_direction": orbit_direction,
        "flip_applied": False,
    }


def _safe_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return clean[:180]


def _iter_date_ranges(start: str, end: str, chunk_by: str) -> Iterator[tuple[date, date]]:
    start_dt = datetime.fromisoformat(start).date()
    end_dt = datetime.fromisoformat(end).date()
    if start_dt > end_dt:
        raise ValueError("start_date must be <= end_date")

    if chunk_by == "year":
        year = start_dt.year
        while date(year, 1, 1) <= end_dt:
            chunk_start = max(start_dt, date(year, 1, 1))
            chunk_end = min(end_dt, date(year, 12, 31))
            yield chunk_start, chunk_end
            year += 1
    else:
        current = date(start_dt.year, start_dt.month, 1)
        while current <= end_dt:
            next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
            chunk_start = max(start_dt, current)
            chunk_end = min(end_dt, next_month - timedelta(days=1))
            yield chunk_start, chunk_end
            current = next_month
