"""
File: earth_engine.py
Description: GEE wrapper (FR-D01). All GEE calls through this tool.
             Safety check + feature extraction, no raw rasters to LLM.
Author: CALM Team
Created: 2026-03-13
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class EarthEngineTool:
    """
    Google Earth Engine tool.

    Bản này hỗ trợ:
    - point / bbox / polygon
    - Sentinel-2, MODIS LST, DEM/SRTM, land cover, Landsat
    - output feature summary usable cho model builder và RSEN

    Lưu ý:
    - Tool này chỉ lấy/summarize dữ liệu không gian từ GEE
    - Teleconnection / population nên lấy từ nguồn khác
    """

    DEFAULT_PRODUCTS = [
        "sentinel2",
        "modis_lst",
        "srtm",
        "land_cover",
    ]

    LAND_COVER_LABELS = {
        1: "Evergreen Needleleaf Forest",
        2: "Evergreen Broadleaf Forest",
        3: "Deciduous Needleleaf Forest",
        4: "Deciduous Broadleaf Forest",
        5: "Mixed Forests",
        6: "Closed Shrublands",
        7: "Open Shrublands",
        8: "Woody Savannas",
        9: "Savannas",
        10: "Grasslands",
        11: "Permanent Wetlands",
        12: "Croplands",
        13: "Urban and Built-up",
        14: "Cropland/Natural Vegetation Mosaic",
        15: "Snow and Ice",
        16: "Barren or Sparsely Vegetated",
        17: "Water Bodies",
    }

    def __init__(
        self,
        safety_checker,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.safety_checker = safety_checker
        self.config = config or {}
        self._initialized = False
        self._cache: Dict[str, Dict[str, Any]] = {}

        self.default_buffer_meters = int(self.config.get("default_buffer_meters", 5000))
        self.default_scale = int(self.config.get("default_scale", 250))
        self.max_pixels = int(self.config.get("max_pixels", 5_000_000))
        self.gee_project = self.config.get("gee_project")

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def fetch_satellite_stats(
        self,
        location: Any = None,
        time_range: Optional[Dict[str, Any]] = None,
        product: str = "sentinel2",
        bbox: Optional[Dict[str, float]] = None,
        polygon: Optional[Any] = None,
        scale: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Backward-compatible wrapper.

        Trước đây chỉ trả NDVI summary.
        Giờ trả feature summary phong phú hơn nhưng vẫn giữ các field phổ biến như:
        - stats
        - ndvi_mean / ndvi_min / ndvi_max
        """
        products = self._normalize_products([product] if product else None)
        return self.fetch_feature_summary(
            location=location,
            bbox=bbox,
            polygon=polygon,
            time_range=time_range,
            products=products,
            scale=scale,
        )

    def fetch_feature_summary(
        self,
        location: Any = None,
        bbox: Optional[Dict[str, float]] = None,
        polygon: Optional[Any] = None,
        time_range: Optional[Dict[str, Any]] = None,
        products: Optional[List[str]] = None,
        scale: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        API chính cho retrieval theo vùng.

        Output:
        {
            "stats": {... flattened summary ...},
            "feature_groups": {
                "satellite_features": {...},
                "terrain_features": {...},
                "land_cover_features": {...},
                "fuel_features": {...}
            },
            "geometry": {...},
            "time_range": {...},
            "products_used": [...],
            "image_ids": {...},
            "error": None | str
        }
        """
        products = self._normalize_products(products)
        action = (
            "GEE fetch_feature_summary "
            f"location={location} bbox={bbox} polygon={'yes' if polygon else 'no'} "
            f"time_range={time_range} products={products}"
        )
        self.safety_checker.check_or_raise(action)

        try:
            self._ensure_initialized()
        except Exception as e:
            logger.warning("GEE initialization failed: %s", e)
            return self._error_result("GEE unavailable: %s" % e)

        geometry, geometry_meta = self._to_ee_geometry(
            location=location,
            bbox=bbox,
            polygon=polygon,
        )
        if geometry is None:
            return self._error_result(
                "Invalid location/bbox/polygon for GEE",
                geometry=geometry_meta,
            )

        start, end = self._normalize_time_range(time_range)
        if not start or not end:
            return self._error_result(
                "Invalid time_range for GEE",
                geometry=geometry_meta,
                time_range={"start": start, "end": end},
            )

        cache_key = self._cache_key(
            geometry_meta=geometry_meta,
            time_range={"start": start, "end": end},
            products=products,
            scale=scale or self.default_scale,
        )
        if cache_key in self._cache:
            return dict(self._cache[cache_key])

        try:
            raw_bundle = self._fetch_raw_feature_bundle(
                geometry=geometry,
                geometry_meta=geometry_meta,
                start=start,
                end=end,
                products=products,
            )
            summarized = self._summarize_feature_bundle(
                raw_bundle=raw_bundle,
                geometry=geometry,
                geometry_meta=geometry_meta,
                start=start,
                end=end,
                scale=scale or self.default_scale,
            )
            self._cache[cache_key] = dict(summarized)
            return summarized
        except Exception as e:
            logger.exception("GEE feature summary failed: %s", e)
            return self._error_result(
                str(e),
                geometry=geometry_meta,
                time_range={"start": start, "end": end},
            )

    # alias cho DataKnowledgeAgent nếu gọi retrieve()
    def retrieve(self, **kwargs: Any) -> Dict[str, Any]:
        return self.fetch_feature_summary(
            location=kwargs.get("location"),
            bbox=kwargs.get("bbox"),
            polygon=kwargs.get("polygon"),
            time_range=kwargs.get("time_range"),
            products=kwargs.get("products"),
            scale=kwargs.get("scale"),
        )

    # ─────────────────────────────────────────
    # GEE init
    # ─────────────────────────────────────────

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        import ee

        if self.gee_project:
            ee.Initialize(project=self.gee_project)
        else:
            ee.Initialize()
        self._initialized = True

    # ─────────────────────────────────────────
    # Raw fetch layer
    # ─────────────────────────────────────────

    def _fetch_raw_feature_bundle(
        self,
        geometry: Any,
        geometry_meta: Dict[str, Any],
        start: str,
        end: str,
        products: List[str],
    ) -> Dict[str, Any]:
        """
        Raw fetch layer:
        - lấy image / collection từ GEE
        - chưa reduceRegion ở đây
        - chưa flatten stats ở đây
        """
        import ee

        bundle: Dict[str, Any] = {
            "geometry_meta": geometry_meta,
            "time_range": {"start": start, "end": end},
            "products_requested": products,
            "products_used": [],
            "image_ids": {},
            "continuous_images": [],
            "land_cover_image": None,
        }

        if "sentinel2" in products:
            s2 = self._fetch_sentinel2_composite(geometry, start, end)
            if s2 is not None:
                bundle["continuous_images"].append(s2["image"])
                bundle["products_used"].append("sentinel2")
                bundle["image_ids"]["sentinel2"] = s2["image_id"]

        if "landsat" in products:
            landsat = self._fetch_landsat_composite(geometry, start, end)
            if landsat is not None:
                bundle["continuous_images"].append(landsat["image"])
                bundle["products_used"].append("landsat")
                bundle["image_ids"]["landsat"] = landsat["image_id"]

        if "modis_lst" in products:
            modis_lst = self._fetch_modis_lst(geometry, start, end)
            if modis_lst is not None:
                bundle["continuous_images"].append(modis_lst["image"])
                bundle["products_used"].append("modis_lst")
                bundle["image_ids"]["modis_lst"] = modis_lst["image_id"]

        if "srtm" in products:
            terrain = self._fetch_srtm_terrain()
            if terrain is not None:
                bundle["continuous_images"].append(terrain["image"])
                bundle["products_used"].append("srtm")
                bundle["image_ids"]["srtm"] = terrain["image_id"]

        if "land_cover" in products:
            land_cover = self._fetch_land_cover(end)
            if land_cover is not None:
                bundle["land_cover_image"] = land_cover["image"]
                bundle["products_used"].append("land_cover")
                bundle["image_ids"]["land_cover"] = land_cover["image_id"]

        if not bundle["products_used"]:
            raise RuntimeError("No usable GEE products found for the given request")

        return bundle

    def _fetch_sentinel2_composite(
        self,
        geometry: Any,
        start: str,
        end: str,
    ) -> Optional[Dict[str, Any]]:
        import ee

        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(start, end)
            .filterBounds(geometry)
            .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 40))
            .map(self._mask_sentinel2_clouds)
        )

        if self._collection_size(collection) == 0:
            return None

        composite = collection.median()

        red = composite.select("B4").multiply(0.0001)
        nir = composite.select("B8").multiply(0.0001)
        blue = composite.select("B2").multiply(0.0001)

        ndvi = nir.subtract(red).divide(nir.add(red)).rename("ndvi")
        evi = (
            nir.subtract(red)
            .multiply(2.5)
            .divide(nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1))
            .rename("evi")
        )
        vegetation_density = ndvi.rename("vegetation_density_proxy")
        dryness_proxy = (
            ee.Image.constant(1).subtract(ndvi)
            .clamp(0, 1)
            .rename("fuel_dryness_proxy")
        )

        image = ee.Image.cat([ndvi, evi, vegetation_density, dryness_proxy])
        image_id = self._safe_first_system_index(collection)
        return {"image": image, "image_id": image_id}

    def _fetch_landsat_composite(
        self,
        geometry: Any,
        start: str,
        end: str,
    ) -> Optional[Dict[str, Any]]:
        import ee

        collection = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterDate(start, end)
            .filterBounds(geometry)
            .sort("CLOUD_COVER")
        )

        if self._collection_size(collection) == 0:
            return None

        composite = collection.median()
        red = composite.select("SR_B4").multiply(0.0000275).add(-0.2)
        nir = composite.select("SR_B5").multiply(0.0000275).add(-0.2)
        blue = composite.select("SR_B2").multiply(0.0000275).add(-0.2)

        ndvi = nir.subtract(red).divide(nir.add(red)).rename("ndvi")
        evi = (
            nir.subtract(red)
            .multiply(2.5)
            .divide(nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1))
            .rename("evi")
        )
        vegetation_density = ndvi.rename("vegetation_density_proxy")
        dryness_proxy = (
            ee.Image.constant(1).subtract(ndvi)
            .clamp(0, 1)
            .rename("fuel_dryness_proxy")
        )

        image = ee.Image.cat([ndvi, evi, vegetation_density, dryness_proxy])
        image_id = self._safe_first_system_index(collection)
        return {"image": image, "image_id": image_id}

    def _fetch_modis_lst(
        self,
        geometry: Any,
        start: str,
        end: str,
    ) -> Optional[Dict[str, Any]]:
        import ee

        collection = (
            ee.ImageCollection("MODIS/061/MOD11A2")
            .filterDate(start, end)
            .filterBounds(geometry)
        )

        if self._collection_size(collection) == 0:
            return None

        composite = collection.mean()

        lst_day = composite.select("LST_Day_1km").multiply(0.02).subtract(273.15).rename("lst_day_c")
        lst_night = composite.select("LST_Night_1km").multiply(0.02).subtract(273.15).rename("lst_night_c")
        lst_delta = lst_day.subtract(lst_night).rename("lst_diurnal_range")

        image = ee.Image.cat([lst_day, lst_night, lst_delta])
        image_id = self._safe_first_system_index(collection)
        return {"image": image, "image_id": image_id}

    def _fetch_srtm_terrain(self) -> Optional[Dict[str, Any]]:
        import ee

        dem = ee.Image("USGS/SRTMGL1_003")
        terrain = ee.Terrain.products(dem)

        elevation = dem.rename("elevation")
        slope = terrain.select("slope").rename("slope")
        aspect = terrain.select("aspect").rename("aspect")

        image = ee.Image.cat([elevation, slope, aspect])
        return {"image": image, "image_id": "USGS/SRTMGL1_003"}

    def _fetch_land_cover(
        self,
        end: str,
    ) -> Optional[Dict[str, Any]]:
        import ee

        end_year = self._safe_year_from_iso(end)

        collection = ee.ImageCollection("MODIS/061/MCD12Q1")
        try:
            year_img = (
                collection
                .filterDate(f"{end_year}-01-01", f"{end_year}-12-31")
                .first()
            )
            if year_img is None:
                return None
            image = year_img.select("LC_Type1").rename("land_cover")
            return {
                "image": image,
                "image_id": f"MODIS/061/MCD12Q1/{end_year}",
            }
        except Exception:
            # fallback lấy ảnh đầu tiên nếu filter năm lỗi
            first = collection.first()
            if first is None:
                return None
            image = first.select("LC_Type1").rename("land_cover")
            return {
                "image": image,
                "image_id": "MODIS/061/MCD12Q1",
            }

    # ─────────────────────────────────────────
    # Feature summarization layer
    # ─────────────────────────────────────────

    def _summarize_feature_bundle(
        self,
        raw_bundle: Dict[str, Any],
        geometry: Any,
        geometry_meta: Dict[str, Any],
        start: str,
        end: str,
        scale: int,
    ) -> Dict[str, Any]:
        """
        Feature summarization layer:
        - reduceRegion
        - histogram / dominant class
        - fuel proxy summary
        """
        import ee

        continuous_stats: Dict[str, Any] = {}
        land_cover_summary: Dict[str, Any] = {}
        geometry_used = self._geometry_for_summary(geometry, geometry_meta)

        if raw_bundle["continuous_images"]:
            merged = raw_bundle["continuous_images"][0]
            for img in raw_bundle["continuous_images"][1:]:
                merged = merged.addBands(img)

            reducer = (
                ee.Reducer.mean()
                .combine(ee.Reducer.minMax(), sharedInputs=True)
                .combine(ee.Reducer.stdDev(), sharedInputs=True)
            )

            region_stats = merged.reduceRegion(
                reducer=reducer,
                geometry=geometry_used,
                scale=scale,
                maxPixels=self.max_pixels,
                bestEffort=True,
            )
            continuous_stats = region_stats.getInfo() or {}

        if raw_bundle.get("land_cover_image") is not None:
            hist = raw_bundle["land_cover_image"].reduceRegion(
                reducer=ee.Reducer.frequencyHistogram(),
                geometry=geometry_used,
                scale=max(scale, 500),
                maxPixels=self.max_pixels,
                bestEffort=True,
            )
            land_cover_summary = self._parse_land_cover_histogram(hist.getInfo() or {})

        area_km2 = None
        try:
            area_km2 = round(float(geometry_used.area(maxError=1).getInfo()) / 1_000_000.0, 4)
        except Exception:
            pass

        fuel_features = self._build_fuel_proxy_features(continuous_stats, land_cover_summary)

        # flatten output cho model builder cũ/new cùng dùng được
        flat_stats = dict(continuous_stats)
        flat_stats.update(fuel_features)

        result = {
            "stats": flat_stats,
            "feature_groups": {
                "satellite_features": {
                    k: v
                    for k, v in flat_stats.items()
                    if k.startswith("ndvi_")
                    or k.startswith("evi_")
                    or k.startswith("lst_")
                    or k.startswith("vegetation_density_proxy")
                    or k.startswith("fuel_dryness_proxy")
                },
                "terrain_features": {
                    k: v
                    for k, v in flat_stats.items()
                    if k.startswith("elevation_")
                    or k.startswith("slope_")
                    or k.startswith("aspect_")
                },
                "land_cover_features": land_cover_summary,
                "fuel_features": fuel_features,
            },
            "land_cover": land_cover_summary,
            "terrain": {
                "elevation_mean": flat_stats.get("elevation_mean"),
                "slope_mean": flat_stats.get("slope_mean"),
                "aspect_mean": flat_stats.get("aspect_mean"),
            },
            "geometry": {
                **geometry_meta,
                "area_km2": area_km2,
            },
            "time_range": {"start": start, "end": end},
            "products_used": raw_bundle.get("products_used", []),
            "image_ids": raw_bundle.get("image_ids", {}),
            "source": raw_bundle.get("products_used", []),
            "error": None,
        }

        # backward-compatible flattened keys
        result.update(flat_stats)

        return result

    # ─────────────────────────────────────────
    # Geometry helpers
    # ─────────────────────────────────────────

    def _to_ee_geometry(
        self,
        location: Any = None,
        bbox: Optional[Dict[str, float]] = None,
        polygon: Optional[Any] = None,
    ) -> Tuple[Optional[Any], Dict[str, Any]]:
        try:
            import ee
        except Exception:
            return None, {"type": "unknown"}

        # polygon
        if polygon:
            try:
                coords = self._extract_polygon_coordinates(polygon)
                if coords:
                    geom = ee.Geometry.Polygon(coords)
                    return geom, {
                        "type": "polygon",
                        "value": polygon,
                    }
            except Exception:
                pass

        # bbox
        if isinstance(bbox, dict):
            try:
                min_lat = float(bbox["min_lat"])
                min_lon = float(bbox["min_lon"])
                max_lat = float(bbox["max_lat"])
                max_lon = float(bbox["max_lon"])
                geom = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
                return geom, {
                    "type": "bbox",
                    "value": {
                        "min_lat": min_lat,
                        "min_lon": min_lon,
                        "max_lat": max_lat,
                        "max_lon": max_lon,
                    },
                }
            except Exception:
                pass

        # point
        if isinstance(location, dict):
            lat = (
                location.get("lat")
                or location.get("latitude")
            )
            lon = (
                location.get("lon")
                or location.get("lng")
                or location.get("longitude")
            )
            if lat is not None and lon is not None:
                try:
                    lat_f = float(lat)
                    lon_f = float(lon)
                    geom = ee.Geometry.Point([lon_f, lat_f])
                    return geom, {
                        "type": "point",
                        "value": {"lat": lat_f, "lon": lon_f},
                    }
                except Exception:
                    pass

        return None, {"type": "invalid"}

    def _geometry_for_summary(
        self,
        geometry: Any,
        geometry_meta: Dict[str, Any],
    ) -> Any:
        """
        Point -> buffer để summary không quá nhiễu
        Area -> dùng nguyên geometry
        """
        gtype = geometry_meta.get("type")
        if gtype == "point":
            try:
                return geometry.buffer(self.default_buffer_meters)
            except Exception:
                return geometry
        return geometry

    def _extract_polygon_coordinates(self, polygon: Any) -> Optional[List[Any]]:
        """
        Hỗ trợ:
        - GeoJSON-like dict {"type":"Polygon","coordinates":[...]}
        - list tọa độ
        """
        if isinstance(polygon, dict):
            if polygon.get("type") == "Polygon" and polygon.get("coordinates"):
                return polygon["coordinates"]
            if polygon.get("coordinates"):
                return polygon["coordinates"]
            return None

        if isinstance(polygon, list) and polygon:
            return polygon

        return None

    # ─────────────────────────────────────────
    # Misc helpers
    # ─────────────────────────────────────────

    @staticmethod
    def _normalize_time_range(time_range: Optional[Dict[str, Any]]) -> Tuple[str, str]:
        if not isinstance(time_range, dict):
            today = dt.date.today().isoformat()
            return today, today

        start = str(time_range.get("start", "") or time_range.get("start_date", "")).strip()
        end = str(time_range.get("end", "") or time_range.get("end_date", "")).strip()

        if not start and not end:
            today = dt.date.today().isoformat()
            return today, today
        if not start:
            start = end
        if not end:
            end = start
        return start, end

    def _normalize_products(self, products: Optional[List[str]]) -> List[str]:
        if not products:
            return list(self.DEFAULT_PRODUCTS)

        normalized = []
        for p in products:
            if not p:
                continue
            text = str(p).strip().lower()

            # map dataset-like strings -> logical products
            if "sentinel" in text or "s2" in text:
                normalized.append("sentinel2")
            elif "mod11" in text or "modis" in text or "lst" in text:
                normalized.append("modis_lst")
            elif "srtm" in text or "dem" in text or "terrain" in text:
                normalized.append("srtm")
            elif "land" in text and "cover" in text:
                normalized.append("land_cover")
            elif "landsat" in text or "lc08" in text:
                normalized.append("landsat")
            else:
                normalized.append(text)

        out = []
        for item in normalized:
            if item not in out:
                out.append(item)
        return out

    def _parse_land_cover_histogram(self, raw_hist: Dict[str, Any]) -> Dict[str, Any]:
        """
        Kết quả reduceRegion(frequencyHistogram) thường ra:
        {"land_cover": {"1": count, "2": count, ...}}
        """
        hist = raw_hist.get("land_cover", {})
        if not isinstance(hist, dict) or not hist:
            return {
                "histogram": {},
                "fractions": {},
                "dominant_class_id": None,
                "dominant_class_label": None,
            }

        total = 0.0
        parsed = {}
        for key, value in hist.items():
            try:
                k = int(key)
                v = float(value)
                parsed[k] = v
                total += v
            except Exception:
                continue

        fractions = {}
        dominant_class_id = None
        dominant_fraction = -1.0
        for class_id, count in parsed.items():
            frac = (count / total) if total > 0 else 0.0
            fractions[class_id] = round(frac, 6)
            if frac > dominant_fraction:
                dominant_fraction = frac
                dominant_class_id = class_id

        return {
            "histogram": parsed,
            "fractions": fractions,
            "dominant_class_id": dominant_class_id,
            "dominant_class_label": self.LAND_COVER_LABELS.get(dominant_class_id),
        }

    def _build_fuel_proxy_features(
        self,
        stats: Dict[str, Any],
        land_cover_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Fuel/vegetation proxy đơn giản từ feature thật, không bịa raster giả.
        """
        out: Dict[str, Any] = {}

        ndvi_mean = self._to_float(stats.get("ndvi_mean"))
        evi_mean = self._to_float(stats.get("evi_mean"))
        lst_day_mean = self._to_float(stats.get("lst_day_c_mean"))
        dryness_mean = self._to_float(stats.get("fuel_dryness_proxy_mean"))
        dominant_lc = land_cover_summary.get("dominant_class_label")

        if ndvi_mean is not None:
            out["vegetation_density_index"] = round(max(0.0, min(1.0, (ndvi_mean + 1.0) / 2.0)), 6)

        if evi_mean is not None:
            out["fuel_proxy_evi_mean"] = evi_mean

        if dryness_mean is not None:
            out["fuel_dryness_proxy_mean"] = dryness_mean

        if lst_day_mean is not None:
            out["surface_heat_proxy"] = lst_day_mean

        if dominant_lc is not None:
            out["dominant_land_cover_class"] = dominant_lc
            out["dominant_land_cover_id"] = land_cover_summary.get("dominant_class_id")

        return out

    def _mask_sentinel2_clouds(self, image: Any) -> Any:
        import ee

        qa = image.select("QA60")
        cloud_bit = 1 << 10
        cirrus_bit = 1 << 11
        mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
        return image.updateMask(mask)

    def _collection_size(self, collection: Any) -> int:
        try:
            return int(collection.size().getInfo())
        except Exception:
            return 0

    def _safe_first_system_index(self, collection: Any) -> Optional[str]:
        try:
            first = collection.first()
            if first is None:
                return None
            value = first.get("system:index").getInfo()
            return str(value) if value is not None else None
        except Exception:
            return None

    def _safe_year_from_iso(self, iso_text: str) -> int:
        try:
            return int(str(iso_text)[:4])
        except Exception:
            return dt.date.today().year

    def _cache_key(
        self,
        geometry_meta: Dict[str, Any],
        time_range: Dict[str, str],
        products: List[str],
        scale: int,
    ) -> str:
        payload = {
            "geometry": geometry_meta,
            "time_range": time_range,
            "products": products,
            "scale": scale,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)

    def _error_result(
        self,
        error: str,
        geometry: Optional[Dict[str, Any]] = None,
        time_range: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        return {
            "stats": {},
            "feature_groups": {
                "satellite_features": {},
                "terrain_features": {},
                "land_cover_features": {},
                "fuel_features": {},
            },
            "land_cover": {},
            "terrain": {},
            "geometry": geometry or {"type": "unknown"},
            "time_range": time_range or {"start": "", "end": ""},
            "products_used": [],
            "image_ids": {},
            "source": [],
            "error": error,
        }

    def _to_float(self, value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None