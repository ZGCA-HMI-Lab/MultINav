"""Pure organized-depth to planar-scan projection.

The module deliberately has no ROS dependency.  The bridge supplies the depth
image, camera intrinsics, and the current base-from-lidar transform; the ROS
adapter only wraps the returned arrays in ``sensor_msgs/LaserScan``.  Keeping
the image organized until the continuity checks run is the important invariant:
an unordered PointCloud2 cannot distinguish a supported surface from a depth
edge/background mixture after flattening.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OrganizedDepthScanConfig:
    angle_increment_deg: float = 0.5
    range_min_m: float = 0.1
    range_max_m: float = 8.0
    # A finite depth surface farther than the mapping horizon proves that the
    # ray is free up to that horizon.  Keep its synthetic no-return value below
    # the LaserScan maximum and above the default 7.9 m usable GMapping range.
    no_return_margin_m: float = 0.05
    # Match the PointCloud pseudo-scan contract: a no-return ray requires more
    # than one far pixel in the same angular bin, so an isolated depth outlier
    # never clears map space.
    min_no_return_samples_per_beam: int = 3
    height_min_m: float = 0.05
    height_max_m: float = 1.85
    vertical_window: int = 5
    vertical_min_cover: int = 2
    horizontal_window: int = 3
    horizontal_min_cover: int = 1
    continuity_gap_abs_m: float = 0.08
    continuity_gap_rel: float = 0.04
    support_bins: int = 2
    min_support_neighbors: int = 2
    support_tolerance_abs_m: float = 0.10
    support_tolerance_rel: float = 0.03


@dataclass(frozen=True)
class ProjectedPlanarScan:
    angle_min_rad: float
    angle_increment_rad: float
    range_min_m: float
    range_max_m: float
    ranges_m: np.ndarray
    intensities: np.ndarray
    diagnostics: dict[str, int | float]


class OrganizedDepthScanProjector:
    """Project fresh depth frames, reusing only immutable camera calibration."""

    def __init__(self, config: OrganizedDepthScanConfig | None = None) -> None:
        self.config = config or OrganizedDepthScanConfig()
        if self.config.angle_increment_deg <= 0.0:
            raise ValueError("angle_increment_deg must be positive")
        if self.config.range_min_m <= 0.0 or self.config.range_max_m <= self.config.range_min_m:
            raise ValueError("invalid range limits")
        if not 0.0 < self.config.no_return_margin_m < self.config.range_max_m - self.config.range_min_m:
            raise ValueError("invalid no_return_margin_m")
        if self.config.min_no_return_samples_per_beam < 1:
            raise ValueError("min_no_return_samples_per_beam must be positive")
        if self.config.vertical_window < 1 or self.config.horizontal_window < 1:
            raise ValueError("continuity windows must be positive")
        self._ray_factor_cache = None

    def _ray_factors(self, shape, intrinsics):
        key = (*shape, *intrinsics)
        cached = self._ray_factor_cache
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        height, width = shape
        fx, fy, cx, cy = intrinsics
        columns = -((np.arange(width, dtype=np.float64) - cx) / fx)
        rows = -((np.arange(height, dtype=np.float64) - cy) / fy)
        columns.flags.writeable = rows.flags.writeable = False
        # One assignment keeps concurrent readers on a complete calibration.
        self._ray_factor_cache = (key, columns, rows)
        return columns, rows

    @staticmethod
    def _continuity_support(
        depth: np.ndarray,
        valid: np.ndarray,
        *,
        axis: int,
        window: int,
        min_cover: int,
        gap_abs_m: float,
        gap_rel: float,
    ) -> np.ndarray:
        """Return pixels covered by enough contiguous, smooth windows."""

        length = depth.shape[axis]
        support = np.zeros(depth.shape, dtype=np.int16)
        if length < window:
            return support.astype(bool)

        def section(start, stop):
            slices = [slice(None)] * depth.ndim
            slices[axis] = slice(start, stop)
            return tuple(slices)

        count = length - window + 1
        good = valid[section(0, count)].copy()
        # Compute each adjacent-pixel comparison once instead of materializing
        # overlapping HxWx(window-1) difference/scale/gap arrays.
        if window > 1:
            left, right = depth[section(0, -1)], depth[section(1, None)]
            gap = np.maximum(gap_abs_m, gap_rel * np.minimum(left, right))
            adjacent = (valid[section(0, -1)] & valid[section(1, None)]
                        & (np.abs(right - left) <= gap))
            for offset in range(window - 1):
                good &= adjacent[section(offset, offset + count)]

        for offset in range(window):
            if axis == 0:
                support[offset : offset + good.shape[0], :] += good
            else:
                support[:, offset : offset + good.shape[1]] += good
        return support >= max(1, min_cover)

    @staticmethod
    def _mask_support(
        mask: np.ndarray,
        *,
        axis: int,
        window: int,
        min_cover: int,
    ) -> np.ndarray:
        """Return pixels covered by contiguous true-only mask windows."""

        length = mask.shape[axis]
        support = np.zeros(mask.shape, dtype=np.int16)
        if length < window:
            return support.astype(bool)
        count = length - window + 1
        slices = [slice(None)] * mask.ndim
        slices[axis] = slice(0, count)
        good = mask[tuple(slices)].copy()
        for offset in range(1, window):
            slices[axis] = slice(offset, offset + count)
            good &= mask[tuple(slices)]
        for offset in range(window):
            if axis == 0:
                support[offset : offset + good.shape[0], :] += good
            else:
                support[:, offset : offset + good.shape[1]] += good
        return support >= max(1, min_cover)

    def project(
        self,
        depth_m: np.ndarray,
        intrinsics: tuple[float, float, float, float],
        base_from_lidar: np.ndarray,
    ) -> ProjectedPlanarScan:
        """Project ``depth_m`` into a base-frame planar scan.

        ``depth_m`` is metric optical-frame depth.  ``base_from_lidar`` must
        match the robot-centric frame used by the existing PointCloud2 bridge:
        lidar x=forward, y=left, z=up. Invalid or rejected returns are NaN.
        A valid, continuous depth surface beyond ``range_max_m`` is distinct:
        it emits a finite no-return beam that clears free cells up to the
        mapping horizon without creating an occupied endpoint.  In the sealed
        simulator camera, a spatially continuous NaN/Inf patch means that the
        ray had no hit inside the sensor horizon; it has the same free-to-range
        semantics.  Isolated invalid pixels remain unknown.
        """

        depth = np.asarray(depth_m, dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError("depth_m must be a HxW array")
        fx, fy, cx, cy = (float(value) for value in intrinsics)
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        transform = np.asarray(base_from_lidar, dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("base_from_lidar must be a finite 4x4 transform")

        cfg = self.config
        finite_depth = np.isfinite(depth)
        valid_depth = finite_depth & (depth >= 0.1) & (depth <= 30.0)
        vertical_support = self._continuity_support(
            depth,
            valid_depth,
            axis=0,
            window=cfg.vertical_window,
            min_cover=cfg.vertical_min_cover,
            gap_abs_m=cfg.continuity_gap_abs_m,
            gap_rel=cfg.continuity_gap_rel,
        )
        horizontal_support = self._continuity_support(
            depth,
            valid_depth,
            axis=1,
            window=cfg.horizontal_window,
            min_cover=cfg.horizontal_min_cover,
            gap_abs_m=cfg.continuity_gap_abs_m,
            gap_rel=cfg.continuity_gap_rel,
        )
        supported_pixels = valid_depth & vertical_support & horizontal_support
        no_hit_depth = ~finite_depth
        no_hit_supported = (
            no_hit_depth
            & self._mask_support(
                no_hit_depth,
                axis=0,
                window=cfg.vertical_window,
                min_cover=cfg.vertical_min_cover,
            )
            & self._mask_support(
                no_hit_depth,
                axis=1,
                window=cfg.horizontal_window,
                min_cover=cfg.horizontal_min_cover,
            )
        )

        # Keep the provenance of rejected image samples separate from the
        # accepted scan.  These counters are diagnostics only: the masks below
        # are intentionally unchanged so invalid/unsupported samples remain
        # unknown in the projected scan.
        nonfinite_depth_pixels = no_hit_depth
        out_of_range_depth_pixels = finite_depth & ~valid_depth
        unsupported_valid_pixels = valid_depth & ~supported_pixels
        unsupported_no_hit_pixels = no_hit_depth & ~no_hit_supported

        column_factors, row_factors = self._ray_factors(depth.shape, (fx, fy, cx, cy))
        rows, cols = np.nonzero(supported_pixels)
        if rows.size:
            ranges_depth = depth[rows, cols].astype(np.float64, copy=False)
            x_lidar = ranges_depth
            y_lidar = column_factors[cols] * ranges_depth
            z_lidar = row_factors[rows] * ranges_depth
            points_lidar = np.stack((x_lidar, y_lidar, z_lidar), axis=1)
            rotation = transform[:3, :3]
            translation = transform[:3, 3]
            points_base = points_lidar @ rotation.T + translation
            base_x = points_base[:, 0]
            base_y = points_base[:, 1]
            base_z = points_base[:, 2]
            planar_range = np.hypot(base_x, base_y)
            planar_ray_valid = np.isfinite(planar_range) & (
                planar_range >= cfg.range_min_m
            )
            planar_height_valid = (base_z >= cfg.height_min_m) & (
                base_z <= cfg.height_max_m
            )
            # A finite obstacle endpoint belongs to the planar scan only inside
            # the configured height band.  A supported ray whose first return is
            # beyond the 8 m horizon proves horizontal free space to that
            # horizon regardless of the far endpoint's height (for example a
            # downward camera ray finally hitting the floor).  A coherent near
            # hit in the same angular bin still wins below.
            planar_hit_valid = (
                planar_ray_valid
                & planar_height_valid
                & (planar_range < cfg.range_max_m)
            )
            planar_no_return = planar_ray_valid & (
                planar_range >= cfg.range_max_m
            )
            planar_geometry_valid = planar_hit_valid | planar_no_return
        else:
            planar_geometry_valid = np.zeros(0, dtype=bool)
            planar_hit_valid = np.zeros(0, dtype=bool)
            planar_no_return = np.zeros(0, dtype=bool)
            planar_range = np.zeros(0, dtype=np.float64)
            base_x = np.zeros(0, dtype=np.float64)
            base_y = np.zeros(0, dtype=np.float64)

        beam_count = max(1, int(round(360.0 / cfg.angle_increment_deg)))
        angle_increment = 2.0 * np.pi / float(beam_count)
        angle_min = -np.pi + 0.5 * angle_increment
        # ``np.minimum.at`` needs a finite identity; convert untouched bins to
        # NaN only after all pixel returns have been accumulated.  Keep far
        # evidence separately: a reliable near surface wins, but a near depth
        # edge which later fails angular support must not hide a continuous far
        # free-space observation in the same bearing.
        best_hit_ranges = np.full(beam_count, np.inf, dtype=np.float32)
        no_return_counts = np.zeros(beam_count, dtype=np.int32)
        if np.any(planar_geometry_valid):
            angles = np.arctan2(
                base_y[planar_geometry_valid], base_x[planar_geometry_valid]
            )
            beams = np.floor((angles + np.pi) / angle_increment).astype(np.int64)
            beams = np.clip(beams, 0, beam_count - 1)
            geometry_ranges = planar_range[planar_geometry_valid]
            geometry_hits = planar_hit_valid[planar_geometry_valid]
            geometry_no_returns = planar_no_return[planar_geometry_valid]
            if np.any(geometry_hits):
                np.minimum.at(
                    best_hit_ranges,
                    beams[geometry_hits],
                    geometry_ranges[geometry_hits].astype(np.float32),
                )
            if np.any(geometry_no_returns):
                no_return_counts += np.bincount(
                    beams[geometry_no_returns], minlength=beam_count,
                ).astype(np.int32, copy=False)

        # A continuous non-finite patch is the simulator's explicit no-hit
        # observation.  Project its camera rays to the planar mapping horizon,
        # and encode them as free-only beams.  A no-hit ray is line-of-sight
        # evidence, so its 8 m endpoint need not lie in the obstacle height
        # slice; reliable near hits in the same bearing still take precedence.
        # Solving the planar
        # ray/horizon intersection avoids clearing beyond ``range_max_m`` even
        # when the lidar origin is translated from base_link.
        no_hit_rows, no_hit_cols = np.nonzero(no_hit_supported)
        invalid_no_return_pixels = 0
        if no_hit_rows.size:
            ray_lidar = np.stack(
                (
                    np.ones(no_hit_rows.size, dtype=np.float64),
                    column_factors[no_hit_cols],
                    row_factors[no_hit_rows],
                ),
                axis=1,
            )
            ray_base = ray_lidar @ transform[:3, :3].T
            origin_base = transform[:3, 3]
            dx = ray_base[:, 0]
            dy = ray_base[:, 1]
            horizontal_sq = dx * dx + dy * dy
            linear = 2.0 * (origin_base[0] * dx + origin_base[1] * dy)
            constant = (
                origin_base[0] * origin_base[0]
                + origin_base[1] * origin_base[1]
                - cfg.range_max_m * cfg.range_max_m
            )
            discriminant = linear * linear - 4.0 * horizontal_sq * constant
            ray_valid = horizontal_sq > 1e-12
            ray_valid &= discriminant >= 0.0
            scale = np.zeros(no_hit_rows.size, dtype=np.float64)
            scale[ray_valid] = (
                -linear[ray_valid] + np.sqrt(discriminant[ray_valid])
            ) / (2.0 * horizontal_sq[ray_valid])
            ray_valid &= scale > 0.0
            if np.any(ray_valid):
                endpoint_x = origin_base[0] + scale[ray_valid] * dx[ray_valid]
                endpoint_y = origin_base[1] + scale[ray_valid] * dy[ray_valid]
                no_hit_angles = np.arctan2(endpoint_y, endpoint_x)
                no_hit_beams = np.floor(
                    (no_hit_angles + np.pi) / angle_increment
                ).astype(np.int64)
                no_hit_beams = np.clip(no_hit_beams, 0, beam_count - 1)
                no_return_counts += np.bincount(no_hit_beams, minlength=beam_count).astype(np.int32, copy=False)
                invalid_no_return_pixels = int(ray_valid.sum())

        hit_candidates = np.isfinite(best_hit_ranges)
        no_return_candidates = (
            no_return_counts >= cfg.min_no_return_samples_per_beam
        )
        # The no-return value is below scan.range_max but above the default
        # usable mapping range, so GMapping ray-traces it as free space.
        no_return_range = np.float32(cfg.range_max_m - cfg.no_return_margin_m)
        no_return_ranges = np.full(beam_count, np.nan, dtype=np.float32)
        no_return_ranges[no_return_candidates] = no_return_range

        def _angularly_supported(
            candidate_ranges: np.ndarray, candidates: np.ndarray
        ) -> np.ndarray:
            """Accept same-kind neighboring evidence without mixing hit/free rays."""

            if cfg.min_support_neighbors <= 0 or cfg.support_bins <= 0:
                return candidates.copy()
            support_counts = np.zeros(beam_count, dtype=np.int16)
            for offset in range(1, cfg.support_bins + 1):
                for direction in (-1, 1):
                    neighbor_ranges = np.roll(candidate_ranges, direction * offset)
                    neighbor_candidates = np.roll(candidates, direction * offset)
                    comparable = candidates & neighbor_candidates
                    safe_current = np.where(candidates, candidate_ranges, 0.0)
                    safe_neighbor = np.where(
                        neighbor_candidates, neighbor_ranges, 0.0
                    )
                    tolerance = np.maximum(
                        cfg.support_tolerance_abs_m,
                        cfg.support_tolerance_rel
                        * np.maximum(safe_current, safe_neighbor),
                    )
                    range_delta = np.zeros(beam_count, dtype=np.float32)
                    range_delta[comparable] = np.abs(
                        candidate_ranges[comparable] - neighbor_ranges[comparable]
                    )
                    support_counts += (comparable & (range_delta <= tolerance)).astype(np.int16)
            return candidates & (support_counts >= cfg.min_support_neighbors)

        accepted_hits = _angularly_supported(best_hit_ranges, hit_candidates)
        accepted_no_returns = _angularly_supported(
            no_return_ranges, no_return_candidates
        )
        # An accepted obstacle endpoint always wins.  If no such surface exists
        # (or it was rejected as an angularly isolated depth edge), retain the
        # independently supported far evidence as a free-space-only ray.
        no_return_selected = accepted_no_returns & ~accepted_hits
        accepted = accepted_hits | no_return_selected
        ranges = np.full(beam_count, np.nan, dtype=np.float32)
        ranges[accepted_hits] = best_hit_ranges[accepted_hits]
        ranges[no_return_selected] = no_return_range
        # 1.0 is an obstacle return, 2.0 a supported no-return free-space ray.
        # Both are observations; downstream consumers can avoid treating the
        # latter as an occupied endpoint.
        intensities = np.where(
            accepted,
            np.where(no_return_selected, 2.0, 1.0),
            0.0,
        ).astype(np.float32, copy=False)
        diagnostics: dict[str, int | float] = {
            "input_valid_pixels": int(valid_depth.sum()),
            "invalid_depth_pixels": int((~valid_depth).sum()),
            "nonfinite_depth_pixels": int(nonfinite_depth_pixels.sum()),
            "out_of_range_depth_pixels": int(out_of_range_depth_pixels.sum()),
            "vertical_supported_pixels": int((valid_depth & vertical_support).sum()),
            "organized_supported_pixels": int(supported_pixels.sum()),
            "unsupported_valid_pixels": int(unsupported_valid_pixels.sum()),
            "unsupported_no_hit_pixels": int(unsupported_no_hit_pixels.sum()),
            "planar_candidate_pixels": int(planar_hit_valid.sum()),
            "no_return_pixels": int(planar_no_return.sum())
            + invalid_no_return_pixels,
            "invalid_no_return_pixels": invalid_no_return_pixels,
            "candidate_beams": int((hit_candidates | no_return_candidates).sum()),
            "candidate_hit_beams": int(hit_candidates.sum()),
            "candidate_no_return_beams": int(no_return_candidates.sum()),
            "accepted_beams": int(accepted.sum()),
            "accepted_hit_beams": int(accepted_hits.sum()),
            "accepted_no_return_beams": int((accepted & no_return_selected).sum()),
            "accepted_no_return_free_beams": int(no_return_selected.sum()),
            "unknown_output_beams": int((~accepted).sum()),
            "insufficient_no_return_samples_beams": int(
                ((no_return_counts > 0) & ~no_return_candidates).sum()
            ),
            "support_rejected_hit_beams": int(
                (hit_candidates & ~accepted_hits).sum()
            ),
            "support_rejected_no_return_beams": int(
                (no_return_candidates & ~accepted_no_returns).sum()
            ),
            "no_return_suppressed_by_hit_beams": int(
                (accepted_no_returns & accepted_hits).sum()
            ),
            "fallback_no_return_beams": int(
                (no_return_selected & hit_candidates).sum()
            ),
            "rejected_angular_support": int(
                ((hit_candidates | no_return_candidates) & ~accepted).sum()
            ),
            "beam_count": int(beam_count),
            "angle_increment_deg": float(np.rad2deg(angle_increment)),
        }
        return ProjectedPlanarScan(
            angle_min_rad=float(angle_min),
            angle_increment_rad=float(angle_increment),
            range_min_m=float(cfg.range_min_m),
            range_max_m=float(cfg.range_max_m),
            ranges_m=ranges,
            intensities=intensities,
            diagnostics=diagnostics,
        )
