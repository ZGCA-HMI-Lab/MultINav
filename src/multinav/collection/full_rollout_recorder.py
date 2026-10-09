from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


SCHEMA_VERSION = "interactive_nav_full_rollout_v1"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class H5StepRolloutRecorder:
    """Incrementally write synchronized images, actions and state for one rollout."""

    def __init__(
        self,
        path: Path,
        *,
        episode_id: str,
        camera_names: Iterable[str],
        metadata: dict[str, Any] | None = None,
        compression: str | None = "lzf",
    ) -> None:
        self.path = Path(path)
        self.partial_path = self.path.with_suffix(self.path.suffix + ".partial")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.camera_names = tuple(camera_names)
        if not self.camera_names:
            raise ValueError("At least one camera is required for full rollout recording")
        self.compression = compression
        self.handle = h5py.File(self.partial_path, "w")
        self.handle.attrs["schema_version"] = SCHEMA_VERSION
        self.handle.attrs["episode_id"] = str(episode_id)
        self.handle.attrs["status"] = "recording"
        self.handle.attrs["metadata_json"] = json.dumps(
            _jsonable(metadata or {}), ensure_ascii=False, sort_keys=True
        )
        self.steps = self.handle.create_group("steps")
        self.images = self.steps.create_group("images")
        self.actions = self.steps.create_group("actions")
        self.states = self.steps.create_group("states")
        text_dtype = h5py.string_dtype(encoding="utf-8")
        float_vlen = h5py.vlen_dtype(np.dtype("float32"))
        self.step_index = self.steps.create_dataset(
            "step_index", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=True
        )
        self.timestamp_seconds = self.steps.create_dataset(
            "timestamp_seconds", shape=(0,), maxshape=(None,), dtype=np.float64, chunks=True
        )
        self.dt_seconds = self.steps.create_dataset(
            "dt_seconds", shape=(0,), maxshape=(None,), dtype=np.float32, chunks=True
        )
        self.segment = self.steps.create_dataset(
            "segment", shape=(0,), maxshape=(None,), dtype=text_dtype, chunks=True
        )
        self.phase = self.steps.create_dataset(
            "phase", shape=(0,), maxshape=(None,), dtype=text_dtype, chunks=True
        )
        self.action_type = self.actions.create_dataset(
            "type", shape=(0,), maxshape=(None,), dtype=text_dtype, chunks=True
        )
        self.action_vector = self.actions.create_dataset(
            "vector", shape=(0,), maxshape=(None,), dtype=float_vlen, chunks=True
        )
        self.action_json = self.actions.create_dataset(
            "json", shape=(0,), maxshape=(None,), dtype=text_dtype, chunks=True
        )
        self.state_json = self.states.create_dataset(
            "json", shape=(0,), maxshape=(None,), dtype=text_dtype, chunks=True
        )
        self.qpos = self.states.create_dataset(
            "qpos", shape=(0,), maxshape=(None,), dtype=float_vlen, chunks=True
        )
        self.qvel = self.states.create_dataset(
            "qvel", shape=(0,), maxshape=(None,), dtype=float_vlen, chunks=True
        )
        self.reward = self.steps.create_dataset(
            "reward", shape=(0,), maxshape=(None,), dtype=np.float32, chunks=True
        )
        self.terminal = self.steps.create_dataset(
            "terminal", shape=(0,), maxshape=(None,), dtype=np.bool_, chunks=True
        )
        self.truncated = self.steps.create_dataset(
            "truncated", shape=(0,), maxshape=(None,), dtype=np.bool_, chunks=True
        )
        self.info_json = self.steps.create_dataset(
            "info_json", shape=(0,), maxshape=(None,), dtype=text_dtype, chunks=True
        )
        self.image_datasets: dict[str, h5py.Dataset] = {}
        self.count = 0
        self.closed = False

    @staticmethod
    def _resize(dataset: h5py.Dataset, size: int) -> None:
        dataset.resize((size, *dataset.shape[1:]))

    def _image_dataset(self, camera_name: str, frame: np.ndarray) -> h5py.Dataset:
        frame = np.asarray(frame, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[-1] not in {3, 4}:
            raise ValueError(
                f"Camera {camera_name} frame must have HxWx3/4 shape, got {frame.shape}"
            )
        if camera_name not in self.image_datasets:
            dataset = self.images.create_dataset(
                camera_name,
                shape=(0, *frame.shape),
                maxshape=(None, *frame.shape),
                dtype=np.uint8,
                chunks=(1, *frame.shape),
                compression=self.compression,
            )
            self.image_datasets[camera_name] = dataset
        dataset = self.image_datasets[camera_name]
        if dataset.shape[1:] != frame.shape:
            raise ValueError(
                f"Camera {camera_name} shape changed from {dataset.shape[1:]} to {frame.shape}"
            )
        return dataset

    def record_step(
        self,
        *,
        images: dict[str, np.ndarray],
        action: dict[str, Any] | None,
        state: dict[str, Any] | None,
        segment: str,
        phase: str,
        reward: float = 0.0,
        terminal: bool = False,
        truncated: bool = False,
        info: dict[str, Any] | None = None,
        timestamp_seconds: float | None = None,
        dt_seconds: float | None = None,
    ) -> None:
        if self.closed:
            raise RuntimeError("Cannot append to a finalized rollout recorder")
        missing = [name for name in self.camera_names if name not in images]
        if missing:
            raise KeyError(f"Missing rollout camera frames: {missing}")
        next_size = self.count + 1
        scalar_datasets = [
            self.step_index,
            self.timestamp_seconds,
            self.dt_seconds,
            self.segment,
            self.phase,
            self.action_type,
            self.action_vector,
            self.action_json,
            self.state_json,
            self.qpos,
            self.qvel,
            self.reward,
            self.terminal,
            self.truncated,
            self.info_json,
        ]
        for dataset in scalar_datasets:
            self._resize(dataset, next_size)
        for camera_name in self.camera_names:
            dataset = self._image_dataset(camera_name, images[camera_name])
            self._resize(dataset, next_size)
            dataset[self.count] = np.asarray(images[camera_name], dtype=np.uint8)

        action = _jsonable(action or {"type": "initial", "vector": []})
        state = _jsonable(state or {})
        vector = np.asarray(action.get("vector", []), dtype=np.float32).reshape(-1)
        self.step_index[self.count] = self.count
        self.timestamp_seconds[self.count] = float(
            self.count if timestamp_seconds is None else timestamp_seconds
        )
        self.dt_seconds[self.count] = float(1.0 if dt_seconds is None else dt_seconds)
        self.segment[self.count] = str(segment)
        self.phase[self.count] = str(phase)
        self.action_type[self.count] = str(action.get("type", "unknown"))
        self.action_vector[self.count] = vector
        self.action_json[self.count] = json.dumps(action, ensure_ascii=False, sort_keys=True)
        self.state_json[self.count] = json.dumps(state, ensure_ascii=False, sort_keys=True)
        self.qpos[self.count] = np.asarray(state.get("qpos", []), dtype=np.float32).reshape(-1)
        self.qvel[self.count] = np.asarray(state.get("qvel", []), dtype=np.float32).reshape(-1)
        self.reward[self.count] = float(reward)
        self.terminal[self.count] = bool(terminal)
        self.truncated[self.count] = bool(truncated)
        self.info_json[self.count] = json.dumps(
            _jsonable(info or {}), ensure_ascii=False, sort_keys=True
        )
        self.count = next_size
        self.handle.flush()

    def finalize(
        self,
        *,
        success: bool,
        terminal_reason: str,
        result: dict[str, Any] | None = None,
    ) -> Path:
        if self.closed:
            return self.path
        self.handle.attrs["status"] = "complete"
        self.handle.attrs["success"] = bool(success)
        self.handle.attrs["terminal_reason"] = str(terminal_reason)
        self.handle.attrs["step_count"] = int(self.count)
        self.handle.attrs["result_json"] = json.dumps(
            _jsonable(result or {}), ensure_ascii=False, sort_keys=True
        )
        self.handle.flush()
        self.handle.close()
        self.partial_path.replace(self.path)
        self.closed = True
        return self.path

    def abort(self, reason: str) -> Path:
        if self.closed:
            return self.path
        return self.finalize(success=False, terminal_reason=reason)

    def close(self) -> None:
        if not self.closed:
            self.abort("recorder_closed_without_finalize")

    def __enter__(self) -> "H5StepRolloutRecorder":
        return self

    def __exit__(self, exc_type, exc, _traceback) -> bool:
        if exc is not None:
            self.abort(f"{exc_type.__name__}: {exc}")
        elif not self.closed:
            self.abort("context_exited_without_finalize")
        return False


def validate_full_rollout(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unexpected full rollout schema version")
        if handle.attrs.get("status") != "complete":
            raise ValueError("Full rollout is not finalized")
        count = int(handle.attrs.get("step_count", -1))
        if count <= 0:
            raise ValueError("Full rollout contains no steps")
        steps = handle["steps"]
        required = [
            steps["step_index"],
            steps["timestamp_seconds"],
            steps["dt_seconds"],
            steps["segment"],
            steps["phase"],
            steps["actions/type"],
            steps["actions/vector"],
            steps["states/json"],
            steps["reward"],
            steps["terminal"],
        ]
        lengths = {len(dataset) for dataset in required}
        lengths.update(len(dataset) for dataset in steps["images"].values())
        if lengths != {count}:
            raise ValueError(f"Full rollout step arrays are misaligned: {sorted(lengths)}")
        if not steps["images"]:
            raise ValueError("Full rollout contains no camera images")
        decode = lambda value: value.decode("utf-8") if isinstance(value, bytes) else str(value)
        action_type_counts = Counter(decode(value) for value in steps["actions/type"][:])
        segment_counts = Counter(decode(value) for value in steps["segment"][:])
        terminal_step_count = int(np.count_nonzero(steps["terminal"][:]))
        dt_values = np.asarray(steps["dt_seconds"][:], dtype=float)
        if not np.allclose(dt_values, dt_values[0], rtol=0.0, atol=1e-7):
            raise ValueError("Full rollout contains non-uniform training dt values")
        return {
            "schema_version": SCHEMA_VERSION,
            "episode_id": str(handle.attrs["episode_id"]),
            "step_count": count,
            "success": bool(handle.attrs.get("success", False)),
            "terminal_reason": str(handle.attrs.get("terminal_reason", "")),
            "camera_names": sorted(steps["images"].keys()),
            "action_type_counts": dict(action_type_counts),
            "segment_counts": dict(segment_counts),
            "terminal_step_count": terminal_step_count,
        }
