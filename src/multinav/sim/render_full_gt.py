"""Build an offline 3D trajectory viewer from actual full GT collection output."""
import argparse
import json
from pathlib import Path


def build_viewer(root):
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    episodes = []
    for result in summary["results"]:
        directory = root / result["directory"]
        frames_path = directory / "frames.json"
        frames = json.loads(frames_path.read_text()) if frames_path.exists() else []
        instruction_path = directory / "instruction.json"
        instruction = json.loads(instruction_path.read_text())["instruction"] if instruction_path.exists() else ""
        episodes.append({"index": result["source_index"], "house": result["house_index"],
            "passed": result["passed"], "error": result.get("error", ""),
            "directory": result["directory"], "instruction": instruction,
            "topdown": f"{result['directory']}/topdown_trajectory.png" if (directory/'topdown_trajectory.png').exists() else None,
            "frames": [{"b": f["base_xy_yaw"], "t": f["time"], "p": f["phase"],
                        "id": f["interaction_id"],
                        "o": f["operation_point"]["xyz_rpy"] if f["operation_point"] else None,
                        "e": f["ee_target"]["xyz_rpy"] if f["ee_target"] else None} for f in frames]})
    data = json.dumps({"hz": manifest["parameters"]["hz"], "episodes": episodes}, ensure_ascii=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("collection") / "full_gt_viewer.html"
    output = root / "index.html"
    output.write_text(template.read_text().replace("__FULL_GT_DATA__", data))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(build_viewer(args.run.resolve()))
