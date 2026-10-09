"""Fixed-clock GT motion primitives; independent of simulator and rendering."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Sampling:
    hz: float = 5.0
    base_speed: float = 0.5
    yaw_speed: float = 0.8
    hinge_speed: float = 0.5
    slide_speed: float = 0.1
    handle_speed: float = 0.8

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("All rates and speeds must be finite and positive")


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def uniform_values(start: float, end: float, speed: float, hz: float):
    """Exclude start; keep exact speed except the endpoint interval remainder."""
    if speed <= 0 or hz <= 0:
        raise ValueError("speed and hz must be positive")
    distance = abs(end - start)
    if distance < 1e-10:
        return
    n = math.ceil(distance * hz / speed - 1e-10)
    for i in range(1, n + 1):
        yield start + math.copysign(min(i * speed / hz, distance), end - start), i == n


def simplify_collinear(points):
    points = np.asarray(points, dtype=float)[:, :2]
    out = [points[0]]
    for i in range(1, len(points) - 1):
        a, b = points[i] - out[-1], points[i + 1] - points[i]
        if np.linalg.norm(a) < 1e-9:
            continue
        if abs(a[0] * b[1] - a[1] * b[0]) > 1e-9 or np.dot(a, b) < 0:
            out.append(points[i])
    if np.linalg.norm(points[-1] - out[-1]) > 1e-9:
        out.append(points[-1])
    return np.asarray(out)


def navigation_samples(points, start_yaw, goal_yaw, sampling):
    """Turn in place, translate at fixed speed, then match the operation yaw.

    No corner cutting: preserve the collision-checked polyline. Endpoint
    remainder intervals are explicit, since exact distance, arbitrary speed
    and a fixed clock cannot all imply a full-speed final interval.
    """
    points = simplify_collinear(points)
    yaw = float(start_yaw)
    for a, b in zip(points[:-1], points[1:]):
        delta = b - a
        heading = yaw + wrap(math.atan2(delta[1], delta[0]) - yaw)
        for value, last in uniform_values(yaw, heading, sampling.yaw_speed, sampling.hz):
            yield np.array([*a, value]), "turn", last
        yaw = heading
        distance = float(np.linalg.norm(delta))
        for value, last in uniform_values(0, distance, sampling.base_speed, sampling.hz):
            yield np.array([*(a + delta * value / distance), yaw]), "navigate", last
    target = yaw + wrap(goal_yaw - yaw)
    for value, last in uniform_values(yaw, target, sampling.yaw_speed, sampling.hz):
        yield np.array([*points[-1], value]), "turn", last


def pose_matrix(xyz, rotation):
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = xyz
    return out


def pose_record(matrix):
    r = Rotation.from_matrix(matrix[:3, :3])
    return {
        "xyz_rpy": [*matrix[:3, 3].tolist(), *r.as_euler("xyz").tolist()],
        "xyz_quat_wxyz": [*matrix[:3, 3].tolist(), *r.as_quat(scalar_first=True).tolist()],
    }


def narration_vertices(points, tolerance=.35):
    """RDP is for language only: never alter the collision-checked motion."""
    points = np.asarray(points)
    if len(points) <= 2:
        return list(range(len(points)))
    delta = points[-1] - points[0]
    t = np.clip((points-points[0]) @ delta / max(delta @ delta, 1e-12), 0, 1)
    distances = np.linalg.norm(points - (points[0] + t[:, None]*delta), axis=1)
    k = int(np.argmax(distances))
    if distances[k] <= tolerance:
        return [0, len(points)-1]
    return narration_vertices(points[:k+1], tolerance)[:-1] + [k+i for i in narration_vertices(points[k:], tolerance)]


def rule_instruction(segments, target):
    """Sparse route instructions: large turns, grouped travel, no angle jargon."""
    clauses = []
    names = {'door': '房门', 'refrigerator': '冰箱', 'fridge': '冰箱',
             'chestofdrawers': '抽屉柜', 'dresser': '抽屉柜', 'drawer': '抽屉', 'cabinet': '柜子',
             'apple': '苹果', 'egg': '鸡蛋', 'tomato': '番茄', 'lettuce': '生菜',
             'remotecontrol': '遥控器', 'cellulartelephone': '手机',
             'compactdisk': '光盘', 'irishpotato': '土豆'}
    categories = {s.get('interaction_id'): names.get(s.get('object_category', '').lower(), s.get('object_category', '目标物体'))
                  for s in segments if s['phase'] == 'open'}
    def add(text, first, last):
        clauses.append({'text': text, 'start_frame': first['start_frame'],
                        'end_frame': last['end_frame'], 'interaction_id': first.get('interaction_id')})

    index = 0
    while index < len(segments):
        segment = segments[index]
        phase = segment["phase"]
        index += 1
        if phase in {'navigate', 'turn'}:
            block = [segment]
            while index < len(segments) and segments[index]['phase'] in {'navigate', 'turn'}:
                block.append(segments[index]); index += 1
            moves = [s for s in block if s['phase'] == 'navigate']
            words, distance = [], 0.
            if moves and all('end_xy' in s for s in moves):
                points = np.asarray([moves[0]['start_xy']] + [s['end_xy'] for s in moves])
                vertices = narration_vertices(points)
                heading = block[0].get('start_yaw', 0.)
                for a, b in zip(vertices[:-1], vertices[1:]):
                    delta = points[b] - points[a]
                    next_heading = math.atan2(delta[1], delta[0])
                    angle = wrap(next_heading-heading)
                    if abs(angle) >= math.radians(45) and np.linalg.norm(delta) >= .6:
                        if distance:
                            words.append(f'直行（约 {distance:.1f} 米）'); distance = 0.
                        words.append('向后转' if abs(angle) > math.radians(150) else ('左转' if angle > 0 else '右转'))
                        heading = next_heading
                    distance += sum(s['distance_m'] for s in moves[a:b])
            else:
                distance = sum(s['distance_m'] for s in moves)
            if distance:
                words.append(f'直行（约 {distance:.1f} 米）')
            destination = categories.get(block[-1].get('interaction_id'))
            if destination:
                words.append(f'在{destination}前停下')
            if words:
                add('，'.join(words), block[0], block[-1])
            continue
        elif phase == "rotate_handle":
            if index < len(segments) and segments[index]['phase'] == 'open' and segments[index].get('interaction_id') == segment.get('interaction_id'):
                end = segments[index]; index += 1
                obj = categories.get(segment.get('interaction_id'), '房门')
                add(f"转动把手并{'拉开' if end.get('pull') else '打开'}{obj}", segment, end)
                continue
            text = "转动把手"
        elif phase == "open":
            verb = "拉开" if segment.get("pull") else "打开"
            text = f"{verb}{categories.get(segment.get('interaction_id'), names.get(segment.get('object_category', '').lower(), '容器'))}"
        elif phase == "observe_target":
            text = f"寻找{names.get(target.lower(), target)}"
        else:
            continue
        add(text, segment, segment)
    return {"generator": "rule_sparse_v2", "locale": "zh-CN", "clauses": clauses,
            "large_turn_threshold_degrees": 45, "narration_path_tolerance_m": .35,
            "instruction": "。".join(c["text"] for c in clauses) + "。"}
