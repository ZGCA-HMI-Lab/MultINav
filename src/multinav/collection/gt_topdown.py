"""Render the final scene and project the recorded world-space GT route onto it."""
import json

import mujoco
import numpy as np
from PIL import Image, ImageDraw


def project_world(points, eye, forward, up, near, half_height, size):
    right = np.cross(forward, up)
    delta = np.asarray(points)-eye
    depth = delta @ forward
    scale = near / depth / half_height * size / 2
    return np.column_stack([size/2+(delta@right)*scale, size/2-(delta@up)*scale])


def render_topdown(model, data, view, frames, output):
    size = 640
    points = np.array([[*f['base_xy_yaw'][:2], .06] for f in frames])
    lo, hi = points[:, :2].min(axis=0)-2, points[:, :2].max(axis=0)+2
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [*((lo+hi)/2), 0]
    camera.distance = float(max(hi-lo)/2/np.tan(np.deg2rad(model.vis.global_.fovy)/2) + 2)
    camera.elevation, camera.azimuth = -90, 90
    renderer = mujoco.Renderer(model, height=size, width=size)
    try:
        renderer.update_scene(data, camera=camera, scene_option=view)
        rgb = renderer.render().copy()
        cameras = renderer.scene.camera
        eye = np.mean([c.pos for c in cameras], axis=0)
        forward, up = np.array(cameras[0].forward), np.array(cameras[0].up)
        near = float(cameras[0].frustum_near)
        half_height = float((cameras[0].frustum_top-cameras[0].frustum_bottom)/2)
        pixels = project_world(points, eye, forward, up, near, half_height, size)
        image = Image.fromarray(rgb)
        image.save(output/'topdown_scene.png')
        draw = ImageDraw.Draw(image)
        draw.line([tuple(p) for p in pixels], fill='#51bdff', width=3)
        for p, color, label in [(pixels[0], '#78efac', 'START'), (pixels[-1], '#ffc65c', 'END')]:
            draw.ellipse([*(p-5), *(p+5)], fill=color)
            draw.text(tuple(p+7), label, fill=color, stroke_width=1, stroke_fill='black')
        draw.line([(30, 610), (75, 610)], fill='#ff7777', width=3)
        draw.line([(30, 610), (30, 565)], fill='#7cdd9a', width=3)
        draw.text((78, 602), '+X', fill='white'); draw.text((18, 548), '+Y', fill='white')
        draw.text((12, 12), 'TOP VIEW | +Z above | final scene + complete GT path', fill='white', stroke_width=1, stroke_fill='black')
        image.save(output/'topdown_trajectory.png')
        (output/'topdown.json').write_text(json.dumps({'scene_state': 'terminal', 'size': [size, size],
            'eye_xyz': eye.tolist(), 'forward': forward.tolist(), 'up': up.tolist(),
            'near': near, 'half_height_at_near': half_height,
            'base_path_world': points.tolist(), 'base_path_pixels': pixels.tolist(),
            'axes': '+X right, +Y up, viewing down -Z'}, indent=2))
    finally:
        renderer.close()
