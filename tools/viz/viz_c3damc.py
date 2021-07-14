# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import numpy as np
import os
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import ezc3d
import c3d
from tools import amc_parser as amc

from PIL import Image

from fairmotion.viz import camera, gl_render, glut_viewer
from fairmotion.data import bvh, asfamc
from fairmotion.ops import conversions, math, motion as motion_ops
from fairmotion.utils import utils
from pathlib import Path


class MocapViewer(glut_viewer.Viewer):
    """
    MocapViewer is an extension of the glut_viewer.Viewer class that implements
    requisite callback functions -- render_callback, keyboard_callback,
    idle_callback and overlay_callback.

    ```
    python fairmotion/viz/bvh_visualizer.py \
        --bvh-files $BVH_FILE1
    ```

    To visualize more than 1 motion sequence side by side, append more files 
    to the `--bvh-files` argument. Set `--x-offset` to an appropriate float 
    value to add space separation between characters in the row.

    ```
    python fairmotion/viz/bvh_visualizer.py \
        --bvh-files $BVH_FILE1 $BVH_FILE2 $BVH_FILE3 \
        --x-offset 2
    ```

    To visualize asfamc motion sequence:

    ```
    python fairmotion/viz/bvh_visualizer.py \
        --asf-files tests/data/11.asf \
        --amc-files tests/data/11_01.amc
    ```

    """

    def __init__(
        self,
        motions,
        play_speed=1.0,
        scale=1.0,
        thickness=15.0,
        render_overlay=False,
        hide_origin=False,
        **kwargs,
    ):
        self.motions = motions[0]
        self.motions_c3d = motions[1]
        self.play_speed = play_speed
        self.render_overlay = render_overlay
        self.hide_origin = hide_origin
        self.file_idx = 0
        self.cur_time = 0.0
        self.scale = scale
        self.thickness = thickness
        super().__init__(**kwargs)

    def keyboard_callback(self, key):
        pass
        # motion = self.motions[self.file_idx]
        # if key == b"s":
        #     self.cur_time = 0.0
        #     self.time_checker.begin()
        # elif key == b"]":
        #     next_frame = min(
        #         motion.num_frames() - 1,
        #         motion.time_to_frame(self.cur_time) + 1,
        #     )
        #     self.cur_time = motion.frame_to_time(next_frame)
        # elif key == b"[":
        #     prev_frame = max(0, motion.time_to_frame(self.cur_time) - 1)
        #     self.cur_time = motion.frame_to_time(prev_frame)
        # elif key == b"+":
        #     self.play_speed = min(self.play_speed + 0.2, 5.0)
        # elif key == b"-":
        #     self.play_speed = max(self.play_speed - 0.2, 0.2)
        # elif (key == b"r" or key == b"v"):
        #     self.cur_time = 0.0
        #     end_time = motion.length()
        #     fps = motion.fps
        #     save_path = input(
        #         "Enter directory/file to store screenshots/video: "
        #     )
        #     cnt_screenshot = 0
        #     dt = 1 / fps
        #     gif_images = []
        #     while self.cur_time <= end_time:
        #         print(
        #             f"Recording progress: {self.cur_time:.2f}s/{end_time:.2f}s ({int(100*self.cur_time/end_time)}%) \r",
        #             end="",
        #         )
        #         if key == b"r":
        #             utils.create_dir_if_absent(save_path)
        #             name = "screenshot_%04d" % (cnt_screenshot)
        #             self.save_screen(dir=save_path, name=name, render=True)
        #         else:
        #             image = self.get_screen(render=True)
        #             gif_images.append(
        #                 image.convert("P", palette=Image.ADAPTIVE)
        #             )
        #         self.cur_time += dt
        #         cnt_screenshot += 1
        #     if key == b"v":
        #         utils.create_dir_if_absent(os.path.dirname(save_path))
        #         gif_images[0].save(
        #             save_path,
        #             save_all=True,
        #             optimize=False,
        #             append_images=gif_images[1:],
        #             loop=0,
        #         )
        # else:
        #     return False

        # return True

    def _render_pose(self, pose, body_model, color):
        joints, child, parent = pose
        for pos_scaled in joints:
            pos = pos_scaled * 0.056444 #inch fix
            gl_render.render_point(pos, radius=0.3 * self.scale, color=np.array([255, 255, 0, 255]) / 255.0)#color)
        for i in range(len(child)):
            pp = parent[i] * 0.056444
            cc = child[i] * 0.056444
            p = 0.5 * (pp + cc)
            l = np.linalg.norm(pp - cc)
            r = 0.1 * self.thickness
            R = math.R_from_vectors(np.array([0, 0, 1]), pp - cc)
            gl_render.render_capsule(
                conversions.Rp2T(R, p),
                l,
                r * self.scale,
                color=color,
                slice=8,
            )

    def _render_characters(self, colors):
        for i, motion in enumerate(self.motions):
            t = self.cur_time % motion['length']
            # pose = motion.get_pose_by_frame(motion.time_to_frame(t))
            pose = motion['data'][int(t * motion['frame_rate'])]
            color = colors[i % len(colors)]

            glEnable(GL_LIGHTING)
            glEnable(GL_DEPTH_TEST)

            glEnable(GL_LIGHTING)
            self._render_pose(pose, "stick_figure2", color)

    def _render_pose_c3d(self, pose, body_model, color):
        for j in range(pose.shape[1]):
            pos = pose[: 3, j] * 1
            pos[[0, 1, 2]] = pos[[1, 2, 0]]
            gl_render.render_point(pos, radius=0.2 * self.scale, color=np.array([255, 0, 0, 255]) / 255.0)#color)

    def _render_characters_c3d(self, colors):
        for i, motion in enumerate(self.motions_c3d):
            t = self.cur_time % motion['length']
            # pose = motion.get_pose_by_frame(motion.time_to_frame(t))
            pose = motion['data'][:, :, int(t * motion['frame_rate'])]
            color = colors[i % len(colors)]

            glEnable(GL_LIGHTING)
            glEnable(GL_DEPTH_TEST)

            glEnable(GL_LIGHTING)
            self._render_pose_c3d(pose, "stick_figure2", color)

    def render_callback(self):
        gl_render.render_ground(
            size=[100, 100],
            color=[0.8, 0.8, 0.8],
            axis='y',
            origin=not self.hide_origin,
            use_arrow=True,
        )
        colors = [
            np.array([123, 174, 85, 255]) / 255.0,  # green
            np.array([255, 255, 0, 255]) / 255.0,  # yellow
            np.array([85, 160, 173, 255]) / 255.0,  # blue
        ]
        self._render_characters(colors)
        self._render_characters_c3d(colors)

    def idle_callback(self):
        time_elapsed = self.time_checker.get_time(restart=False)
        self.cur_time += self.play_speed * time_elapsed
        self.time_checker.begin()

    def overlay_callback(self):
        pass
        # if self.render_overlay:
        #     w, h = self.window_size
        #     t = self.cur_time % self.motions[0].length()
        #     frame = self.motions[0].time_to_frame(t)
        #     gl_render.render_text(
        #         f"Frame number: {frame}",
        #         pos=[0.05 * w, 0.95 * h],
        #         font=GLUT_BITMAP_TIMES_ROMAN_24,
        #     )

class Viewer:
    def __init__(self, joints=None, motions=None):
        """
        Display motion sequence in 3D.
        Parameter
        ---------
        joints: Dict returned from `amc_parser.parse_asf`. Keys are joint names and
        values are instance of Joint class.
        motions: List returned from `amc_parser.parse_amc. Each element is a dict
        with joint names as keys and relative rotation degree as values.
        """
        self.joints = joints
        self.motions = motions
        self.frame = 0 # current frame of the motion sequence
        self.playing = True # whether is playing the motion sequence
        self.fps = 120 # frame rate

        # whether is dragging
        self.rotate_dragging = False
        self.translate_dragging = False
        # old mouse cursor position
        self.old_x = 0
        self.old_y = 0
        # global rotation
        self.global_rx = 0
        self.global_ry = 0
        # rotation matrix for camera moving
        self.rotation_R = np.eye(3)
        # rotation speed
        self.speed_rx = np.pi / 90
        self.speed_ry = np.pi / 90
        # translation speed
        self.speed_trans = 0.25
        self.speed_zoom = 0.5
        # whether the main loop should break
        self.done = False
        # default translate set manually to make sure the skeleton is in the middle
        # of the window
        # if you can't see anything in the screen, this is the first parameter you
        # need to adjust
        self.default_translate = np.array([0, 0, 0], dtype=np.float32)
        self.translate = np.copy(self.default_translate)

    def set_joints(self, joints):
        """
        Set joints for viewer.
        Parameter
        ---------
        joints: Dict returned from `amc_parser.parse_asf`. Keys are joint names and
        values are instance of Joint class.
        """
        self.joints = joints

    def set_motion(self, motions):
        """
        Set motion sequence for viewer.
        Paramter
        --------
        motions: List returned from `amc_parser.parse_amc. Each element is a dict
        with joint names as keys and relative rotation degree as values.
        """
        self.motions = motions

    def draw(self):
        """
        Draw the skeleton with balls and sticks.
        """
        a, b, c = [], [], []
        for j in self.joints.values():
            coord = np.array(
                np.squeeze(j.coordinate).dot(self.rotation_R) + \
                self.translate, dtype=np.float32
            )
            a.append(coord)

        for j in self.joints.values():
            child = j
            parent = j.parent
            if parent is not None:
                coord_x = np.array(
                    np.squeeze(child.coordinate).dot(self.rotation_R)+self.translate,
                    dtype=np.float32
                )
                b.append(coord_x)
                coord_y = np.array(
                    np.squeeze(parent.coordinate).dot(self.rotation_R)+self.translate,
                    dtype=np.float32
                )
                c.append(coord_y)
        return (a, b, c)

    def run(self):
        """
        Main loop.
        """
        a = []
        while not self.done:
            self.joints['root'].set_motion(self.motions[self.frame])
            if self.playing:
                self.frame += 1
                if self.frame >= len(self.motions):
                    return a
            a.append(self.draw())

def main(args):
    v_up_env = utils.str_to_axis(args.axis_up)
    if args.amc_files:
        motions = []
        for filename in args.amc_files:
            amcfile = filename
            fn_splitted = filename.split('_')
            asffile = ''.join(fn_splitted[: -1]) + '.asf'
            j = amc.parse_asf(asffile)
            m = amc.parse_amc(amcfile)
            v = Viewer(j, m)
            mo = v.run()
            motions.append({
                'data': mo,  #inch fix
                'frame_rate': 120.0,
                'length': len(mo) / 120.0
            })
    if args.c3d_files:
        motions2 = []
        for filename in args.c3d_files:
            c = ezc3d.c3d(filename)
            c['data']['points'] = c['data']['points'][:, :, :-2]
            motions2.append({
                'data': c['data']['points'] * 0.001,  #inch fix
                'frame_rate': c['header']['points']['frame_rate'],
                'length': c['data']['points'].shape[2] / c['header']['points']['frame_rate']
            })
            
    cam = camera.Camera(
        pos=np.array(args.camera_position),
        origin=np.array(args.camera_origin),
        vup=v_up_env,
        fov=45.0,
    )
    viewer = MocapViewer(
        motions=[motions, motions2],
        play_speed=args.speed,
        scale=args.scale,
        thickness=args.thickness,
        render_overlay=args.render_overlay,
        hide_origin=args.hide_origin,
        title="Motion Graph Viewer",
        cam=cam,
        size=(1280, 720),
    )
    viewer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize BVH file with block body"
    )
    parser.add_argument("--c3d-files", type=str, nargs="+", required=False)
    parser.add_argument("--asf-files", type=str, nargs="+", required=False)
    parser.add_argument("--amc-files", type=str, nargs="+", required=False)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument(
        "--thickness", type=float, default=1.0,
        help="Thickness (radius) of character body"
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--axis-up", type=str, choices=["x", "y", "z"], default="y"
    )
    parser.add_argument(
        "--axis-face", type=str, choices=["x", "y", "z"], default="y"
    )
    parser.add_argument(
        "--camera-position",
        nargs="+",
        type=float,
        required=False,
        default=(5.0, 5.0, 5.0),
    )
    parser.add_argument(
        "--camera-origin",
        nargs="+",
        type=float,
        required=False,
        default=(0.0, 0.0, 0.0),
    )
    parser.add_argument("--hide-origin", action="store_true")
    parser.add_argument("--render-overlay", action="store_true")
    parser.add_argument(
        "--x-offset",
        type=int,
        default=0,
        help="Translates each character by x-offset*idx to display them "
        "simultaneously side-by-side",
    )
    args = parser.parse_args()
    assert len(args.camera_position) == 3 and len(args.camera_origin) == 3, (
        "Provide x, y and z coordinates for camera position/origin like "
        "--camera-position x y z"
    )
    main(args)
