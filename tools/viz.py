# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import numpy as np
import os
import sys
import ezc3d
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from PIL import Image
import cv2

from fairmotion.viz import camera, gl_render, glut_viewer
from fairmotion.data import bvh, asfamc
from fairmotion.ops import conversions, math, motion as motion_ops
from fairmotion.utils import utils

viewer = None
cam = None
focus_markers = [4, 9, 23, 28, 39, 40]
local_ref_markers = [3, 7, 11, 21, 32, 36, 46, 54]
focus_joint = 0

class MocapViewer(glut_viewer.Viewer):
    """
    MocapViewer is an extension of the glut_viewer.Viewer class that implements
    requisite callback functions -- render_callback, keyboard_callback,
    idle_callback and overlay_callback.
    """

    def __init__(
        self,
        markers,
        skels,
        fname,
        fps_vid,
        res,
        focus,
        draw_lrfm,
        play_speed=1.0,
        scale=0.1,
        thickness=1.0,
        render_overlay=False,
        hide_origin=False,
        **kwargs,
    ):
        self.markers = markers
        self.skels = skels
        self.fname = fname
        self.fps_vid = fps_vid
        self.res = res
        self.focus = focus
        self.draw_lrfm = draw_lrfm
        self.play_speed = play_speed
        self.render_overlay = render_overlay
        self.hide_origin = hide_origin
        self.file_idx = 0
        self.cur_time = 0.0
        self.scale = scale
        self.thickness = thickness
        self.saving = False
        super().__init__(**kwargs)

    def save_video(self):
        first_frame_passed = False
        motion = None
        if len(self.skels) != 0:
            motion = self.skels[0]
        else:
            motion = self.markers[0]
        self.cur_time = 0.0
        end_time = motion['length']
        fps = self.fps_vid
        dt = 1 / fps
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(self.fname, fourcc, fps, self.res)
        while self.cur_time <= end_time:
            print(
                f"Recording progress: {self.cur_time:.2f}s/{end_time:.2f}s ({int(100*self.cur_time/end_time)}%) \r",
                end="",
            )
            image = self.get_screen(render=True)
            if first_frame_passed:
                video.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
            else:
                first_frame_passed = True
            self.cur_time += dt
        video.release()
        viewer.destroy()
        

    def keyboard_callback(self, key):
        pass

    def _render_pose(self, pose, adj, body_model, color):
        if self.focus == "skel":
            cam.origin = pose[focus_joint, :, 3].reshape((3, ))
        for j in range(pose.shape[0]):
            rot = pose[j, :, :3].reshape(3, 3)
            pos = pose[j, :, 3].reshape((3, )) 
            for i in adj[j]:
                posc = pose[i, :, 3].reshape((3, ))
                p = 0.5 * (posc + pos)
                l = np.linalg.norm(posc - pos)
                r = 0.1 * self.thickness
                R = math.R_from_vectors(np.array([0, 0, 1]), posc - pos)
                gl_render.render_capsule(
                    conversions.Rp2T(R, p),
                    l,
                    r * self.scale,
                    color=np.array(color) / 255.0,
                    slice=8,
                )
            axes = np.array([[1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]]) * 0.1
            ax = rot.dot(axes)
            pos = pos.reshape(3, 1)
            pts = ax + pos
            gl_render.render_point(pos, radius=0.7 * self.scale, color=np.array(color) / 255.0)#color)

        if not self.saving:
            self.saving = True
            self.save_video()

    def _render_characters(self):
        for i, motion in enumerate(self.skels):
            t = self.cur_time % motion['length']
            pose = motion['data'][int(t * motion['frame_rate'])]

            glEnable(GL_LIGHTING)
            glEnable(GL_DEPTH_TEST)

            glEnable(GL_LIGHTING)
            self._render_pose(pose, motion['adj'], "stick_figure2", motion['color'])
    
    def _render_pose2(self, pose, body_model, color):
        if self.focus == "marker":
            cam.origin = np.mean(pose[focus_markers], axis=-2).reshape((3, ))
        for m in range(pose.shape[0]):
            pos = pose[m].reshape((3, 1))
            if self.draw_lrfm and m in local_ref_markers:
                gl_render.render_point(pos, radius=0.6 * self.scale, color=np.array([0.0, 0.0, 0.0, 255.0]) / 255.0)
            else:
                gl_render.render_point(pos, radius=0.6 * self.scale, color=np.array(color) / 255.0)
                
        if not self.saving:
            self.saving = True
            self.save_video()

    def _render_characters2(self):
        for i, motion in enumerate(self.markers):
            t = self.cur_time % motion['length']
            pose = motion['data'][int(t * motion['frame_rate'])]

            glEnable(GL_LIGHTING)
            glEnable(GL_DEPTH_TEST)

            glEnable(GL_LIGHTING)
            self._render_pose2(pose, "stick_figure2", motion['color'])

    def render_callback(self):
        gl_render.render_ground(
            size=[100, 100],
            color=[0.8, 0.8, 0.8],
            axis='y',
            origin=not self.hide_origin,
            use_arrow=True,
        )
        if len(self.skels) != 0:
            self._render_characters()
        if len(self.markers) != 0:
            self._render_characters2()

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


def visualize(Xs=np.array([]), Ys=np.array([]), colors_X=[[0, 0, 0, 255]], colors_Y=[[0, 255, 0, 255], [255, 0, 0, 255]],
              hierarchy_file="dataset/hierarchy.txt", export_fname="./test.mp4", res=(1920, 1080),
              fps_anim=120.0, fps_vid=25.0, focus="skel", draw_lrfm=False):
    """
    Visualize skeleton and/or markers and export as mp4

    Args:
        Xs: Marker positions as numpy of shape (k x n x m x 3)
        Ys: Joint transformations as numpy of shape (l x n x j x 3 x 4)
        colors_X: Color for markers
        colors_Y: Color for joints and bones
        hierarchy_file: Joint hierarchy file
        export_fname: Filename of the exported video
        res: Resolution of the video
        fps_anim: Fps of the animation
        fps_vid: Fps of the exported video
        focus: "skel" for skeleton, "marker" for ref markers
        draw_lrfm: draw local ref markers in different color

    """
    X, Y = Xs, Ys
    if X.ndim == 3:
        X = np.expand_dims(X, axis=0)
    if Y.ndim == 4:
        Y = np.expand_dims(Y, axis=0)
    
    nX, nY = 0, 0
    adj = None
    
    if X.ndim != 1:
        nX = X.shape[0]
    
    if Y.ndim != 1:
        num_joints = Y.shape[2]
        nY = Y.shape[0]

        adj = [[] for i in range(num_joints)]
        with open(hierarchy_file) as file:
            lines = file.readlines()
            for line in lines:
                splitted = line.split()
                for i in range(len(splitted) - 1):
                    adj[int(splitted[0])].append(int(splitted[i + 1]))
    
    markers = []
    skels = []
    for i in range(nX):
        markers.append({
            'data': X[i],
            'frame_rate': fps_anim,
            'length': X[i].shape[0] / fps_anim,
            'color': colors_X[i % len(colors_X)]
        })
    for i in range(nY):
        skels.append({
            'data': Y[i],
            'frame_rate': fps_anim,
            'length': Y[i].shape[0] / fps_anim,
            'adj': adj,
            'color': colors_Y[i % len(colors_Y)]
        })

    v_up_env = utils.str_to_axis("y")
    global cam
    cam = camera.Camera(
        pos=np.array((15.0, 15.0, 15.0)),
        origin=np.array((0.0, 0.0, 0.0)),#xform[0, 0, :, 3]
        vup=v_up_env,
        fov=45.0,
    )
    global viewer
    viewer = MocapViewer(
        markers=markers,
        skels=skels,
        fname = export_fname,
        fps_vid = fps_vid,
        res = res,
        focus = focus,
        draw_lrfm = draw_lrfm,
        render_overlay=True,
        hide_origin=True,
        title="Motion Graph Viewer",
        cam=cam,
        size=res,
    )
    viewer.run()

def main(args):
    from tools.utils import LBS
    import torch
    a = np.load("dataset/synthetic/70_09_poses_0.npz")
    Y = np.concatenate([a.f.J_R, np.expand_dims(a.f.J_t, axis=-1)], axis=-1)
    markers = a.f.raw_markers
    # Z = np.expand_dims(a.f.marker_configuration, axis=0)
    # w = np.load("dataset/weights.npy")
    # markers = LBS(torch.tensor(w), torch.tensor(Y), torch.tensor(Z), device="cpu")
    # raw_markers = a.f.raw_markers[..., [0, 2, 1]] * 10
    # clean_markers = a.f.clean_markers[..., [0, 2, 1]] * 10
    visualize(Xs=markers[..., [0, 2, 1]] * 10, Ys=Y[..., [0, 2, 1], :] * 10, colors_X=[[255, 0, 0, 255], [0, 0, 255, 255]], hierarchy_file="dataset/hierarchy_synthetic_bfs.txt", draw_lrfm=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize BVH file with block body"
    )
    parser.add_argument("--skel-npy-files", type=str, nargs="+", required=False)
    parser.add_argument("--marker-npy-files", type=str, nargs="+", required=False)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument(
        "--thickness", type=float, default=1.0,
        help="Thickness (radius) of character body"
    )
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
        default=2,
        help="Translates each character by x-offset*idx to display them "
        "simultaneously side-by-side",
    )
    args = parser.parse_args()
    assert len(args.camera_position) == 3 and len(args.camera_origin) == 3, (
        "Provide x, y and z coordinates for camera position/origin like "
        "--camera-position x y z"
    )
    main(args)
