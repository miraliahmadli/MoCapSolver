import argparse
import os

from fairmotion.viz.bvh_visualizer import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize BVH file with block body"
    )
    parser.add_argument("--bvh-files", type=str, nargs="+", required=False)
    parser.add_argument("--asf-files", type=str, nargs="+", required=False)
    parser.add_argument("--amc-files", type=str, nargs="+", required=False)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument(
        "--thickness", type=float, default=1.0,
        help="Thickness (radius) of character body"
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--axis-up", type=str, choices=["x", "y", "z"], default="y"
    )
    parser.add_argument(
        "--axis-face", type=str, choices=["x", "y", "z"], default="x"
    )
    parser.add_argument(
        "--camera-position",
        nargs="+",
        type=float,
        required=False,
        default=(45.0, 45.0, 45.0),
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

