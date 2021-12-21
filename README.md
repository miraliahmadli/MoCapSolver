# MoCap Data Denoising
Marker-Based Motion Capture Data Denoising

This repository contains unofficial implementation of [Robust solving of optical motion capture data by denoising](https://dl.acm.org/doi/10.1145/3197517.3201302) and [MoCap-Solver: A Neural Solver for Optical Motion Capture Data](https://dl.acm.org/doi/abs/10.1145/3450626.3459681) papers. For official implementation of __MocapSolver__ paper, please refer to [Official Repo](https://github.com/NetEase-GameAI/MoCap-Solver)

## Docker

To build docker image
```
docker build -t mocap-env -f docker/Dockerfile .
```

To run the docker image creating a container (this command connects the current directory to /host, all the other changes are removed)
```
docker run --rm -it --runtime=nvidia --ipc=host -v $PWD:/host --network=host --name mocap-dev mocap-env /bin/sh -c 'cd /host; Xvfb :5 -screen 0 1920x1080x24 & export DISPLAY=:5; bash'
```

## Dataset

### Synthetic Dataset

Please follow the steps in [Official Implementation](https://github.com/NetEase-GameAI/MoCap-Solver)

__Note__: Data has to preprocessed further, such as converting joint orientations from global to local and etc.

### CMU Dataset

To fetch the dataset
```
sh dataset/get_dataset.sh
```

To reduce the number of markers to 41 and remove one subject from two subjects (Could take long)
```
python dataset/c3d_cleaner.py
```

To parse asf/amc and save global transformation matrices of each frame into npy format (Could take long, must be optimized)
```
python dataset/asfamc2npy.py
```

To create a csv metadata of the dataset (already created .csv file included in the 'dataset/' directory)
```
python dataset/create_meta.py
```

### Dataset Hierarchy

Tree structure of the skeleton for the joint order in the dataset

Please specify the hierarchy of the dataset you are using and save it to `.txt` file.

## Visualization

Use visualization function in tools/viz.py

Usage:

``` python
from tools.viz import visualize
Xs = numpy array of size n x frames x num_marker x 3] # n different marker sequence
colors_X = # n colors for each sequence
Ys = numpy array of size n x frames x num_joint x 3 x 4 # similar to X
colors_Y = # n colors for each sequence
res = [1024, 784] # video resolution
fps_vid = 120 # fps

visualize(Xs=Xs, Ys=Ys, colors_X=colors_X, colors_Y=colors_Y, res=res, fps_vid=fps_vid)

```

## Training Model

1. Create `xxx.json` file and specify parameters
2. Specify which model you are training
3. Run ```python main.py --model [Model name] --config [path to xxx.json]```

### Examples

Training CMU dataset with Holden's model

`python main.py --model RobustSolver --config configs/holden_cmu.json`

Training Synthetic dataset with MocapSolver model

`python main.py --model MocapSolver --config configs/ms_config.json`

## Testing model

Similar to training with `--mode test`.

`python main.py --model RobustSolver --config configs/holden_cmu.json --mode test`

`python main.py --model MocapSolver --config configs/ms_config.json --mode test`

## References
- [MocapSolver](https://github.com/NetEase-GameAI/MoCap-Solver)
- [Fairmotion](https://github.com/facebookresearch/fairmotion)
- [AMCParser](https://github.com/CalciferZh/AMCParser)
- [ezc3d](https://github.com/pyomeca/ezc3d)
- [DeepMotion](https://github.com/DeepMotionEditing/deep-motion-editing/tree/master/retargeting)
- [VNN](https://github.com/FlyingGiraffe/vnn)
