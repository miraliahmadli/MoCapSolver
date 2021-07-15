# Denoising
Marker-Based Motion Capture Data Denoising


## Docker

To build docker image
```
docker build -t mocap-env -f docker/Dockerfile .
```

To run the docker image creating a container (this command connects the current directory to /host, all the other changes are removed)
```
docker run --rm -it --runtime=nvidia --ipc=host -v $PWD:/host --network=host --name mocap-dev mocap-env /bin/sh -c 'cd /host; bash'
```

## Dataset

To fetch the dataset
```
sh get_dataset.sh
```

To reduce the number of markers to 41 and remove one subject from two subjects (Could take long)
```
python c3d_cleaner.py
```

To parse asf/amc and save global transformation matrices of each frame into npy format (Could take long, must be optimized)
```
python asfamc2npy.py
```

## Visualization
```
python tools/viz/viz_bvh.py --bvh-files [space separated .bvh filenames]
python tools/viz/viz_c3d.py --c3d-files [space seperated .c3d filenames]
python tools/viz/viz_c3damc.py --c3d-files [space seperated .c3d filenames] --amc-files [space seperated .amc filenames]
python tools/viz/viz_c3dbvh.py --c3d-files [space seperated .c3d filenames] --bvh-files [space seperated .bvh filenames]
```

## Extras
- [Fairmotion](https://github.com/facebookresearch/fairmotion)
- [AMCParser](https://github.com/CalciferZh/AMCParser)
- [ezc3d](https://github.com/pyomeca/ezc3d)
