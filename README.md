# Denoising
Marker-Based Motion Capture Data Denoising

**All the scripts should be run from the main project directory!**

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

dataset/hierarchy.txt
- Tree structure of the skeleton for the joint order in the npy files

## Visualization

Use visualization function in tools/viz.py


## Extras
- [Fairmotion](https://github.com/facebookresearch/fairmotion)
- [AMCParser](https://github.com/CalciferZh/AMCParser)
- [ezc3d](https://github.com/pyomeca/ezc3d)
