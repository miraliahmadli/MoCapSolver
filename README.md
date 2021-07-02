# Denoising
Marker-Based Motion Capture Data Denoising


## Dependencies

+ fairmotion

To install fairmotion from source, first clone the git repository, use pip to download dependencies and build the project.
```
git clone https://github.com/facebookresearch/fairmotion.git
cd fairmotion
pip install -e .
```

+ ezc3d

To install ezc3d
```
conda install -c conda-forge cmake numpy swig
conda install -c conda-forge ezc3d
```

## Visualization
```
python tools/viz/viz_bvh.py --bvh-files [space separated .bvh filenames]
python tools/viz/viz_c3d.py --c3d-files [space seperated .c3d filenames]
```
