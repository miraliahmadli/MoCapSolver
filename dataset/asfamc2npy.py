import numpy as np
from pathlib import Path
from tools import amc_parser as amc
import multiprocess
import os, sys

def main():
    BASE_DIR = Path('.') / 'dataset' / 'all_asfamc' / 'subjects'
    asf_paths = list(BASE_DIR.glob('*/*.asf'))
    sys.stdout = open("progress_log.txt", "a")
    DIR_NP = Path('.') / 'dataset' / 'asfamc_npy' / 'subjects'
    DIR_NP.mkdir(parents = True, exist_ok = True)

    def save_as_npy(amc_path):
        Y = np.empty((0, 31, 3, 4))
        motions = amc.parse_amc(amc_path)
        for motion in motions:
            Y_i = np.empty((0, 3, 4))
            joints['root'].set_motion(motion)
            for joint in joints.values():
                xform = np.concatenate([joint.matrix, joint.coordinate], axis = 1).reshape((1, 3, 4))
                Y_i = np.vstack([Y_i, xform])
            Y = np.vstack([Y, Y_i.reshape(1, 31, 3, 4)])
            filename = str(DIR_NP / asf_path.parent.stem / amc_path.stem)
            np.save(filename, Y)
    cur = 1
    for asf_path in asf_paths:
        print("Current file: " + str(asf_path.parent.stem) + " Progress: %" + str((cur / 112.0) * 100.0))
        (DIR_NP / asf_path.parent.stem).mkdir(parents = True, exist_ok = True)
        joints = amc.parse_asf(str(asf_path))
        amc_paths = BASE_DIR.glob(str(asf_path.parent.stem) + "/*.amc")
        with multiprocess.Pool(processes = os.cpu_count()) as pool:
            pool.map(save_as_npy, amc_paths)
        cur += 1

if __name__ == '__main__':
    main()
