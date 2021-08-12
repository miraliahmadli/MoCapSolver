from pathlib import Path
import pandas as pd
import ezc3d
import math


exclude = ['103_05', '93_05', '87_02', '89_04', '89_05', # Issues with the number of markers 
           '79_01', '79_02', '79_04', '79_05', '79_06',
           '79_09', '79_10', '79_12', '79_13', '79_14',
           '79_15', '79_16', '79_17', '79_18', '79_19',
           '79_20', '79_21', '79_23', '79_24', '79_26',
           '79_27', '79_28', '79_29', '79_31', '79_32',
           '79_36', '79_37', '79_38', '79_39', '79_41',
           '79_42', '79_43', '79_44', '79_54', '79_64',
           '79_66', '79_71', '79_78', '79_79', '79_80',
           '79_81', '79_82', '79_83', '79_84', '79_85',
           '79_86', '79_88', '79_89', '79_90', '79_91',
           '79_92', '79_93', '79_94', '79_95', '79_96',
           '111_02', '111_16', '111_25', '111_32', '111_37',
           '90_30', '90_20', 
           '90_31', '106_21', '141_21', '90_21', '90_19',
           '80_67', '74_09', '84_09', '83_11', '122_11'] # Issues with the skeleton

def get_fps(c3d_file):
    return ezc3d.c3d(str(c3d_file))['header']['points']['frame_rate']

def main():
    csv_filename = 'meta_data.csv'
    BASE_DIR = Path('.') / 'dataset'
    df = pd.DataFrame({'bvh_path': list(BASE_DIR.glob('all_bvh/subjects/*/*.bvh'))})
    bvh_files = list(BASE_DIR.glob('all_bvh/subjects/*/*.bvh'))
    avg_bone = {}
    for file in bvh_files:
        avg_single = 0.0
        with open(file, 'r') as content_file:
            content = content_file.read().split()
            idx = [i for i, x in enumerate(content) if x == "OFFSET"]
            n = 0
            for i in idx:
                x, y, z = float(content[i + 1]), float(content[i + 2]), float(content[i + 3])
                if x == 0 and y == 0 and z == 0:
                    continue
                n += 1
                avg_single += math.sqrt(x ** 2 + y ** 2 + z ** 2)
            avg_single /= n
        avg_bone[file] = avg_single

    df['c3d_path'] = df['bvh_path'].map(lambda x: BASE_DIR / 'clean_c3d' / 'subjects' / x.parent.stem / (x.stem + '.c3d'))
    df.drop(df[df['c3d_path'].map(lambda x: not x.exists())].index, inplace = True)
    df.drop(df[df['c3d_path'].map(lambda x: x.stem in exclude)].index, inplace = True)
    df['motion_npy_path'] = df['bvh_path'].map(lambda x: BASE_DIR / 'bvh_npy' / 'subjects' / x.parent.stem / (x.stem + '.npy'))
    df['marker_npy_path'] = df['c3d_path'].map(lambda x: BASE_DIR / 'c3d_npy' / 'subjects' / x.parent.stem / (x.stem + '.npy'))
    df['subject'] = df['bvh_path'].map(lambda x: x.parent.stem)
    df['activity'] = df['bvh_path'].map(lambda x: x.stem.split('_')[-1].lower())
    df['file_stem'] = df['bvh_path'].map(lambda x: x.stem)
    df['frame_rate'] = df['c3d_path'].map(lambda x: get_fps(x))
    df['avg_bone'] = df['bvh_path'].map(lambda x: avg_bone[x])

    df.to_csv(BASE_DIR / csv_filename)

if __name__ == '__main__':
    main()
