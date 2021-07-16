from pathlib import Path
import pandas as pd

def main():
    csv_filename = 'meta_data.csv'
    BASE_DIR = Path('.') / 'dataset'
    df = pd.DataFrame({'amc_path': list(BASE_DIR.glob('all_asfamc/subjects/*/*.amc'))})
    asf_files = list(BASE_DIR.glob('all_asfamc/subjects/*/*.asf'))
    avg_bone = {}
    for file in asf_files:
        avg_single = 0.0
        with open(file, 'r') as content_file:
            content = content_file.read().split('length')
            for i in range(len(content)):
                splitted = content[i].split()
                if i > 1:
                    avg_single += float(splitted[0])
            avg_single /= (len(content) - 2)
        avg_bone[file] = avg_single

    df['asf_path'] = df['amc_path'].map(lambda x: x.parent / (x.parent.stem + '.asf'))
    df['c3d_path'] = df['amc_path'].map(lambda x: BASE_DIR / 'clean_c3d' / 'subjects' / x.parent.stem / (x.stem + '.c3d'))
    df.drop(df[df['c3d_path'].map(lambda x: not x.exists())].index, inplace = True)
    df['motion_npy_path'] = df['amc_path'].map(lambda x: BASE_DIR / 'asfamc_npy' / 'subjects' / x.parent.stem / (x.stem + '.npy'))
    df['subject'] = df['amc_path'].map(lambda x: x.parent.stem)
    df['activity'] = df['amc_path'].map(lambda x: x.stem.split('_')[-1].lower())
    df['avg_bone'] = df['asf_path'].map(lambda x: avg_bone[x])

    df.to_csv(BASE_DIR / csv_filename)

if __name__ == '__main__':
    main()
