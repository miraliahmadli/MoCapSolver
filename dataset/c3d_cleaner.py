import os
from random import randint

import numpy as np
import ezc3d
from tqdm import tqdm
import multiprocess

c3d_dir = "dataset/all_c3d/subjects"
c3d_folders = os.listdir(c3d_dir)
two_subject_folders = ['18', '19', '20', '21', '22', '23', '33', '34']

two_subjects = [('18', '19'), ('20', '21'), ('22', '23'), ('33', '34')]
subject_names =\
        [("Justin", 'rory'), ("Justin1", 'rory1'), 
        ("justin", "Rory"), ("Justin", "TwoSubject1")]
main_labels =\
        ['C7', 'CLAV', 'LANK', 'LBHD', 'LBWT', 'LELB', 'LFHD', 'LFIN', 
        'LFRM', 'LFWT', 'LHEE', 'LKNE', 'LMT5', 'LSHN', 'LSHO', 'LTHI', 
        'LTOE', 'LUPA', 'LWRA', 'LWRB', 'RANK', 'RBAC', 'RBHD', 'RBWT', 
        'RELB', 'RFHD', 'RFIN', 'RFRM', 'RFWT', 'RHEE', 'RKNE', 'RMT5', 
        'RSHN', 'RSHO', 'RTHI', 'RTOE', 'RUPA', 'RWRA', 'RWRB', 'STRN', 'T10']


def parse_label(label):
    splitted = label.split(":")
    subject = None
    if len(splitted) == 2:
        subject = splitted[0]
    label_name = splitted[-1]
    return subject, label_name


def remove_one_subject(c3d_files, subjects, out_paths=[], save=True):
    # c = ezc3d.c3d(c3d_file)
    fn1, fn2 = c3d_files
    sub1, sub2 = subjects
    c1 = ezc3d.c3d(fn1)
    c2 = ezc3d.c3d(fn2)

    # cleanup first file
    if "LABELS2" in c1['parameters']['POINT'].keys():
        c1['parameters']['POINT']['LABELS']['value'] += c1['parameters']['POINT']['LABELS2']['value']
        c1['parameters']['POINT']['LABELS2']['value'] = []
    if "LABELS3" in c1['parameters']['POINT'].keys():
        c1['parameters']['POINT']['LABELS']['value'] += c1['parameters']['POINT']['LABELS3']['value']
        c1['parameters']['POINT']['LABELS3']['value'] = []
    labels1 = c1['parameters']['POINT']['LABELS']['value']

    pred1 = [True for _ in range(len(labels1))]
    labels_to_remove = []
    for i, label in enumerate(labels1):
        sub, label_name = parse_label(label)
        if sub != sub1:
            pred1[i] = False
            labels_to_remove.append(label)

    for l in labels_to_remove:
        c1['parameters']['POINT']['LABELS']['value'].remove(l)

    c1['data']['points'] = c1['data']['points'][:, pred1, :]

    # cleanup second file
    if "LABELS2" in c2['parameters']['POINT'].keys():
        c2['parameters']['POINT']['LABELS']['value'] += c2['parameters']['POINT']['LABELS2']['value']
        c2['parameters']['POINT']['LABELS2']['value'] = []
    if "LABELS3" in c2['parameters']['POINT'].keys():
        c2['parameters']['POINT']['LABELS']['value'] += c2['parameters']['POINT']['LABELS3']['value']
        c2['parameters']['POINT']['LABELS3']['value'] = []
    labels2 = c2['parameters']['POINT']['LABELS']['value']

    pred2 = [True for _ in range(len(labels2))]
    labels_to_remove = []
    for i, label in enumerate(labels2):
        sub, _ = parse_label(label)
        if sub != sub2:
            pred2[i] = False
            labels_to_remove.append(label)

    for l in labels_to_remove:
        c2['parameters']['POINT']['LABELS']['value'].remove(l)
    c2['data']['points'] = c2['data']['points'][:, pred2, :]

    if save:
        # Write the data
        clean_markers(c3d=c1, out_path=out_paths[0])
        clean_markers(c3d=c2, out_path=out_paths[1])


def clean_markers(c3d_file="", c3d=None, out_path="", save=True):
    if c3d is None:
        c = ezc3d.c3d(c3d_file)
    else:
        c = c3d

    if "LABELS2" in c['parameters']['POINT'].keys():
        c['parameters']['POINT']['LABELS']['value'] += c['parameters']['POINT']['LABELS2']['value']
        c['parameters']['POINT']['LABELS2']['value'] = []
    if "LABELS3" in c['parameters']['POINT'].keys():
        c['parameters']['POINT']['LABELS']['value'] += c['parameters']['POINT']['LABELS3']['value']
        c['parameters']['POINT']['LABELS3']['value'] = []
    labels = c['parameters']['POINT']['LABELS']['value']

    duplicate_labels = {l:[] for l in main_labels}
    labels_to_remove = []
    pred = [False for _ in range(len(labels))]

    # find the labels with multiple markers
    for i, label in enumerate(labels):
        label = label.replace("ANT", "LBHD")
        label = label.replace("NOSE", "RBHD")
        _, label_name = parse_label(label)
        is_label = False
        for l in main_labels:
            if label_name.startswith(l):
                duplicate_labels[l].append((label_name, label, i))
                pred[i] = False
                is_label = True
                break
        if not is_label:
            labels_to_remove.append(label)

    for label, duplicates in duplicate_labels.items():
        length = len(duplicates)
        if length == 0: 
            continue

        pick = duplicates.index(min(duplicates))
        _, _, idx = duplicates[pick]
        c['parameters']['POINT']['LABELS']['value'][idx] = label
        pred[idx] = True
        duplicates.pop(pick)

        for _, l, _ in duplicates:
            labels_to_remove.append(l)

    for label in labels_to_remove:
        c['parameters']['POINT']['LABELS']['value'].remove(label)
    c['data']['points'] = c['data']['points'][:, pred, :]

    labels = c['parameters']['POINT']['LABELS']['value']
    sorted_labels = sorted(enumerate(labels), key=lambda x:x[1])
    indices = [l[0] for l in sorted_labels]
    c['parameters']['POINT']['LABELS']['value'] = [l[1] for l in sorted_labels]
    c['data']['points'] = c['data']['points'][:, indices, :]

    if save:
        np.save(out_path, c['data']['points'])


def clean_all(save_dir):
    for (folder1, folder2), subjects in zip(two_subjects, subject_names):
        path1 = os.path.join(c3d_dir, folder1)
        path2 = os.path.join(c3d_dir, folder2)
        c3d_files1 = [os.path.join(path1, fn) for fn in sorted(os.listdir(path1))]
        c3d_files2 = [os.path.join(path2, fn) for fn in sorted(os.listdir(path2))]

        tqdm_batch = tqdm(total=len(c3d_files1), dynamic_ncols=True)
        for i, (fn1, fn2) in enumerate(zip(c3d_files1, c3d_files2)):
            out_path1 = fn1.replace(c3d_dir,save_dir).replace(".c3d",".npy")
            out_path2 = fn2.replace(c3d_dir,save_dir).replace(".c3d",".npy")
            remove_one_subject((fn1, fn2), subjects, (out_path1, out_path2))

            tqdm_update = f"iteration={i},files={fn1}, {fn2}"
            tqdm_batch.set_postfix_str(tqdm_update)
            tqdm_batch.update()
        tqdm_batch.close()

    def parallel_fn(folder):
        if folder in two_subject_folders:
            return
        in_path = os.path.join(c3d_dir, folder)
        c3d_files = [os.path.join(in_path, fn) for fn in sorted(os.listdir(in_path))]

        for i, fn in enumerate(c3d_files):
            out_path = fn.replace(c3d_dir,save_dir).replace(".c3d",".npy")
            clean_markers(c3d_file=fn, out_path=out_path)

        print(f'Done: {folder}')
    with multiprocess.Pool(processes = os.cpu_count()) as pool:
        pool.map(parallel_fn, c3d_folders)


def main(data_dir="dataset/c3d_npy/subjects/"):
    dir_split = data_dir.split("/")
    if not os.path.exists(dir_split[0] + "/" + dir_split[1]):
        os.mkdir(dir_split[0] + "/" + dir_split[1])
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    for folder in c3d_folders:
        path = os.path.join(data_dir, folder)
        if not os.path.exists(path):
            os.mkdir(path)
    clean_all(data_dir)


if __name__ == "__main__":
    main()
    # c3d_file = "../subjects/01/01_01.c3d"
    # c3d_file = "../subjects/02/02_04.c3d"
    # c3d_files = ["../subjects/18/18_09.c3d", "../subjects/19/19_09.c3d"]

    # out_path = "data/01/01_01.c3d"
    # c = clean_markers(c3d_file, out_path=out_path)

    # out_paths = ["data/18/18_09.c3d", "data/19/19_09.c3d"]
    # c = remove_one_subject(c3d_files, subject_names[0], out_paths, save=True)
