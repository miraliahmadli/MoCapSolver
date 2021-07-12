import os
import ezc3d
from random import randint


two_subjects = [('18', '19'), ('20', '21'), ('22', '23'), ('33', '34')]
subject_names =\
        [("Justin", 'rory'), ("Justin1", 'rory1'), 
        ("justin", "Rory"), ("Justin", "TwoSubject")]
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
    if "LABELS3" in c1['parameters']['POINT'].keys():
        c1['parameters']['POINT']['LABELS']['value'] += c1['parameters']['POINT']['LABELS3']['value']
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
    if "LABELS3" in c2['parameters']['POINT'].keys():
        c2['parameters']['POINT']['LABELS']['value'] += c2['parameters']['POINT']['LABELS3']['value']
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

    del c1['data']['meta_points']
    del c2['data']['meta_points']

    if save:
        # Write the data
        c1.write(out_paths[0])
        c2.write(out_paths[1])

    return c1, c2


def clean_markers(c3d_file="", c3d=None, out_path="", save=True):
    if c3d is None:
        c = ezc3d.c3d(c3d_file)
    else:
        c = c3d

    if "LABELS2" in c['parameters']['POINT'].keys():
        c['parameters']['POINT']['LABELS']['value'] += c['parameters']['POINT']['LABELS2']['value']
    if "LABELS3" in c['parameters']['POINT'].keys():
        c['parameters']['POINT']['LABELS']['value'] += c['parameters']['POINT']['LABELS3']['value']
    labels = c['parameters']['POINT']['LABELS']['value']

    duplicate_labels = {l:[] for l in main_labels}
    labels_to_remove = []
    pred = [False for _ in range(len(labels))]

    # find the labels with multiple markers
    for i, label in enumerate(labels):
        _, label_name = parse_label(label)
        is_label = False
        for l in main_labels:
            if label_name.startswith(l):
                duplicate_labels[l].append((i, label_name, label))
                pred[i] = False
                is_label = True
                break
        if not is_label:
            labels_to_remove.append(label)

    # randomly pick one element and remove others
    for label, duplicates in duplicate_labels.items():
        length = len(duplicates)
        if length == 0: 
            continue

        pick = randint(0, length-1)
        idx, _, _ = duplicates[pick]
        c['parameters']['POINT']['LABELS']['value'][idx] = label
        pred[idx] = True
        duplicates.pop(pick)

        for _, _, l in duplicates:
            labels_to_remove.append(l)

    for label in labels_to_remove:
        c['parameters']['POINT']['LABELS']['value'].remove(label)
    c['data']['points'] = c['data']['points'][:, pred, :]

    if save:
        del c['data']['meta_points']
        c.write(out_path)

    return c

if __name__ == "__main__":
    c3d_file = "../subjects/01/01_01.c3d"
    c3d_file = "../subjects/02/02_04.c3d"
    c3d_files = ["../subjects/18/18_06.c3d", "../subjects/19/19_06.c3d"]

    # out_path = "data/01/01_01.c3d"
    # c = clean_markers(c3d_file, out_path)

    out_paths = ["data/18/18_06.c3d", "data/19/19_06.c3d"]
    c = remove_one_subject(c3d_files, subject_names[0], out_paths, save=True)
