import numpy as np


def ade_classes():
    """ADE20K class names for external use."""
    return [
        'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ',
        'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
        'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
        'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug',
        'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe',
        'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
        'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
        'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
        'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door',
        'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table',
        'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove',
        'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
        'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
        'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
        'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister',
        'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
        'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
        'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
        'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
        'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
        'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
        'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
        'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
        'clock', 'flag'
    ]


if __name__ == '__main__':
    # import json
    #
    # with open("test_data.json", 'r') as load_f:
    #     load_dict = json.load(load_f)
    #     # print(load_dict)
    # l = [len(load_dict[j]) for j in load_dict]
    # print(l)
    # gd = np.loadtxt('gd.txt')
    # gd[gd==0]=1
    # gd = np.log(gd)
    # np.savetxt('gd2.txt', gd)

    # raw_filter = [
    #     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
    #     , 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 38
    #     , 39, 40, 41, 42, 43, 44, 45, 46, 47, 50, 51, 52, 53, 55, 57, 59, 60, 61
    #     , 62, 63, 64, 66, 67, 69, 70, 71, 72, 74, 75, 76, 80, 81, 82, 83, 85, 86
    #     , 87, 88, 89, 90, 92, 93, 95, 97, 98, 100, 101, 102, 104, 108, 110, 111, 112, 115
    #     , 116, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 129, 131, 132, 133, 134, 135, 136
    #     , 137, 138, 139, 140, 141, 142, 143, 144, 146, 147, 148, 149
    # ]
    #
    # dell = [59,  46,  50,  51,  52,  74,  77,  81,  83,  72,  90,  93,  94, 110,  98, 103, 100,  86,
    #  44,  27,  64,  61]
    #
    # new = []
    # for s, i in enumerate(raw_filter):
    #     if s in dell:
    #         continue
    #     new.append(i)
    # print(len(new))

    import json
    save_name = 'temp.json'
    with open(save_name,'r') as load_f:
        person_choose = json.load(load_f)

    s = ''
    for k in person_choose:
        s+=' '.join(person_choose[k].keys())+'\n'
    w = open('experiment.txt', 'w')
    w.write(s)
    w.close()