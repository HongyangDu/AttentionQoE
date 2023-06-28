import os
import shutil

import numpy as np
import json
import matplotlib.pyplot as plt
import cv2
from util import ade_classes
from tqdm import tqdm
import numpy as np
import pandas as pd

save = '../UOAL/Labels'
raw_image = '../UOAL/Images'
fix = '../UOAL/Attention'

# save = '/Users/liujiazhen/Downloads/pas_save_new'
# raw_image = '/Users/liujiazhen/Downloads/all_images_release'
# fix = '/Users/liujiazhen/Downloads/fixation_map_30_release'


def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0, size):
        list_of_objects.append( list() ) #different object reference each time
    return list_of_objects

class PersonSplit():
    def __init__(self, seg_path, image_path, fix_map_path):
        self.seg_path = seg_path
        self.image_path = image_path
        self.fix_map_path = fix_map_path

        self.seg_files = os.listdir(seg_path)[:]
        self.seg_suffix = '.txt'
        self.image_files = os.listdir(image_path)[:]
        self.image_suffix = '.jpg'
        self.fix_map_files = os.listdir(fix_map_path)[:]  # contain dirs
        self.fix_suffix = '.npy'

        self.pre_load_seg = {}
        self.pre_load_fix = {}
        self.pre_load_img = {}

        self.raw_filter = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                           11, 12, 13, 14, 15, 16, 17, 18,
                           19, 20, 21, 22, 23, 24, 26, 27,
                           28, 30, 31, 32, 33, 34, 36, 38,
                           39, 41, 42, 43, 44, 45, 46,
                           47, 51, 53, 55, 57, 62, 63, 64,
                           66, 67, 69, 71, 74, 75, 80, 81,
                           82, 83, 85, 86, 87, 89, 92, 93, 97,
                           98, 100, 102, 108, 110, 112, 115, 116,
                           119, 120, 123, 124, 125, 127, 131, 132,
                           134, 135, 136, 137, 138, 139, 141, 142, 143,
                           144, 146, 147, 148, 149]

        for file in tqdm(self.seg_files):
            self.pre_load_seg[file] = np.loadtxt(os.path.join(self.seg_path, file)).astype(int)
        for file in tqdm(self.fix_map_files):
            self.pre_load_fix[file] = {}
            if not os.path.isdir(os.path.join(self.fix_map_path, file)):
                continue
            for sub_file in os.listdir(os.path.join(self.fix_map_path, file)):
                if sub_file.split(self.fix_suffix)[0]+self.seg_suffix not in self.seg_files:
                    continue
                self.pre_load_fix[file][sub_file] = np.load(os.path.join(self.fix_map_path, file, sub_file))

        print(len(self.raw_filter))
        s = 0
        self.filter = {}
        for i in self.raw_filter:
            self.filter[i] = s
            s += 1

    @staticmethod
    def getCluRes(path='clu_label.txt', suffix='.txt'):
        """
        Parameters
        ----------
        path clu_results.txt: dir/path, class
        Returns dict: classes of all images
        -------
        """
        maxx = -1
        clu = open(path, 'r')
        clu_cs = {}
        for i in clu.readlines():
            name = i.split('\t')[0]
            if '\\' in name:
                name = name.split('\\')[-1].split(suffix)[0]
            else:
                name = os.path.split(name)[-1].split(suffix)[0]
            cls = int(i.split('\t')[1].replace('\n', ''))

            clu_cs[name] = cls  # 下标从0开始，连续
            maxx = max(maxx, cls)
        clu.close()

        return clu_cs, maxx + 1

    def get_person_all(self, person_num=30):
        persons_see = {}
        persons_value = {}
        for person in range(person_num):
            person = str(person)
            person_image = self.seg_files
            persons_see[person] = {}
            persons_value[person] = {}
            for im in tqdm(person_image):
                im = im.replace(self.seg_suffix, '')
                seg_result = self.pre_load_seg[im + self.seg_suffix]
                # fix_file = os.path.join(fix, 'Sub_' + str(person + 1), im+self.fix_suffix)
                fix_im = self.pre_load_fix['User' + str(int(person) + 1)][im + self.fix_suffix]
                if not fix_im.shape == seg_result.shape:
                    print(im)
                    continue
                cls_set = set(list(seg_result.reshape(-1)))
                tmp = []
                for cc in cls_set:
                    if cc in self.filter:
                        tmp.append(cc)
                    # else:
                    #     fix_im[seg_result == int(cc)] = 0 # 不在筛选的类里，直接全部删除
                cls_set = tmp
                # sm = fix_im.sum()
                for cls in cls_set:  # get number of each person's scene
                    trans_cls = str(self.filter[cls])
                    cls = str(cls)

                    if trans_cls not in persons_see[person]:
                        persons_see[person][trans_cls] = 1
                        persons_value[person][trans_cls] = (fix_im[seg_result == int(cls)]).sum() / 255
                    else:
                        persons_see[person][trans_cls] += 1
                        persons_value[person][trans_cls] += (fix_im[seg_result == int(cls)]).sum() / 255

            final_value = {}
            for k in persons_see[person]:
                times = persons_see[person][k]

                final_value[k] = float(persons_value[person][k] / times)
            persons_value[person] = final_value

        persons_value, self.values = self.split_ranking_dep(persons_value)
        item_maxx = -1
        for p in persons_value:
            mx = np.array([int(k) for k in persons_value[p].keys()]).max()
            item_maxx = max(item_maxx, mx)
        print(item_maxx)
        gd = np.zeros([person_num, item_maxx+1])
        for p in persons_value:
            for c in persons_value[p]:
                gd[int(p), int(c)] = persons_value[p][c]
        print(gd.shape)
        np.savetxt('gd.txt', gd)

    def get_person_choose(self, clu_cs, group=10, person_num=30, filter_times=10, chose_group=3):

        clu_ls = init_list_of_objects(group)
        for seg in self.seg_files:
            if not seg.endswith(self.seg_suffix):
                continue
            name = seg.split(self.seg_suffix)[0]
            cls = int(clu_cs[name])
            clu_ls[cls].append(name)
        persons_see = {}
        persons_value = {}
        for person in range(person_num):
            # select_id = person % len(clu_ls)
            person = str(person)
            person_image = []
            for tt in range(chose_group):
                select_id = np.random.randint(len(clu_ls))
                person_image += clu_ls[select_id]
            np.random.shuffle(person_image)
            person_image = person_image[:len(person_image)//2]
            persons_see[person] = {}
            persons_value[person] = {}
            for im in tqdm(person_image):
                seg_result = self.pre_load_seg[im+self.seg_suffix]
                # fix_file = os.path.join(fix, 'Sub_' + str(person + 1), im+self.fix_suffix)
                fix_im = self.pre_load_fix['User' + str(int(person) + 1)][im+self.fix_suffix]
                if not fix_im.shape == seg_result.shape:
                    print(im)
                    continue
                cls_set = set(list(seg_result.reshape(-1)))
                tmp = []
                for cc in cls_set:
                    if cc in self.filter:
                        tmp.append(cc)
                    # else:
                    #     fix_im[seg_result == int(cc)] = 0 # 不在筛选的类里，直接全部删除
                cls_set = tmp
                # sm = fix_im.sum()
                for cls in cls_set:  # get number of each person's scene
                    trans_cls = str(self.filter[cls])
                    cls = str(cls)

                    if trans_cls not in persons_see[person]:
                        persons_see[person][trans_cls] = 1
                        persons_value[person][trans_cls] = (fix_im[seg_result == int(cls)]).sum() / 255
                    else:
                        persons_see[person][trans_cls] += 1
                        persons_value[person][trans_cls] += (fix_im[seg_result == int(cls)]).sum() / 255

            final_value = {}
            for k in persons_see[person]:
                times = persons_see[person][k]

                if times < filter_times:
                    continue
                final_value[k] = float(persons_value[person][k] / times)
            persons_value[person] = final_value
        persons_value, _ = self.split_ranking_dep(persons_value)
        return persons_value

    def split_ranking_dep(self, persons_value, values=None):
        if values is None:
            for p in persons_value:
                scores = []
                for c in persons_value[p]:
                    scores.append(persons_value[p][c])
                scores = np.array(scores)
                scores = np.sort(scores)
                arr = np.array_split(scores, 5)
                values = np.array([v[0] for v in arr])

                for s, c in enumerate(persons_value[p]):
                    try:
                        judge = list(values > persons_value[p][c]).index(1)
                    except Exception:
                        judge = 5
                    persons_value[p][c] = judge
        # for p in persons_value:
        #     for c in persons_value[p]:
        #         try:
        #             judge = list(values>persons_value[p][c]).index(1)
        #         except Exception:
        #             judge = 5
        #         persons_value[p][c] = judge
        return persons_value, values

    def split_ranking(self, persons_value, values=None):
        if values is None:
            scores = []
            for p in persons_value:
                for c in persons_value[p]:
                    scores.append(persons_value[p][c])
            scores = np.array(scores)
            scores = np.sort(scores)
            arr = np.array_split(scores, 5)
            values = np.array([v[0] for v in arr])
        for p in persons_value:
            for c in persons_value[p]:
                try:
                    judge = list(values>persons_value[p][c]).index(1)
                except Exception:
                    judge = 5
                persons_value[p][c] = judge
        return persons_value, values

    def seg2Hist(self, n_clusters=5):
        from sklearn.cluster import KMeans
        cls = 150
        hots = []
        person_image = self.seg_files
        for im in tqdm(person_image):
            hot = np.zeros(150)
            im = im.replace(self.seg_suffix, '')
            seg_result = self.pre_load_seg[im + self.seg_suffix]
            for i in range(cls):
                hot[i] = (seg_result==i).sum()
            hots.append(hot)
        hots = np.array(hots)
        Zmax, Zmin = hots.max(axis=0), hots.min(axis=0)

        hots = (hots - Zmin) / (Zmax - Zmin + 1e-6)
        # np.savetxt('hots.txt', hots)
        estimator = KMeans(n_clusters=n_clusters, max_iter=100, tol=0.001)
        # 实现聚类结果
        estimator.fit(hots)
        label = estimator.labels_
        strr = ''
        for s, i in enumerate(person_image):
            strr += 'seg\\'+i+'\t'+str(label[s])+'\n'
        f = open('clu_label.txt', 'w')
        f.write(strr)
        f.close()


class ConvertFix2Ranking():
    def __init__(self, person_choose):
        self.person_choose = person_choose

    def convert2Rating(self):
        colunm = ['userId', 'objectId', 'rating', 'timestamp']
        df = pd.DataFrame()
        # for i in colunm:
        #     df[i] = data[i]
        data = {'userId':[], 'objectId':[], 'rating':[], 'timestamp':[]}

        for p in self.person_choose:
            value = person_choose[p]

            for cls in value:
                data['userId'].append(int(p))
                data['objectId'].append(int(cls))
                data['rating'].append(float(value[cls]))

        data['timestamp'] = data['rating'].copy()
        for i in colunm:
            df[i] = data[i]
        df.reset_index(drop=True, inplace=True)
        df.to_csv('my_rating.csv', index=False)


n_clusters = 5
chose_group = 3

c = PersonSplit(save, raw_image, fix)
c.seg2Hist(n_clusters=n_clusters)
c.get_person_all()

clu_cs, group = PersonSplit.getCluRes()
person_choose = c.get_person_choose(clu_cs, group, chose_group=chose_group)
# print(person_choose)

#
# # save
save_name = 'temp.json'
json_str = json.dumps(person_choose)
with open(save_name, 'w') as json_file:
    json_file.write(json_str)

##########################################
with open(save_name,'r') as load_f:
    person_choose = json.load(load_f)

convert = ConvertFix2Ranking(person_choose)
convert.convert2Rating()
pass
