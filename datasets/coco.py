# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
if __name__=="__main__":
    # for debug only
    import os, sys
    sys.path.append(os.path.dirname(sys.path[0]))

import copy 
import json
from pathlib import Path
import random
import numpy as np
import os

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

from datasets.data_util import preparing_dataset
import datasets.transforms as T
from util.box_ops import box_cxcywh_to_xyxy, box_iou

from torchvision.transforms import functional as F

__all__ = ['build']



coco_category_map_str = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11, "13": 12, "14": 13, "15": 14, "16": 15, "17": 16, "18": 17, "19": 18, "20": 19, "21": 20, "22": 21, "23": 22, "24": 23, "25": 24, "27": 25, "28": 26, "31": 27, "32": 28, "33": 29, "34": 30, "35": 31, "36": 32, "37": 33, "38": 34, "39": 35, "40": 36, "41": 37, "42": 38, "43": 39, "44": 40, "46": 41, "47": 42, "48": 43, "49": 44, "50": 45, "51": 46, "52": 47, "53": 48, "54": 49, "55": 50, "56": 51, "57": 52, "58": 53, "59": 54, "60": 55, "61": 56, "62": 57, "63": 58, "64": 59, "65": 60, "67": 61, "70": 62, "72": 63, "73": 64, "74": 65, "75": 66, "76": 67, "77": 68, "78": 69, "79": 70, "80": 71, "81": 72, "82": 73, "84": 74, "85": 75, "86": 76, "87": 77, "88": 78, "89": 79, "90": 80}

coco_names = {'1': 'person', '2': 'bicycle', '3': 'car', '4': 'motorcycle', '5': 'airplane', '6': 'bus', '7': 'train', '8': 'truck', '9': 'boat', '10': 'traffic light', '11': 'fire hydrant', '13': 'stop sign', '14': 'parking meter', '15': 'bench', '16': 'bird', '17': 'cat', '18': 'dog', '19': 'horse', '20': 'sheep', '21': 'cow', '22': 'elephant', '23': 'bear', '24': 'zebra', '25': 'giraffe', '27': 'backpack', '28': 'umbrella', '31': 'handbag', '32': 'tie', '33': 'suitcase', '34': 'frisbee', '35': 'skis', '36': 'snowboard', '37': 'sports ball', '38': 'kite', '39': 'baseball bat', '40': 'baseball glove', '41': 'skateboard', '42': 'surfboard', '43': 'tennis racket', '44': 'bottle', '46': 'wine glass', '47': 'cup', '48': 'fork', '49': 'knife', '50': 'spoon', '51': 'bowl', '52': 'banana', '53': 'apple', '54': 'sandwich', '55': 'orange', '56': 'broccoli', '57': 'carrot', '58': 'hot dog', '59': 'pizza', '60': 'donut', '61': 'cake', '62': 'chair', '63': 'couch', '64': 'potted plant', '65': 'bed', '67': 'dining table', '70': 'toilet', '72': 'tv', '73': 'laptop', '74': 'mouse', '75': 'remote', '76': 'keyboard', '77': 'cell phone', '78': 'microwave', '79': 'oven', '80': 'toaster', '81': 'sink', '82': 'refrigerator', '84': 'book', '85': 'clock', '86': 'vase', '87': 'scissors', '88': 'teddy bear', '89': 'hair drier', '90': 'toothbrush'}

def preprocess_caption(caption: str) -> str: 
    result = caption.lower().strip() 
    if result.endswith("."):    
        return result
    return result + "." 


class label2compat():
    def __init__(self) -> None:
        self.category_map_str = coco_category_map_str 
        self.category_map = {int(k):v for k,v in self.category_map_str.items()}

    def __call__(self, target, img=None):
        labels = target['labels']
        res = torch.zeros(labels.shape, dtype=labels.dtype)
        for idx, item in enumerate(labels):
            res[idx] = self.category_map[item.item()] - 1
        target['label_compat'] = res
        if img is not None:
            return target, img
        else:
            return target


class label_compat2onehot():
    def __init__(self, num_class=80, num_output_objs=1):
        self.num_class = num_class
        self.num_output_objs = num_output_objs
        if num_output_objs != 1:
            raise DeprecationWarning("num_output_objs!=1, which is only used for comparison")

    def __call__(self, target, img=None):
        labels = target['label_compat']
        place_dict = {k:0 for k in range(self.num_class)}
        if self.num_output_objs == 1:
            res = torch.zeros(self.num_class)
            for i in labels:
                itm = i.item()
                res[itm] = 1.0
        else:
            # compat with baseline
            res = torch.zeros(self.num_class, self.num_output_objs)
            for i in labels:
                itm = i.item()
                res[itm][place_dict[itm]] = 1.0
                place_dict[itm] += 1
        target['label_compat_onehot'] = res
        if img is not None:
            return target, img
        else:
            return target


class box_label_catter():
    def __init__(self):
        pass

    def __call__(self, target, img=None):
        labels = target['label_compat']
        boxes = target['boxes']
        box_label = torch.cat((boxes, labels.unsqueeze(-1)), 1)
        target['box_label'] = box_label
        if img is not None:
            return target, img
        else:
            return target


class RandomSelectBoxlabels():
    def __init__(self, num_classes, leave_one_out=False, blank_prob=0.8,
                    prob_first_item = 0.0,
                    prob_random_item = 0.0,
                    prob_last_item = 0.8,
                    prob_stop_sign = 0.2
                ) -> None:
        self.num_classes = num_classes
        self.leave_one_out = leave_one_out
        self.blank_prob = blank_prob

        self.set_state(prob_first_item, prob_random_item, prob_last_item, prob_stop_sign)
        

    def get_state(self):
        return [self.prob_first_item, self.prob_random_item, self.prob_last_item, self.prob_stop_sign]

    def set_state(self, prob_first_item, prob_random_item, prob_last_item, prob_stop_sign):
        sum_prob = prob_first_item + prob_random_item + prob_last_item + prob_stop_sign
        assert sum_prob - 1 < 1e-6, \
            f"Sum up all prob = {sum_prob}. prob_first_item:{prob_first_item}" \
            + f"prob_random_item:{prob_random_item}, prob_last_item:{prob_last_item}" \
            + f"prob_stop_sign:{prob_stop_sign}"

        self.prob_first_item = prob_first_item
        self.prob_random_item = prob_random_item
        self.prob_last_item = prob_last_item
        self.prob_stop_sign = prob_stop_sign
        

    def sample_for_pred_first_item(self, box_label: torch.FloatTensor):
        box_label_known = torch.Tensor(0,5)
        box_label_unknown = box_label
        return box_label_known, box_label_unknown

    def sample_for_pred_random_item(self, box_label: torch.FloatTensor):
        n_select = int(random.random() * box_label.shape[0])
        box_label = box_label[torch.randperm(box_label.shape[0])]
        box_label_known = box_label[:n_select]
        box_label_unknown = box_label[n_select:]
        return box_label_known, box_label_unknown

    def sample_for_pred_last_item(self, box_label: torch.FloatTensor):
        box_label_perm = box_label[torch.randperm(box_label.shape[0])]
        known_label_list = []
        box_label_known = []
        box_label_unknown = []
        for item in box_label_perm:
            label_i = item[4].item()
            if label_i in known_label_list:
                box_label_known.append(item)
            else:
                # first item
                box_label_unknown.append(item)
                known_label_list.append(label_i)
        box_label_known = torch.stack(box_label_known) if len(box_label_known) > 0 else torch.Tensor(0,5)
        box_label_unknown = torch.stack(box_label_unknown) if len(box_label_unknown) > 0 else torch.Tensor(0,5)
        return box_label_known, box_label_unknown

    def sample_for_pred_stop_sign(self, box_label: torch.FloatTensor):
        box_label_unknown = torch.Tensor(0,5)
        box_label_known = box_label
        return box_label_known, box_label_unknown

    def __call__(self, target, img=None):
        box_label = target['box_label'] # K, 5

        dice_number = random.random()

        if dice_number < self.prob_first_item:
            box_label_known, box_label_unknown = self.sample_for_pred_first_item(box_label)
        elif dice_number < self.prob_first_item + self.prob_random_item:
            box_label_known, box_label_unknown = self.sample_for_pred_random_item(box_label)
        elif dice_number < self.prob_first_item + self.prob_random_item + self.prob_last_item:
            box_label_known, box_label_unknown = self.sample_for_pred_last_item(box_label)
        else:
            box_label_known, box_label_unknown = self.sample_for_pred_stop_sign(box_label)

        target['label_onehot_known'] = label2onehot(box_label_known[:,-1], self.num_classes)
        target['label_onehot_unknown'] = label2onehot(box_label_unknown[:, -1], self.num_classes)
        target['box_label_known'] = box_label_known
        target['box_label_unknown'] = box_label_unknown

        return target, img


class RandomDrop():
    def __init__(self, p=0.2) -> None:
        self.p = p

    def __call__(self, target, img=None):
        known_box = target['box_label_known']
        num_known_box = known_box.size(0)
        idxs = torch.rand(num_known_box)
        # indices = torch.randperm(num_known_box)[:int((1-self).p*num_known_box + 0.5 + random.random())]
        target['box_label_known'] = known_box[idxs > self.p]
        return target, img


class BboxPertuber():
    def __init__(self, max_ratio = 0.02, generate_samples = 1000) -> None:
        self.max_ratio = max_ratio
        self.generate_samples = generate_samples
        self.samples = self.generate_pertube_samples()
        self.idx = 0

    def generate_pertube_samples(self):
        import torch
        samples = (torch.rand(self.generate_samples, 5) - 0.5) * 2 * self.max_ratio
        return samples

    def __call__(self, target, img):
        known_box = target['box_label_known'] # Tensor(K,5), K known bbox
        K = known_box.shape[0]
        known_box_pertube = torch.zeros(K, 6) # 4:bbox, 1:prob, 1:label
        if K == 0:
            pass
        else:
            if self.idx + K > self.generate_samples:
                self.idx = 0
            delta = self.samples[self.idx: self.idx + K, :]
            known_box_pertube[:, :4] = known_box[:, :4] + delta[:, :4]
            iou = (torch.diag(box_iou(box_cxcywh_to_xyxy(known_box[:, :4]), box_cxcywh_to_xyxy(known_box_pertube[:, :4]))[0])) * (1 + delta[:, -1])
            known_box_pertube[:, 4].copy_(iou)
            known_box_pertube[:, -1].copy_(known_box[:, -1])

        target['box_label_known_pertube'] = known_box_pertube
        return target, img


class RandomCutout():
    def __init__(self, factor=0.5) -> None:
        self.factor = factor

    def __call__(self, target, img=None):
        unknown_box = target['box_label_unknown']           # Ku, 5
        known_box = target['box_label_known_pertube']       # Kk, 6
        Ku = unknown_box.size(0)

        known_box_add = torch.zeros(Ku, 6) # Ku, 6
        known_box_add[:, :5] = unknown_box
        known_box_add[:, 5].uniform_(0.5, 1) 
        

        known_box_add[:, :2] += known_box_add[:, 2:4] * (torch.rand(Ku, 2) - 0.5) / 2
        known_box_add[:, 2:4] /= 2

        target['box_label_known_pertube'] = torch.cat((known_box, known_box_add))
        return target, img


class RandomSelectBoxes():
    def __init__(self, num_class=80) -> None:
        Warning("This is such a slow function and will be deprecated soon!!!")
        self.num_class = num_class

    def __call__(self, target, img=None):
        boxes = target['boxes']
        labels = target['label_compat']

        # transform to list of tensors
        boxs_list = [[] for i in range(self.num_class)]
        for idx, item in enumerate(boxes):
            label = labels[idx].item()
            boxs_list[label].append(item)
        boxs_list_tensor = [torch.stack(i) if len(i) > 0 else torch.Tensor(0,4) for i in boxs_list]

        # random selection
        box_known = []
        box_unknown = []
        for idx, item in enumerate(boxs_list_tensor):
            ncnt = item.shape[0]
            nselect = int(random.random() * ncnt) # close in both sides, much faster than random.randint

            item = item[torch.randperm(ncnt)]
            # random.shuffle(item)
            box_known.append(item[:nselect])
            box_unknown.append(item[nselect:])

        # box_known_tensor = [torch.stack(i) if len(i) > 0 else torch.Tensor(0,4) for i in box_known]
        # box_unknown_tensor = [torch.stack(i) if len(i) > 0 else torch.Tensor(0,4) for i in box_unknown]
        # print('box_unknown_tensor:', box_unknown_tensor)
        target['known_box'] = box_known
        target['unknown_box'] = box_unknown
        return target, img


def label2onehot(label, num_classes):
    """
    label: Tensor(K)
    """
    res = torch.zeros(num_classes)
    for i in label:
        itm = int(i.item())
        res[itm] = 1.0
    return res


class MaskCrop():
    def __init__(self) -> None:
        pass

    def __call__(self, target, img):
        known_box = target['known_box']
        h,w = img.shape[1:] # h,w
        # imgsize = target['orig_size'] # h,w

        scale = torch.Tensor([w, h, w, h])

        # _cnt = 0
        for boxes in known_box:
            if boxes.shape[0] == 0:
                continue
            box_xyxy = box_cxcywh_to_xyxy(boxes) * scale
            for box in box_xyxy:
                x1, y1, x2, y2 = [int(i) for i in box.tolist()]
                img[:, y1:y2, x1:x2] = 0
                # _cnt += 1
        # print("_cnt:", _cnt)
        return target, img


dataset_hook_register = {
    'label2compat': label2compat,
    'label_compat2onehot': label_compat2onehot,
    'box_label_catter': box_label_catter,
    'RandomSelectBoxlabels': RandomSelectBoxlabels,
    'RandomSelectBoxes': RandomSelectBoxes,
    'MaskCrop': MaskCrop,
    'BboxPertuber': BboxPertuber,
}


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, aux_target_hacks=None,
                 caption_file=None, use_text_labels=False, do_text_shuffle=True,
                 nouns_list=None, relations_list=None, rln_pretraining=False,
                 gpt4sgg_file=None, name2predicates=None, 
                 coco_ids_in_vg_file=None):

        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.aux_target_hacks = aux_target_hacks
        self.rln_pretraining = rln_pretraining

        self.ind_to_classes = ['__background__'] + [v['name'] for k, v in self.coco.cats.items()]

        self.caption_data = None 
        if caption_file is not None and gpt4sgg_file is None:
            with open(caption_file, 'r') as fin:
                caption_data = json.load(fin)
            
            def is_valid(x):
                res = False
                for rel in x:
                    if len(rel) > 0:
                        res = True

                return res

            caption_data = [e for e in caption_data if is_valid(e['relations'])]
            self.caption_data = caption_data

            self.caption_data = sorted(self.caption_data, key=lambda x: x['image_id'])
            self.ids = [e['image_id'] for e in self.caption_data]

        # gpt4sgg
        self.gpt4sgg_data = None 
        if gpt4sgg_file is not None:
            with open(gpt4sgg_file, 'r') as fin:
                gpt4sgg_data = json.load(fin)
            self.gpt4sgg_data = {str(e['image_id']): e for e in gpt4sgg_data}
            self.ids = [e for e in self.ids if str(e) in self.gpt4sgg_data]
            self.name2predicates = name2predicates

        #if self.rln_pretraining:
            #coco_ids_in_vg_test = torch.load(coco_ids_in_vg_file)
            #coco_ids_in_vg_test = set([str(e) for e in coco_ids_in_vg_test])
            #print("Loading  %s coco_ids_in_vg_test from :%s" % (len(coco_ids_in_vg_test), coco_ids_in_vg_file))
            #self.ids = [e for e in self.ids if str(e) not in coco_ids_in_vg_test]

        # add text
        self.use_text_labels = use_text_labels
        self.do_text_shuffle = do_text_shuffle
        self.nouns_list = nouns_list
        self.relations_list = relations_list

        if self.use_text_labels:
            self.name2classes =  {v: int(k) for k, v in coco_names.items()}
            self.class2name = {v : k for k, v in self.name2classes.items() }

            assert self.nouns_list is not None, "nouns list cannot be None!"
            assert self.relations_list is not None, "relations list cannot be None !"


    def change_hack_attr(self, hackclassname, attrkv_dict):
        target_class = dataset_hook_register[hackclassname]
        for item in self.aux_target_hacks:
            if isinstance(item, target_class):
                for k,v in attrkv_dict.items():
                    setattr(item, k, v)

    def get_hack(self, hackclassname):
        target_class = dataset_hook_register[hackclassname]
        for item in self.aux_target_hacks:
            if isinstance(item, target_class):
                return item

    def __getitem__(self, idx):
        """
        Output:
            - target: dict of multiple items
                - boxes: Tensor[num_box, 4]. \
                    Init type: x0,y0,x1,y1. unnormalized data.
                    Final type: cx,cy,w,h. normalized data. 
        """
        if idx >= len(self):
            raise IndexError
        try:
            img, target = super(CocoDetection, self).__getitem__(idx)
        except:
            raise Exception("Fail to load:{}".format(idx))
            #print("Error idx: {}".format(idx))
            #idx += 1
            #img, target = super(CocoDetection, self).__getitem__(idx)

        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        # text 
        if self.use_text_labels:
            if self.gpt4sgg_data is not None:
                gpt4sgg_data = self.gpt4sgg_data[str(image_id)]
                del target['boxes']
                del target['labels']
                del target['area']
                del target['iscrowd']

                target['boxes'] = torch.as_tensor(gpt4sgg_data['bboxes'])
                names = [e.split('.')[0] for e in gpt4sgg_data['names']]
                target['gt_names'] = copy.deepcopy(names)
                target['labels'] = torch.tensor([self.name2classes[e] for e in names])
                
                neg_names = list(set(self.nouns_list) - set(names))
                neg_sample = []
                neg_count = max(1, 80 - len(set(names)))
                neg_sample = np.random.choice(neg_names, size=neg_count, replace=False).tolist()

                nouns = target['gt_names'] + neg_sample
                random.shuffle(nouns)

                target['caption'] = '. '.join(nouns)
                target['caption'] = preprocess_caption(target['caption'])
                 
                relations = []
                org_names = gpt4sgg_data['names']
                for rel in gpt4sgg_data['relationships']:
                    sub = org_names.index(rel['source'])
                    obj = org_names.index(rel['target'])
                    pred = rel['relation']
                    relations.append([sub, obj, pred])

                gt_rels = list(set([e[2] for e in relations]))
                neg_rels = list(set(self.relations_list) - set(gt_rels))
                neg_rel_sample = []

                neg_count = max(1, 80 - len(gt_rels))
                #neg_count = min(len(rels)*3, neg_count)
                neg_rel_sample = np.random.choice(neg_rels, size=neg_count, replace=False).tolist()
                target['gt_rels'] = gt_rels
             
                rels = gt_rels + neg_rel_sample
                random.shuffle(rels)
                rel_caption = '. '.join(rels) + '.'
                target['rel_caption'] = rel_caption
                target['edges'] = relations
            else:
                cap_data = self.caption_data[idx]
                assert cap_data['image_id'] == image_id, "image_id does not match"
                
                target['relations'] = []
                boxes = []
                _try_cnt = 0
                while len(target['relations']) == 0 or len(boxes) == 0:
                    ridx = random.randint(0, len(cap_data['relations'])-1)
                    target['relations'] = cap_data['relations'][ridx]
                    boxes = cap_data['boxes'][ridx]

                    random.shuffle(target['relations'])
                    _try_cnt += 1
                    if _try_cnt > 100:
                        break
                if _try_cnt > 100:
                    return self[idx - random.randint(1, 4)]
                
                nouns = []
                rels = []
                for rel in target['relations']:
                    nouns.append(rel[0])
                    nouns.append(rel[1])
                    rels.append(rel[2])

                nouns = list(set(nouns))
                rels = list(set(rels))
                gt_sample = nouns 
                target['gt_names'] = copy.deepcopy(nouns) #gt_sample

                del target['boxes']
                del target['labels']
                del target['area']
                del target['iscrowd']
                if 'boxes' in cap_data:
                    h, w = target['size']
                    scale_fct = torch.tensor([w, h, w, h], dtype=torch.float)
                    target['boxes'] = torch.tensor(cap_data['boxes'][ridx], dtype=torch.float) * scale_fct

                    target['gt_names'] = cap_data['phrases'][ridx]
                    gt_sample = target['gt_names']

                neg_names = list(set(self.nouns_list) - set(nouns))
                neg_sample = []
                neg_count = max(1, 80 - len(gt_sample))
                #neg_count = min(3*len(gt_sample), neg_count)
                neg_sample = np.random.choice(neg_names, size=neg_count, replace=False).tolist()
                
                nouns = gt_sample + neg_sample
                if os.environ.get("DEBUG") == '1':
                    nouns = gt_sample
                nouns = list(set(nouns))
                # shuffle 
                if True:
                    random.shuffle(nouns)

                target['caption'] = '. '.join(nouns)
                target['caption'] = preprocess_caption(target['caption'])

                # rel
                neg_rels = list(set(self.relations_list) - set(rels))
                neg_rel_sample = []

                neg_count = max(1, 80 - len(rels))
                #neg_count = min(len(rels)*3, neg_count)
                neg_rel_sample = np.random.choice(neg_rels, size=neg_count, replace=False).tolist()

                target['gt_rels'] = copy.deepcopy(rels)
                #rels = ['[UNK]'] + rels + neg_rel_sample
                rels = rels + neg_rel_sample
                if True:
                    random.shuffle(rels)
                rel_caption = '. '.join(rels) + '.'
                target['rel_caption'] = rel_caption



        if self._transforms is not None:
            img, target = self._transforms(img, target)
        # convert to needed format
        if self.aux_target_hacks is not None:
            for hack_runner in self.aux_target_hacks:
                target, img = hack_runner(target, img=img)

        return img, target

    def get_groundtruth(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        relation = None 


        return target['boxes'], target['labels'], relation



def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        #image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, fix_size=False, 
                         strong_aug=False, args=None):
    data_mean = getattr(args, "data_mean", [0.485, 0.456, 0.406])
    data_std = getattr(args, "data_std", [0.229, 0.224, 0.225])
    do_crop = getattr(args, "do_crop", True)

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(data_mean, data_std)
    ])

    # config the params for data aug
    max_size = 1333
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]


    scales2_resize = [400, 500, 600]
    scales2_crop = [384, 600]
    
    # update args from config files
    scales = getattr(args, 'data_aug_scales', scales)
    max_size = getattr(args, 'data_aug_max_size', max_size)
    scales2_resize = getattr(args, 'data_aug_scales2_resize', scales2_resize)
    scales2_crop = getattr(args, 'data_aug_scales2_crop', scales2_crop)

    # resize them
    data_aug_scale_overlap = getattr(args, 'data_aug_scale_overlap', None)
    if data_aug_scale_overlap is not None and data_aug_scale_overlap > 0:
        data_aug_scale_overlap = float(data_aug_scale_overlap)
        scales = [int(i*data_aug_scale_overlap) for i in scales]
        max_size = int(max_size*data_aug_scale_overlap)
        scales2_resize = [int(i*data_aug_scale_overlap) for i in scales2_resize]
        scales2_crop = [int(i*data_aug_scale_overlap) for i in scales2_crop]

    datadict_for_print = {
        'scales': scales,
        'max_size': max_size,
        'scales2_resize': scales2_resize,
        'scales2_crop': scales2_crop
    }
    print("data_aug_params:", json.dumps(datadict_for_print, indent=2))
        

    if image_set == 'train':
        if fix_size:
            return T.Compose([
                #T.RandomHorizontalFlip(),
                #T.RandomResize([(max_size, max(scales))]),
                T.RandomResize([max(scales)], max_size=max_size),
                normalize,
            ])

        if strong_aug:
            import datasets.sltransform as SLT
            
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose([
                        T.RandomResize(scales2_resize),
                        T.RandomSizeCrop(*scales2_crop),
                        T.RandomResize(scales, max_size=max_size),
                    ])
                ),
                SLT.RandomSelectMulti([
                    SLT.RandomCrop(),
                    SLT.LightingNoise(),
                    SLT.AdjustBrightness(2),
                    SLT.AdjustContrast(2),
                ]),
                normalize,
            ])
        if do_crop: 
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose([
                        T.RandomResize(scales2_resize),
                        T.RandomSizeCrop(*scales2_crop),
                        T.RandomResize(scales, max_size=max_size),
                    ])
                ),
                normalize,
            ])
        else:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomResize(scales, max_size=max_size), 
                normalize,
            ])


    if image_set in ['val', 'eval_debug', 'train_reg', 'test']:
        if os.environ.get("GFLOPS_DEBUG_SHILONG", False) == 'INFO':
            print("Under debug mode for flops calculation only!!!!!!!!!!!!!!!!")
            return T.Compose([
                T.ResizeDebug((1280, 800)),
                normalize,
            ])   

        return T.Compose([
            T.RandomResize([max(scales)], max_size=max_size),
            normalize,
        ])



    raise ValueError(f'unknown {image_set}')


def get_aux_target_hacks_list(image_set, args):
    if args.modelname in ['q2bs_mask', 'q2bs']:
        aux_target_hacks_list = [
            label2compat(), 
            label_compat2onehot(), 
            RandomSelectBoxes(num_class=args.num_classes)
        ]
        if args.masked_data and image_set == 'train':
            # aux_target_hacks_list.append()
            aux_target_hacks_list.append(MaskCrop())
    elif args.modelname in ['q2bm_v2', 'q2bs_ce', 'q2op', 'q2ofocal', 'q2opclip', 'q2ocqonly']:
        aux_target_hacks_list = [
            label2compat(),
            label_compat2onehot(),
            box_label_catter(),
            RandomSelectBoxlabels(num_classes=args.num_classes,
                                    prob_first_item=args.prob_first_item,
                                    prob_random_item=args.prob_random_item,
                                    prob_last_item=args.prob_last_item,
                                    prob_stop_sign=args.prob_stop_sign,
                                    ),
            BboxPertuber(max_ratio=0.02, generate_samples=1000),
        ]
    elif args.modelname in ['q2omask', 'q2osa']:
        if args.coco_aug:
            aux_target_hacks_list = [
                label2compat(),
                label_compat2onehot(),
                box_label_catter(),
                RandomSelectBoxlabels(num_classes=args.num_classes,
                                        prob_first_item=args.prob_first_item,
                                        prob_random_item=args.prob_random_item,
                                        prob_last_item=args.prob_last_item,
                                        prob_stop_sign=args.prob_stop_sign,
                                        ),
                RandomDrop(p=0.2),
                BboxPertuber(max_ratio=0.02, generate_samples=1000),
                RandomCutout(factor=0.5)
            ]
        else:
            aux_target_hacks_list = [
                label2compat(),
                label_compat2onehot(),
                box_label_catter(),
                RandomSelectBoxlabels(num_classes=args.num_classes,
                                        prob_first_item=args.prob_first_item,
                                        prob_random_item=args.prob_random_item,
                                        prob_last_item=args.prob_last_item,
                                        prob_stop_sign=args.prob_stop_sign,
                                        ),
                BboxPertuber(max_ratio=0.02, generate_samples=1000),
            ]
    else:
        aux_target_hacks_list = None

    return aux_target_hacks_list


def build(image_set, args):
    data_path = os.path.join(args.data_path, "coco")

    root = Path(data_path)
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "train_reg": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
        "eval_debug": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
        "test": (root / "test2017", root / "annotations" / 'image_info_test-dev2017.json' ),
    }

    # add some hooks to datasets
    aux_target_hacks_list = get_aux_target_hacks_list(image_set, args)
    img_folder, ann_file = PATHS[image_set]

    # copy to local path
    if os.environ.get('DATA_COPY_SHILONG') == 'INFO':
        preparing_dataset(dict(img_folder=img_folder, ann_file=ann_file), image_set, args)

    strong_aug = getattr(args, "strong_aug", False)

    try:
        caption_file = os.path.join(str(root), 'annotations', f'coco_{image_set}2017_triple.json')
        assert os.path.exists(caption_file), "caption_file:%s does not exist!" % caption_file
    except:
        caption_file = None

    gpt4sgg_file = getattr(args, "gpt4sgg_file", None)
    use_text_labels = getattr(args, "use_text_labels", False)

    nouns_list, relations_list = None, None
    name2predicates = None
    if use_text_labels:
        if gpt4sgg_file is not None:
            nouns_list = [line.split(',')[0] for line in \
                    open(os.path.join(str(root), 'annotations', 'coco_nouns_gpt.txt'), 'r')]
            relations_list = [line.split(',')[0] for line in \
                    open(os.path.join(str(root), 'annotations', 'coco_relations_gpt.txt'), 'r')]
            name2predicates = {name: idx+1 for idx, name in enumerate(relations_list)}
        else:
            nouns_list = [line.split(',')[0] for line in \
                    open(os.path.join(str(root), 'annotations', 'coco_nouns.txt'), 'r')]
            relations_list = [line.split(',')[0] for line in \
                    open(os.path.join(str(root), 'annotations', 'coco_relations.txt'), 'r')]

    dataset = CocoDetection(img_folder, ann_file, 
            transforms=make_coco_transforms(image_set, 
                            fix_size=getattr(args, "fix_size", False), 
                            strong_aug=strong_aug, args=args), 
            return_masks=args.masks,
            aux_target_hacks=aux_target_hacks_list,
            caption_file=caption_file,
            use_text_labels=use_text_labels,
            nouns_list=nouns_list,
            relations_list=relations_list,
            rln_pretraining=getattr(args, "rln_pretraining", False),
            gpt4sgg_file=gpt4sgg_file,
            name2predicates=name2predicates,
            coco_ids_in_vg_file=getattr(args, "coco_ids_in_vg_file", 
                os.path.join(str(root), "annotations", "coco_ids_in_vg_test.pth"))
        )

    return dataset



if __name__ == "__main__":
    from tqdm import tqdm

    import pdb; pdb.set_trace()
    image_set = "train"
    root = './data/'
    nouns_list = [line.rstrip().split(',')[0] for line in \
            open(os.path.join(str(root), 'coco/annotations', 'coco_nouns_gpt.txt'), 'r')]
    relations_list = [line.rstrip().split(',')[0] for line in \
                open(os.path.join(str(root), 'coco/annotations', 'coco_relations_gpt.txt'), 'r')]
    dataset = CocoDetection("./data/coco/train2017", 
            "./data/coco/annotations/instances_train2017.json", 
            #caption_file="./data/coco/annotations/coco_train2017_triple.json",
            transforms=make_coco_transforms(image_set, fix_size=False), 
            return_masks=False,
            use_text_labels=True,
            nouns_list=nouns_list,
            relations_list = relations_list,
            rln_pretraining=True,
            gpt4sgg_file='./instruction/coco-gpt/gpt-aligned.json',
            coco_ids_in_vg_file= os.path.join("data/coco/", "annotations", "coco_ids_in_vg_test.pth")
        )

    from torch.utils.data import DataLoader
    from util.misc import collate_fn

    loader = DataLoader(dataset, num_workers=0, batch_size=8, collate_fn=collate_fn)

    for images, item in tqdm(loader):
        import pdb; pdb.set_trace()



