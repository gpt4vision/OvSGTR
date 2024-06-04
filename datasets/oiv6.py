# OpenImages (v6) dataset, ref.: https://github.com/open-mmlab/mmdetection/blob/main/mmdet/datasets/openimages.py
# @author: Joseph Chen
# @email: zychen.uestc@gmail.com
# @date:  May 5, 2023 
# 
import json
import logging
import os
import copy
import pickle
import random
from collections import defaultdict, OrderedDict, Counter
from typing import Dict, List
import csv

from PIL import ImageFile 
#ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from util.bounding_box import BoxList
from pycocotools.coco import COCO

from datasets.coco import make_coco_transforms


def preprocess_caption(caption: str) -> str: 
    result = caption.lower().strip() 
    if result.endswith("."):    
        return result
    return result + "." 


def read_json(name):
    if 'jsonl' in name:
        with open(name, 'r') as fin:
            data = [json.loads(line) for line in fin]
    else:
        with open(name, 'r') as fin:
            data = json.load(fin)

    return data

"""
Args:
   img_dir: path that contains jpeg files
   ann_file: jsonl file for bbox 
   label2cls_file: label id - name
   img_level_ann_file:
   hierarch_file: 
"""
class OICAPDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_file, 
                 label2cls_file, 
                 img_info_file,
                 transforms=None, config=None,
                 image_level_ann_file=None,
                 hierarchy_file=None,
                 use_text_labels=False,
                 do_text_shuffle=True,
                 nouns_list=None,
                 relations_list=None):
        super().__init__()
        self.img_dir = img_dir
        data = read_json(ann_file)

        self.images = data
        self.transforms = transforms
        if self.transforms is None:
            print("Warning : transforms is None ")

        with open(label2cls_file, 'rb') as fin:
            self.label2cls = pickle.load(fin)
        self.num_classes = len(self.label2cls.keys())

        self.do_text_shuffle = do_text_shuffle
        self.nouns_list = nouns_list
        self.relations_list = relations_list

        self.use_text_labels = use_text_labels and len(nouns_list) > 0 and len(relations_list) > 0

        if self.use_text_labels:
            self.name2classes = {v['name'].lower(): int(v['idx'])  for k, v in self.label2cls.items() if v['name'] != '__background__'}
            self.class2name = {v: k for k, v in self.name2classes.items()}

        self.img_info_file = img_info_file
        if self.img_info_file is not None:
            with open(img_info_file, 'rb') as fin:
                self.img_info = pickle.load(fin)
        else:
            self.img_info = None

        # 
        self.image_level_ann_file = image_level_ann_file
        if self.image_level_ann_file is not None:
            self.img_level_anns = self._parse_img_level_ann(
                self.image_level_ann_file)
        else:
            self.img_level_anns = None

        # OpenImagesMetric can get the relation matrix from the dataset meta
        self.hierarchy_file = hierarchy_file
        assert hierarchy_file is not None, "hierarchy_file should not be None!"
        # label2cls start from 1 instead of 0
        self.relation_matrix = self._get_relation_matrix(self.hierarchy_file, self.num_classes+1)

        self.ids = [item['image_id'] for item in self.images]
        self._coco = None 

        for image in self.images:
            image['id'] = image['image_id']

        #  label start from 1 to 601
        self.ind_to_classes = ['__background__'] + [v['name'] for k, v in self.label2cls.items()]
        self.categories = [{'supercategory': 'none', # not used?
                            'id': v['idx'], 
                            'name': v['name']
                            }  
                            for k, v in self.label2cls.items()
                                  if v['name'] != '__background__'
                            ]



    @property
    def coco(self):
        if self._coco is None:
            # for custom datasets
            _coco = COCO()
            coco_dicts = dict(
                            images=self.images, 
                            annotations=[],
                            categories=self.categories)
            
            for index, ann in enumerate(self.images):
                img_id = ann['image_id']
                assert self.img_info is not None, " img_info_file cannot be None!"
                ih, iw = self.img_info[img_id]['orig_size']

                scale_fct = torch.tensor([iw, ih, iw, ih])
                gt_boxes, gt_classes, relation = self.get_groundtruth(index)
                gt_boxes = gt_boxes * scale_fct
                gt_boxes   = gt_boxes.numpy()
                gt_classes = gt_classes.numpy()

                # add supercategories to gt. 
                is_group_ofs = np.array(self.images[index]['is_group_of'], dtype=bool)
                gt_labels_unique = np.unique(gt_classes)
                for label in gt_labels_unique:
                    super_labels = np.where(self.relation_matrix[label])[0]
                    for cls in super_labels:
                        if cls == label:
                            continue
                        idx = np.where(gt_classes == label)[0]
                        gt_boxes = np.concatenate((gt_boxes, gt_boxes[idx]))
                        extend_labels = np.full(idx.shape, cls, dtype=np.int64)
                        gt_classes = np.concatenate((gt_classes, extend_labels))
                        is_group_ofs = np.concatenate((is_group_ofs, is_group_ofs[idx]))


                for box, cls, is_group_of in zip(gt_boxes, gt_classes, is_group_ofs):
                    item = {
                            'area': (box[3] - box[1]) * (box[2] - box[0]),
                            'bbox': [box[0], box[1], box[2] - box[0], box[3] - box[1]], # xywh
                            'category_id': cls,
                            'image_id': ann['image_id'], 
                            'id': len(coco_dicts['annotations']),
                            'iscrowd': 0,
                            'is_group_of': is_group_of
                           }                    
                    coco_dicts['annotations'].append(item)

            _coco.dataset = coco_dicts
            _coco.createIndex()
            self._coco = _coco

        return self._coco

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        item = self.images[index]
        
        img_name = os.path.join(self.img_dir, '%s.jpg' % item['image_id'])
        image = Image.open(img_name).convert("RGB")
        assert image is not None, "image:%s is None!" % img_name

        iw, ih = image.size

        target = dict(image_id=item['image_id'])

        gt_boxes, gt_classes, relation = self.get_groundtruth(index)

        target["iscrowd"] = torch.zeros(gt_boxes.shape[0])
        target['labels'] = gt_classes
        target['boxes'] = gt_boxes * torch.tensor([iw, ih, iw, ih]).reshape(1, 4)
        target["orig_size"] = torch.tensor([ih, iw])
        target['is_occluded'] = item['is_occluded']
        target['is_truncated'] = item['is_truncated']
        target['is_group_of'] = item['is_group_of']
        target['is_depiction'] = item['is_depiction']
        target['is_inside'] = item['is_inside']
        if 'relations' in item:
            target['relations'] = item['relations']

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        # text
        if self.use_text_labels:
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

            neg_names = list(set(self.nouns_list) - set(nouns))
            neg_sample = []

            neg_count = max(1, 80 - len(gt_sample))
            neg_sample = np.random.choice(neg_names, size=neg_count, replace=False).tolist()

            nouns = gt_sample + neg_sample
            nouns = list(set(nouns))
            # shuffle 
            if True: #self.do_text_shuffle:
                random.shuffle(nouns)

            target['caption'] = '. '.join(nouns)
            target['caption'] = preprocess_caption(target['caption'])

            # rel
            neg_rels = list(set(self.relations_list) - set(rels))
            neg_rel_sample = []

            neg_count = max(1, 80 - len(rels))
            neg_rel_sample = np.random.choice(neg_rels, size=neg_count, replace=False).tolist()

            target['gt_rels'] = copy.deepcopy(rels)
            rels = rels + neg_rel_sample
            random.shuffle(rels)
            rel_caption = '. '.join(rels) + '.'
            target['rel_caption'] = rel_caption

        return image, target 

    def get_groundtruth(self, index):
        relation = None
        item = self.images[index]

        gt_boxes = torch.tensor(item['bboxes'], dtype=torch.float)
        gt_classes = torch.tensor([self.label2cls[e]['idx'] for e in item['labels']], dtype=torch.long)

        return gt_boxes, gt_classes, relation

    def _parse_img_level_ann(self,
                             img_level_ann_file: str) -> Dict[str, List[dict]]:
        """Parse image level annotations from csv style ann_file.
        Args:
            img_level_ann_file (str): CSV style image level annotation
                file path.
        Returns:
            Dict[str, List[dict]]: Annotations where item of the defaultdict
            indicates an image, each of which has (n) dicts.
            Keys of dicts are:
                - `image_level_label` (int): Label id.
                - `confidence` (float): Labels that are human-verified to be
                  present in an image have confidence = 1 (positive labels).
                  Labels that are human-verified to be absent from an image
                  have confidence = 0 (negative labels). Machine-generated
                  labels have fractional confidences, generally >= 0.5.
                  The higher the confidence, the smaller the chance for
                  the label to be a false positive.
        """

        item_lists = defaultdict(list)
        with open(img_level_ann_file, 'r') as f:
            reader = csv.reader(f)
            for i, line in enumerate(reader):
                if i == 0:
                    continue
                img_id = line[0]
                item_lists[img_id].append(
                        dict(image_level_label=int(self.label2cls[line[2]]['idx']),
                            confidence=float(line[3]) 
                        )
                        )
        return item_lists

    def _get_relation_matrix(self, hierarchy_file: str, class_num: int = 601) -> np.ndarray:
        """Get the matrix of class hierarchy from the hierarchy file. Hierarchy
        for 600 classes can be found at https://storage.googleapis.com/openimag
        es/2018_04/bbox_labels_600_hierarchy_visualizer/circle.html.
        Args:
            hierarchy_file (str): File path to the hierarchy for classes.
        Returns:
            np.ndarray: The matrix of the corresponding relationship between
            the parent class and the child class, of shape
            (class_num, class_num).
        """  # noqa
        hierarchy = read_json(hierarchy_file)

        relation_matrix = np.eye(class_num, class_num)
        relation_matrix = self._convert_hierarchy_tree(hierarchy,
                                                       relation_matrix)
        return relation_matrix        

    def _convert_hierarchy_tree(self,
                                hierarchy_map: dict,
                                relation_matrix: np.ndarray,
                                parents: list = [],
                                get_all_parents: bool = True) -> np.ndarray:
        """Get matrix of the corresponding relationship between the parent
        class and the child class.
        Args:
            hierarchy_map (dict): Including label name and corresponding
                subcategory. Keys of dicts are:
                - `LabeName` (str): Name of the label.
                - `Subcategory` (dict | list): Corresponding subcategory(ies).
            relation_matrix (ndarray): The matrix of the corresponding
                relationship between the parent class and the child class,
                of shape (class_num, class_num).
            parents (list): Corresponding parent class.
            get_all_parents (bool): Whether get all parent names.
                Default: True
        Returns:
            ndarray: The matrix of the corresponding relationship between
            the parent class and the child class, of shape
            (class_num, class_num).
        """

        if 'Subcategory' in hierarchy_map:
            for node in hierarchy_map['Subcategory']:
                if 'LabelName' in node:
                    children_name = node['LabelName']
                    children_index = self.label2cls[children_name]['idx']
                    children = [children_index]
                else:
                    continue
                if len(parents) > 0:
                    for parent_index in parents:
                        if get_all_parents:
                            children.append(parent_index)
                        relation_matrix[children_index, parent_index] = 1
                relation_matrix = self._convert_hierarchy_tree(
                    node, relation_matrix, parents=children)
        return relation_matrix




def build_oicap(image_set, args, disable_transforms=False):

    rln_pretraining = getattr(args, "rln_pretraining", False)
    data_path = os.path.join(args.data_path, "open-imagev6")


    label2cls_file=os.path.join(data_path, "annotations/name2idx.pkl")
    if image_set == "train":
        img_dir = os.path.join(data_path, "train")
        try:
            anno_train_file = getattr(args, "anno_train_file", "oiv6-train-bbox.json")

            ann_file = os.path.join(data_path, "annotations", anno_train_file)
            assert os.path.exists(ann_file), "****** OpenImages: Please check train anno. file:%s"%ann_file
        except:
            raise Exception("*"*10 +  "args.anno_train_file:%s" % args.anno_train_file + " does not exist!")


        image_level_ann_file = os.path.join(data_path,
                "annotations/train-annotations-human-imagelabels-boxable.csv")

        #img_info_file = os.path.join(data_path, "annotations/train-img-info.pkl")
        img_info_file = None  # there is no need for training mode

    elif image_set in ["val", "validation"]:
        img_dir = os.path.join(data_path, "validation")
        try:
            ann_file = os.path.join(data_path, "annotations",
                                    args.anno_val_file)
            assert os.path.exists(ann_file), "***** OpenImages: Please check validation anno. file:%s"%ann_file
        except:
            raise Exception("*"*10 + " CHECK ann_file:%s"%ann_file)

        image_level_ann_file = os.path.join(data_path, 
                'annotations/validation-annotations-human-imagelabels-boxable.csv')

        img_info_file = os.path.join(data_path, "annotations/validation-img-info.pkl")
    else:
        raise ValueError("unknow image_set:%s for open-images"% image_set)

    strong_aug = getattr(args, "strong_aug", False)

    hierarchy_file= os.path.join(data_path, "annotations/bbox_labels_600_hierarchy.json")

    use_text_labels = getattr(args, "use_text_labels", False)

    nouns_list, relations_list = None, None
    if use_text_labels:
        oi_nouns_file = getattr(args, "oi_nouns_file", 
                                os.path.join(data_path, "annotations/oi-nouns.txt")
                                )
        oi_relations_file = getattr(args, "oi_relations_file",
                                os.path.join(data_path, "annotations/oi-relations.txt")
                                )
                             
        try:
            with open(oi_nouns_file, 'r') as fin:
                nouns_list = [line.rstrip().split(',')[0] for line in fin]
            with open(oi_relations_file, 'r') as fin:
                relations_list = [line.rstrip().split(',')[0] for line in fin]
        except:
            nouns_list = relations_list = []
            print("Warning: nouns_list and relations_list are empty!")

    return OICAPDataset(img_dir=img_dir, ann_file=ann_file, 
                        label2cls_file=label2cls_file,
                        img_info_file=img_info_file,
                        transforms=make_coco_transforms(image_set, 
                        fix_size=args.fix_size, 
                        strong_aug=strong_aug, args=args) if not disable_transforms else None,
                        image_level_ann_file=image_level_ann_file,
                        hierarchy_file=hierarchy_file,
                        use_text_labels=use_text_labels,
                        nouns_list=nouns_list,
                        relations_list=relations_list
                        )



if __name__ == "__main__":
    import sys
    from tqdm import tqdm 
    from groundingdino.util.slconfig import SLConfig
    from multiprocessing import Pool

    cfg_file = sys.argv[1]
    args = SLConfig.fromfile(cfg_file)
    args.data_path = './data' 


    dataset = build_oicap("train", args, True)

    def process_item(item):
        if len(item[1]['boxes']) < 2:
            return None
        item[1]['boxes'] = item[1]['boxes'].tolist()
        item[1]['labels'] = item[1]['labels']
        iw, ih = item[0].size
        item[1]['width'] = iw
        item[1]['height'] = ih
        return item[1]        

    def collate_fn(batch):
        batch = list(zip(*batch))
        return tuple(batch)


    processed_items = []
    for item in tqdm(dataset.images):
        if len(item['bboxes']) < 2:
            continue
        processed_items.append(item)

    print("total data:", len(processed_items), " sample[0]:", processed_items[0])
    # Save to JSON
    with open("oiv6-train-o2.json", "w") as fout:
        json.dump(processed_items, fout)        
