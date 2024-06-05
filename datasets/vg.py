# modified from https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/maskrcnn_benchmark/data/datasets/visual_genome.py


import os
import sys
import torch
import h5py
import json
import copy
from PIL import Image
import numpy as np
from collections import defaultdict
from collections import OrderedDict
from tqdm import tqdm
import random

from util.bounding_box import BoxList, boxlist_iou
from util.box_ops import box_iou

from datasets.coco import make_coco_transforms
from pycocotools.coco import COCO

BOX_SCALE = 1024  # Scale at which we have the boxes


VG150_OBJ_CATEGORIES = ['__background__', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike', 'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building', 'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup', 'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence', 'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy', 'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean', 'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men', 'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw', 'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post', 'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt', 'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow', 'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel', 'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle', 'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

# Following "Towards Open-vocabulary Scene Graph Generation with Prompt-based Finetuning"
VG150_BASE_OBJ_CATEGORIES = ['__background__', 'tile', 'drawer', 'men', 'railing', 'stand', 'towel', 'sneaker', 'vegetable', 'screen', 'vehicle', 'animal', 'kite', 'cabinet', 'sink', 'wire', 'fruit', 'curtain', 'lamp', 'flag', 'pot', 'sock', 'boot', 'guy', 'kid', 'finger', 'basket', 'wave', 'lady', 'orange', 'number', 'toilet', 'post', 'room', 'paper', 'mountain', 'paw', 'banana', 'rock', 'cup', 'hill', 'house', 'airplane', 'plant', 'skier', 'fork', 'box', 'seat', 'engine', 'mouth', 'letter', 'windshield', 'desk', 'board', 'counter', 'branch', 'coat', 'logo', 'book', 'roof', 'tie', 'tower', 'glove', 'sheep', 'neck', 'shelf', 'bottle', 'cap', 'vase', 'racket', 'ski', 'phone', 'handle', 'boat', 'tire', 'flower', 'child', 'bowl', 'pillow', 'player', 'trunk', 'bag', 'wing', 'light', 'laptop', 'pizza', 'cow', 'truck', 'jean', 'eye', 'arm', 'leaf', 'bird', 'surfboard', 'umbrella', 'food', 'people', 'nose', 'beach', 'sidewalk', 'helmet', 'face', 'skateboard', 'motorcycle', 'clock', 'bear']

VG150_NOVEL2BASE = {'bed': [], 'bench': ['seat'], 'bike': ['vehicle', 'motorcycle'], 'boy': ['men'], 'building': [], 'bus': ['vehicle'], 'car': ['vehicle'], 'cat': ['animal'], 'chair': ['seat'], 'dog': ['animal'], 'door': [], 'ear': [], 'elephant': ['animal'], 'fence': [], 'giraffe': ['animal'], 'girl': ['lady'], 'glass': [], 'hair': [], 'hand': ['paw'], 'hat': ['cap'], 'head': ['men'], 'horse': ['animal'], 'jacket': ['coat'], 'leg': ['branch'], 'man': ['animal'], 'pant': [], 'person': ['people'], 'plane': ['airplane'], 'plate': ['food'], 'pole': [], 'shirt': [], 'shoe': [], 'short': [], 'sign': ['house'], 'snow': [], 'street': ['sidewalk'], 'table': ['board', 'desk'], 'tail': [], 'track': ['railing'], 'train': ['vehicle'], 'tree': ['plant'], 'wheel': ['tire'], 'window': [], 'woman': ['lady'], 'zebra': ['animal']}


PREDICATE_BG = '[UNK]' # speical token

VG150_PREDICATES = [PREDICATE_BG, "above", "across", "against", "along", "and", "at", "attached to", "behind", "belonging to", "between", "carrying", "covered in", "covering", "eating", "flying in", "for", "from", "growing on", "hanging from", "has", "holding", "in", "in front of", "laying on", "looking at", "lying on", "made of", "mounted on", "near", "of", "on", "on back of", "over", "painted on", "parked on", "part of", "playing", "riding", "says", "sitting on", "standing on", "to", "under", "using", "walking in", "walking on", "watching", "wearing", "wears", "with"]

VG150_BASE_PREDICATE = [PREDICATE_BG, "between", "to", "made of", "looking at", "along", "laying on", "using", "carrying", "against", "mounted on", "sitting on", "flying in", "covering", "from", "over", "near", "hanging from", "across", "at", "above", "watching", "covered in", "wearing", "holding", "and", "standing on", "lying on", "growing on", "under", "on back of", "with", "has", "in front of", "behind", "parked on"]

VG150_NOVEL_PREDICATE = ["belonging to", "part of", "riding", "walking in", "in", "of", "painted on", "playing", "for", "walking on", "says", "attached to", "eating", "on", "wears"]



def preprocess_caption(caption: str) -> str: 
    result = caption.lower().strip() 
    if result.endswith("."):    
        return result
    return result + "." 

class VGDataset(torch.utils.data.Dataset):
    def __init__(self, split, img_dir, roidb_file, dict_file, image_file, transforms=None,
                filter_empty_rels=True, num_im=-1, num_val_im=5000,
                filter_duplicate_rels=True, filter_non_overlap=True, flip_aug=False, 
                num_obj=100, 
                use_text_labels=False,
                roidb_key='split',
                ovd_mode=False, 
                ovr_mode=False,
                use_distill=False,
                unsupervised_distill=False,
                gpt4sgg_file=None,
		use_gpt4sgg=False
                ):
        """
        Torch dataset (COCO format) for VisualGenome
        Parameters:
            split: Must be train, test, or val
            img_dir: folder containing all vg images
            roidb_file:  HDF5 containing the GT boxes, classes, and relationships
            dict_file: JSON Contains mapping of classes/relationships to words
            image_file: HDF5 containing image filenames
            filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
            filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
            num_im: Number of images in the entire dataset. -1 for all images.
            num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
        """
        self.dataset_name = "vg"
        self.box_scale = BOX_SCALE

        assert split in {'train', 'val', 'test'}
        self.num_obj = num_obj
        self.flip_aug = flip_aug
        self.split = split
        self.img_dir = img_dir
        self.dict_file = dict_file
        self.roidb_file = roidb_file
        self.roidb_key = roidb_key
        self.image_file = image_file
        self.filter_non_overlap = filter_non_overlap and self.split == 'train'
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'
        self.transforms = transforms
        self.ovd_mode = ovd_mode 
        self.ovr_mode = ovr_mode
        self.use_distill = use_distill
        self.unsupervised_distill = unsupervised_distill

        if self.roidb_key != 'split':
            print("VG dataset use roidb key:", self.roidb_key)

        ind_to_classes, ind_to_predicates, ind_to_attributes = load_info(dict_file) # contiguous 151, 51 containing __background__

        #replace __background__ with '[bg]'
        if ind_to_predicates[0] == '__background__':
            ind_to_predicates[0] = PREDICATE_BG 
        
        self.ind_to_classes = ind_to_classes 
        self.ind_to_predicates = ind_to_predicates



        self.name2classes = OrderedDict({name: idx  for idx, name in enumerate(self.ind_to_classes) if name != '__background__'})
        self.class2name = OrderedDict({v:k for k, v in self.name2classes.items()})

        self.categories = [{'supercategory': 'person', # not used?
                            'id': idx, 
                            'name': self.ind_to_classes[idx]}  
                            for idx in range(len(self.ind_to_classes)) if self.ind_to_classes[idx] != '__background__'
                            ]
        self.ind_to_predicates = ind_to_predicates
        self.name2predicates = {ind_to_predicates[k]: k for k in range(len(ind_to_predicates))}


        split_mask, gt_boxes, gt_classes, gt_attributes, relationships, self.keep_nouns, self.keep_rels = load_graphs(
                self.roidb_file, self.split, num_im, num_val_im=num_val_im,
                filter_empty_rels=filter_empty_rels,
                filter_non_overlap=self.filter_non_overlap,
                roidb_key=self.roidb_key,
                keep_base_objects=(self.split=='train' and self.ovd_mode),
                keep_base_relations=(self.split=='train' and self.ovr_mode),
                ind_to_predicates=ind_to_predicates,
            )
        self.keep_nouns = sorted(self.keep_nouns.items(), key=lambda x:x[1], reverse=True)
        self.keep_rels = sorted(self.keep_rels.items(), key=lambda x:x[1], reverse=True)
        self.keep_nouns = [(self.ind_to_classes[e[0]], e[1]) for e in self.keep_nouns]
        self.keep_rels = [(self.ind_to_predicates[e[0]], e[1]) for e in self.keep_rels]

        self.gt_classes = gt_classes
        self.relationships = relationships
        self.gt_boxes = gt_boxes
        filenames, img_info  = load_image_filenames(img_dir, image_file) # length equals to split_mask

        filenames = [filenames[i] for i in np.where(split_mask)[0]]
        img_info = [img_info[i] for i in np.where(split_mask)[0]]
        assert len(img_info) == len(gt_boxes), " len(img_info) != len(gt_boxes)"


        # GPT4SGG
        self.use_gpt4sgg = use_gpt4sgg and self.split == 'train'
        self.gpt4sgg_data = None
        if self.use_gpt4sgg:
            print("VG150 GPT4SGG :%s used!"%gpt4sgg_file)
            with open(gpt4sgg_file, 'r') as fin:
                self.gpt4sgg_data = json.load(fin)
                self.gpt4sgg_data = {str(e['image_id']) : e for e in tqdm(self.gpt4sgg_data)}


        self.images = []
        keep_im = [1]*len(filenames)
        for idx, (name, im_info) in enumerate(zip(filenames, img_info)):
            if self.use_gpt4sgg and str(im_info['image_id']) not in self.gpt4sgg_data:
                keep_im[idx] = 0

            im_info.update({'file_name': name})
            im_info.update({'id': im_info['image_id']})
            self.images.append(im_info)

        self.annotations = []
        self.ids = [im['image_id'] for im, keep in zip(self.images, keep_im) if keep]

        images = []
        for idx, boxes, labels, edges in zip(range(len(self.images)), gt_boxes, gt_classes, relationships):
            if keep_im[idx] != 1:
                continue
            item = {'image_id': self.images[idx]['image_id'],
                    'boxes': boxes, #xyxy
                    'labels': labels,
                    'edges' : edges, 
                    'iscrowd': 0
                    }
            if self.use_gpt4sgg:
                item = {'image_id': self.images[idx]['image_id'],
                        'iscrowd': 0}
                gpt_item = self.gpt4sgg_data[str(item['image_id'])]
                iw = gpt_item['width']
                ih = gpt_item['height']
                boxes = gpt_item['bboxes'] if 'bboxes' in gpt_item else gpt_item['boxes']
                boxes = torch.as_tensor(boxes) * BOX_SCALE / max(iw, ih)
                item['boxes'] = boxes
                item['labels'] = gpt_item['labels']
                item['edges'] = gpt_item['edges']

            self.annotations.append(item)
            images.append(self.images[idx])

        self.images = images
        self._coco = None

        # WARNING: original image_file.json has several pictures with false image size
        # use correct function to check the validity before training
        # it will take a while, you only need to do it once
        ##correct_img_info(self.img_dir, self.image_file)

        self.use_text_labels = use_text_labels

        


    @property
    def coco(self):
        if self._coco is None:
            _coco = COCO()
            coco_dicts = dict(
                            images=self.images, 
                            annotations=[],
                            categories=self.categories)
            
            for ann in tqdm(self.annotations):
                for cls, box in zip(ann['labels'], ann['boxes']):
                    if True: # Visual Genome, BOX_SCALE
                        # important: recover original boxes 
                        box_scale = self.box_scale
                        id_ = self.ids.index(ann['image_id'])
                        img_w = self.images[id_]['width']
                        img_h = self.images[id_]['height']
                        box = box / box_scale * max(img_h, img_w)

                    item = {
                            'area': (box[3] - box[1]) * (box[2] - box[0]),
                            'bbox': [box[0], box[1], box[2] - box[0], box[3] - box[1]], # xywh
                            'category_id': cls,
                            'image_id': ann['image_id'], 
                            'id': len(coco_dicts['annotations']),
                            'iscrowd': 0,
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
        image_id = item["image_id"]
        w, h = item['width'], item['height']
        
        # load image
        img = Image.open(item['file_name']).convert("RGB")
        if img.size[0] != item['width'] or img.size[1] != item['height']:
            print('='*20, ' ERROR index ', str(index), ' ', str(img.size), ' ', 
                  str(item['width']), ' ', str(item['height']), ' ', '='*20)

        flip_img = (random.random() > 0.5) and self.flip_aug and (self.split == 'train')

        # load gt
        gt_boxes, gt_classes, gt_edges = self.get_groundtruth(index)

        if self.filter_duplicate_rels and self.split == "train":
            # Filter out dupes!
            old_size = gt_edges.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in gt_edges:
                all_rel_sets[(o0, o1)].append(r)
            gt_edges = [(k[0], k[1], np.random.choice(v)) for k,v in all_rel_sets.items()]
            gt_edges = np.array(gt_edges, dtype=np.int32)

        if self.split == "train":
            gt_boxes[:, [1,3]] = gt_boxes[:, [1, 3]].clip(None, h)
            gt_boxes[:, [0,2]] = gt_boxes[:, [0, 2]].clip(None, w)

        if flip_img:
            new_xmin = w - gt_boxes[:,2]
            new_xmax = w - gt_boxes[:,0]
            gt_boxes[:,0] = new_xmin
            gt_boxes[:,2] = new_xmax

            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        
        target = dict()
        target["iscrowd"] = torch.zeros(gt_boxes.shape[0], dtype=torch.float32)
        target["boxes"] = torch.as_tensor(gt_boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(gt_classes, dtype=torch.int64)
        target["edges"] = torch.as_tensor(gt_edges, dtype=torch.int64)
        target["image_id"] = image_id
        target["orig_size"] = torch.as_tensor([int(h), int(w)])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # text
        if self.use_text_labels:
            labels = target['labels'].cpu().numpy().tolist()
            gt_sample = [self.ind_to_classes[e] for e in labels]

            triples = []
            nouns = []
            rels = []
            for edge in target['edges']:
                sub = gt_sample[edge[0]]
                obj = gt_sample[edge[1]]

                triple = (sub, 
                          obj,
                          self.ind_to_predicates[edge[2]])
                triples.append(triple)
                nouns.append(sub)
                nouns.append(obj)
                rels.append(triple[2])

            target['relations'] = triples
            nouns = list(set(nouns))
            rels = list(set(rels))

            target['gt_names'] = copy.deepcopy(gt_sample)

            all_nouns = VG150_BASE_OBJ_CATEGORIES[1:] if (self.split == 'train' and self.ovd_mode) else \
                    VG150_OBJ_CATEGORIES[1:]
            
            #if self.ovd_mode and self.ovr_mode: # release the constraint 
            #    all_nouns = VG150_OBJ_CATEGORIES[1:]
            if self.split == 'train': 
                random.shuffle(all_nouns)

            
            target['caption'] = '. '.join(all_nouns)
            target['caption'] = preprocess_caption(target['caption'])


            target['gt_rels'] = copy.deepcopy(rels)
            all_rels = VG150_BASE_PREDICATE[1:] if (self.split == 'train' and self.ovr_mode) else \
                       VG150_PREDICATES[1:] # include __background__ 

            if self.use_distill and (not self.unsupervised_distill):
                all_rels = VG150_PREDICATES[1:] # we release the constraint of OvR mode, to let the model access to all categories but does not know which one is base or novel. And novel categories are removed in annotations.

            if self.split == 'train':
                random.shuffle(all_rels)

            rel_caption = '. '.join(all_rels) + '.'
            target['rel_caption'] = rel_caption

        if len(target['edges']) == 0 and self.split == 'train':
            return self[index - random.randint(1, 10)] # remap to a valid sample

        return img, target

    def get_groundtruth(self, index):
        item = self.images[index]
        image_id = item["image_id"]
        w, h = item['width'], item['height']

        ann = self.annotations[index]
        gt_boxes = ann['boxes']
        gt_classes = ann['labels']
        gt_edges = ann['edges']

        # truncate the gt if need
        max_obj = self.num_obj
        if len(gt_boxes) > max_obj:
            idx = np.random.choice(torch.arange(len(gt_boxes)), max_obj, replace=False)
            gt_boxes = gt_boxes[idx,:]
            gt_classes = gt_classes[idx]
            gt_edges = [[idx.tolist().index(rel[0]),idx.tolist().index(rel[1]),rel[2]]for rel in gt_edges if rel[0] in idx and rel[1] in idx]
            gt_edges = np.array(gt_edges)            

        # important: recover original box from BOX_SCALE
        gt_boxes = gt_boxes / BOX_SCALE * max(w, h)
        gt_boxes = torch.from_numpy(np.asarray(gt_boxes, dtype=np.float32)).reshape(-1, 4)  # guard against no boxes

        gt_classes = torch.as_tensor(gt_classes, dtype=torch.int64)

        # combine entities by with iou>0.9
        all_entities_infos_after_iou_merge = []
        ent_match_info = (box_iou(gt_boxes, gt_boxes)[0] > 0.9 ) & (gt_classes.unsqueeze(1) == gt_classes)

        keep_tgt_ids, ent_id_to_new_id = [], {}
        for ent_id in range(len(ent_match_info)):
            matched_ent_ids = ent_match_info[ent_id].nonzero().squeeze(1).tolist()
            if len(matched_ent_ids) > 0:
                keep_tgt_ids.append(ent_id)
                for id in matched_ent_ids: ent_id_to_new_id[id] = len(all_entities_infos_after_iou_merge)
                #all_entities_infos_after_iou_merge.append(all_entities_infos[ent_id])
                all_entities_infos_after_iou_merge.append(-1)
                ent_match_info[:, matched_ent_ids] = False
        
        gt_boxes = gt_boxes[keep_tgt_ids]
        gt_classes = gt_classes[keep_tgt_ids]
        new_edges = []
        for edge in gt_edges:
            new_edges.append([ent_id_to_new_id[edge[0]], ent_id_to_new_id[edge[1]], edge[2]])
        gt_edges = np.asarray(new_edges)


        return gt_boxes, gt_classes, gt_edges


    def get_statistics(self):
        fg_matrix, bg_matrix = get_VG_statistics(img_dir=self.img_dir, roidb_file=self.roidb_file, dict_file=self.dict_file,
                                                image_file=self.image_file, must_overlap=True)
        eps = 1e-3
        bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix
        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)

        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'obj_classes': self.ind_to_classes,
            'rel_classes': self.ind_to_predicates,
            #'att_classes': self.ind_to_attributes,
        }
        return result    

    

def box_filter(boxes, must_overlap=False):
    """ Only include boxes that overlap as possible relations. 
    If no overlapping boxes, use all of them."""
    n_cands = boxes.shape[0]

    overlaps = bbox_overlaps(boxes.astype(np.float), boxes.astype(np.float), to_move=0) > 0
    np.fill_diagonal(overlaps, 0)

    all_possib = np.ones_like(overlaps, dtype=np.bool)
    np.fill_diagonal(all_possib, 0)

    if must_overlap:
        possible_boxes = np.column_stack(np.where(overlaps))

        if possible_boxes.size == 0:
            possible_boxes = np.column_stack(np.where(all_possib))
    else:
        possible_boxes = np.column_stack(np.where(all_possib))
    return possible_boxes

def bbox_overlaps(boxes1, boxes2, to_move=1):
    """
    boxes1 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    boxes2 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    """
    #print('boxes1: ', boxes1.shape)
    #print('boxes2: ', boxes2.shape)
    num_box1 = boxes1.shape[0]
    num_box2 = boxes2.shape[0]
    lt = np.maximum(boxes1.reshape([num_box1, 1, -1])[:,:,:2], boxes2.reshape([1, num_box2, -1])[:,:,:2]) # [N,M,2]
    rb = np.minimum(boxes1.reshape([num_box1, 1, -1])[:,:,2:], boxes2.reshape([1, num_box2, -1])[:,:,2:]) # [N,M,2]

    wh = (rb - lt + to_move).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    return inter

def correct_img_info(img_dir, image_file):
    with open(image_file, 'r') as f:
        data = json.load(f)
    # make a copy
    with open(image_file.replace(".json", "_bak.json"), "w") as fout:
        json.dump(data, fout)

    tmp = []
    for i in range(len(data)):
        img = data[i]
        basename = '{}.jpg'.format(img['image_id'])
        filename = os.path.join(img_dir, basename)
        img_data = Image.open(filename).convert("RGB")
        if img['width'] != img_data.size[0] or img['height'] != img_data.size[1]:
            print('--------- False id: ', i, '---------')
            print(img_data.size)
            print(img)
            data[i]['width'] = img_data.size[0]
            data[i]['height'] = img_data.size[1]
            tmp.append(img)

    print("replace:", len(tmp))
    with open(image_file, 'w') as outfile:  
        json.dump(data, outfile)

def load_info(dict_file, add_bg=True):
    """
    Loads the file containing the visual genome label meanings
    """
    info = json.load(open(dict_file, 'r'))
    ind_to_attributes = None 
    if add_bg:
        info['label_to_idx']['__background__'] = 0
        info['predicate_to_idx']['__background__'] = 0
        #info['attribute_to_idx']['__background__'] = 0

    class_to_ind = info['label_to_idx']
    predicate_to_ind = info['predicate_to_idx']
    #attribute_to_ind = info['attribute_to_idx']
    ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
    ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])
    #ind_to_attributes = sorted(attribute_to_ind, key=lambda k: attribute_to_ind[k])

    return ind_to_classes, ind_to_predicates, ind_to_attributes


def load_image_filenames(img_dir, image_file):
    """
    Loads the image filenames from visual genome from the JSON file that contains them.
    This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
    Parameters:
        image_file: JSON file. Elements contain the param "image_id".
        img_dir: directory where the VisualGenome images are located
    Return: 
        List of filenames corresponding to the good images
    """
    with open(image_file, 'r') as f:
        im_data = json.load(f)

    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    fns = []
    img_info = []
    for i, img in enumerate(im_data):
        basename = '{}.jpg'.format(img['image_id'])
        if basename in corrupted_ims:
            continue

        filename = os.path.join(img_dir, basename)
        if os.path.exists(filename):
            fns.append(filename)
            img_info.append(img)

    assert len(fns) == 108073
    assert len(img_info) == 108073
    return fns, img_info


def load_graphs(roidb_file, split, num_im, num_val_im, filter_empty_rels, filter_non_overlap, 
                roidb_key='split', keep_base_objects=False, keep_base_relations=False, 
                ind_to_predicates=None):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    Parameters:
        roidb_file: HDF5
        split: (train, val, or test)
        num_im: Number of images we want
        num_val_im: Number of validation images
        filter_empty_rels: (will be filtered otherwise.)
        filter_non_overlap: If training, filter images that dont overlap.
    Return: 
        image_index: numpy array corresponding to the index of images we're using
        boxes: List where each element is a [num_gt, 4] array of ground 
                    truth boxes (x1, y1, x2, y2)
        gt_classes: List where each element is a [num_gt] array of classes
        relationships: List where each element is a [num_r, 3] array of 
                    (box_ind_1, box_ind_2, predicate) relationships
    """
    roi_h5 = h5py.File(roidb_file, 'r')
    data_split = roi_h5[roidb_key][:]
    split_flag = 2 if split == 'test' else 0
    split_mask = data_split == split_flag

    # Filter out images without bounding boxes
    split_mask &= roi_h5['img_to_first_box'][:] >= 0
    if filter_empty_rels:
        split_mask &= roi_h5['img_to_first_rel'][:] >= 0

    image_index = np.where(split_mask)[0]
    if num_im > -1:
        image_index = image_index[:num_im]
    if num_val_im > 0:
        if split == 'val':
            image_index = image_index[:num_val_im]
        elif split == 'train':
            image_index = image_index[num_val_im:]


    split_mask = np.zeros_like(data_split).astype(bool)
    split_mask[image_index] = True

    # Get box information
    all_labels = roi_h5['labels'][:, 0]
    ##all_attributes = roi_h5['attributes'][:, :]
    all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # cx,cy,w,h
    assert np.all(all_boxes[:, :2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box

    # convert from xc, yc, w, h to x1, y1, x2, y2
    all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2.0
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

    im_to_first_box = roi_h5['img_to_first_box'][split_mask]
    im_to_last_box = roi_h5['img_to_last_box'][split_mask]
    im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
    im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

    # load relation labels
    _relations = roi_h5['relationships'][:]
    _relation_predicates = roi_h5['predicates'][:, 0]
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check

    # Get everything by image.
    boxes = []
    gt_classes = []
    gt_attributes = []
    relationships = []
    
    keep_nouns, keep_rels = {}, {}
    for i in range(len(image_index)):
        i_obj_start = im_to_first_box[i]
        i_obj_end = im_to_last_box[i]
        i_rel_start = im_to_first_rel[i]
        i_rel_end = im_to_last_rel[i]

        boxes_i = all_boxes[i_obj_start : i_obj_end + 1, :]
        gt_classes_i = all_labels[i_obj_start : i_obj_end + 1]
        ##gt_attributes_i = all_attributes[i_obj_start : i_obj_end + 1, :]

        if i_rel_start >= 0:
            predicates = _relation_predicates[i_rel_start : i_rel_end + 1]
            obj_idx = _relations[i_rel_start : i_rel_end + 1] - i_obj_start # range is [0, num_box)
            assert np.all(obj_idx >= 0)
            assert np.all(obj_idx < boxes_i.shape[0])
            rels = np.column_stack((obj_idx, predicates)) # (num_rel, 3), representing sub, obj, and pred
        else:
            assert not filter_empty_rels
            rels = np.zeros((0, 3), dtype=np.int32)

        # map to base object classes
        if keep_base_objects:
            assert split == 'train'
            new_boxes, new_labels  = [], []
            new_id = {}
            for org_bid, (b, l) in enumerate(zip(boxes_i, gt_classes_i)):
                org_label = VG150_OBJ_CATEGORIES[l]

                if org_label in VG150_BASE_OBJ_CATEGORIES:
                    new_boxes.append(b)
                    new_labels.append(l)
                    new_id[org_bid] = len(new_id)
                elif len(VG150_NOVEL2BASE[org_label]) > 0:
                    tmp = random.choice(VG150_NOVEL2BASE[org_label])
                    new_l = VG150_OBJ_CATEGORIES.index(tmp)

                    new_boxes.append(b)
                    new_labels.append(new_l)
                    new_id[org_bid] = len(new_id)


            # filter relations
            new_rels = []
            for r in rels:
                if r[0] in new_id and r[1] in new_id:
                    new_rels.append([new_id[r[0]], new_id[r[1]], r[2]])

            if len(new_rels) == 0 or len(new_boxes) < 2:
                split_mask[image_index[i]] = 0
                continue
            else:
                boxes_i = np.stack(new_boxes)
                gt_classes_i = np.array(new_labels)
                rels = np.array(new_rels)        


        # base relations 
        if keep_base_relations:
            assert split == 'train'
            new_rels = []
            for rel in rels:
                rel_name = ind_to_predicates[rel[2]]
                if rel_name in VG150_BASE_PREDICATE:
                    new_rels.append(rel)

            if len(new_rels) == 0:
                split_mask[image_index[i]] = 0
                continue
            else:
                rels = np.asarray(new_rels)

        if filter_non_overlap:
            assert split == 'train'
            # construct BoxList object to apply boxlist_iou method
            # give a useless (height=0, width=0)
            boxes_i_obj = BoxList(boxes_i, (1000, 1000), 'xyxy')
            inters = boxlist_iou(boxes_i_obj, boxes_i_obj)
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            inc = np.where(rel_overs > 0.0)[0]

            if inc.size > 0:
                rels = rels[inc]
            else:
                split_mask[image_index[i]] = 0
                continue

        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        ##gt_attributes.append(gt_attributes_i)
        relationships.append(rels)

        for r in rels:
            r0 = gt_classes_i[r[0]]
            r1 = gt_classes_i[r[1]]

            if r0 not in keep_nouns:
                keep_nouns[r0] = 0
            if r1 not in keep_nouns:
                keep_nouns[r1] = 0
            keep_nouns[r0] += 1
            keep_nouns[r1] += 1
            if r[2] not in keep_rels:
                keep_rels[r[2]] = 0
            keep_rels[r[2]] += 1


    return split_mask, boxes, gt_classes, gt_attributes, relationships, keep_nouns, keep_rels 


def build_vg(image_set, args, disable_transforms=False):
    data_path = os.path.join(args.data_path, "visual_genome")

    try:
        strong_aug = args.strong_aug
    except:
        strong_aug = False

    transforms = make_coco_transforms(image_set, 
        fix_size=getattr(args, "fix_size", False),  
        strong_aug=strong_aug, args=args) if not disable_transforms else None

    use_text_labels = getattr(args, "use_text_labels", False)
    roidb_key = getattr(args, "vg_roidb_key", "split")

    filter_non_overlap = getattr(args, "filter_non_overlap", True)
    sg_ovd_mode = getattr(args, "sg_ovd_mode", False)
    sg_ovr_mode = getattr(args, "sg_ovr_mode", False)

    return VGDataset(split=image_set,
                     img_dir=os.path.join(data_path, "VG_100K"), 
                     roidb_file=os.path.join(data_path, "stanford_filtered/VG-SGG.h5"), 
                     dict_file=os.path.join(data_path, "stanford_filtered/VG-SGG-dicts.json"), 
                     image_file=os.path.join(data_path, "stanford_filtered/image_data.json"),
                     transforms=transforms,
                     use_text_labels=use_text_labels,
                     roidb_key=roidb_key,
                     filter_non_overlap=filter_non_overlap,
                     ovd_mode=sg_ovd_mode,
                     ovr_mode=sg_ovr_mode,
                     use_distill=getattr(args, "use_distill", False),
                     unsupervised_distill=getattr(args, "unsupervised_distill", False),
                     gpt4sgg_file=getattr(args, "gpt4sgg_file", None),
                     use_gpt4sgg=getattr(args, "use_gpt4sgg", False)
                     )

def get_VG_statistics(img_dir, roidb_file, dict_file, image_file, must_overlap=True):
    train_data = VGDataset(split='train', img_dir=img_dir, roidb_file=roidb_file, 
                        dict_file=dict_file, image_file=image_file, num_val_im=5000, 
                        filter_duplicate_rels=False)
    num_obj_classes = len(train_data.ind_to_classes)
    num_rel_classes = len(train_data.ind_to_predicates)
    fg_matrix = np.zeros((num_obj_classes, num_obj_classes, num_rel_classes), dtype=np.int64)
    bg_matrix = np.zeros((num_obj_classes, num_obj_classes), dtype=np.int64)

    for ex_ind in tqdm(range(len(train_data))):
        gt_classes = train_data.gt_classes[ex_ind].copy()
        gt_relations = train_data.relationships[ex_ind].copy()
        gt_boxes = train_data.gt_boxes[ex_ind].copy()

        # For the foreground, we'll just look at everything
        o1o2 = gt_classes[gt_relations[:, :2]]
        for (o1, o2), gtr in zip(o1o2, gt_relations[:,2]):
            fg_matrix[o1, o2, gtr] += 1
        # For the background, get all of the things that overlap.
        o1o2_total = gt_classes[np.array(
            box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]
        for (o1, o2) in o1o2_total:
            bg_matrix[o1, o2] += 1

    return fg_matrix, bg_matrix    


if __name__ == "__main__":
    import sys
    from tqdm import tqdm 
    from groundingdino.util.slconfig import SLConfig
    #from models.GroundingDINO.utils import FrequencyBias

    cfg_file = sys.argv[1]
    args = SLConfig.fromfile(cfg_file)

    args.data_path = './data' #sys.argv[2]
    args.sg_ovd_mode = False #True
    args.sg_ovr_mode = False #True
    args.filter_non_overlap = False
    args.vg_roidb_key = 'split'

    #args.gpt4sgg_file = "./data/visual_genome/vg150_gpt4sgg.json"
    #args.use_gpt4sgg = True


    dataset = build_vg("test", args, True)
    cnt = [0, 0]
    for item in tqdm(dataset):
        import pdb; pdb.set_trace()

