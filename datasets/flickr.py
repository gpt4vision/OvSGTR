import os
import torch
import json
import random
import numpy as np
import copy

from PIL import Image
from torchvision.datasets import Flickr30k

from datasets.coco import make_coco_transforms


class Flickr(torch.utils.data.Dataset):
    def __init__(self, root, ann_file,
            transforms, 
            nouns_list=None, relations_list=None):
        self.transforms = transforms
        self.root = root
        self.nouns_list = nouns_list
        self.relations_list = relations_list

        with open(ann_file, 'r') as fin:
            self.data = json.load(fin)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        img_id = item['image_id']

        # Image
        filename = os.path.join(self.root, img_id +'.jpg')
        img = Image.open(filename).convert("RGB")

        rels = []
        boxes = []
        _try_cnt = 0
        ridx = 0
        while len(rels) == 0 or len(boxes) == 0:
            ridx = random.randint(0, len(item['relations'])-1)
            rels = item['relations'][ridx]
            boxes = item['boxes'][ridx]
            _try_cnt += 1
            if _try_cnt > 100:
                break

        if _try_cnt > 100:
            #raise Exception("CHECK:{}".format(item))
            print("Warning: Flickr30k please check:{}".format(item))
            return self[index - random.randint(0, 10)]

        target = {}
        target['relations'] = rels
        target['gt_names'] = item['phrases'][ridx]
        target['image_id'] = img_id

        w, h = img.size        
        scale_fct = torch.as_tensor([w, h, w, h], dtype=torch.float)
        target['boxes'] = torch.as_tensor(item['boxes'][ridx]) * scale_fct

        # sample negative phrases.
        gt_names = copy.deepcopy(target['gt_names'])
        neg_count = max(1, 80 - len(gt_names))
        neg_names = list(set(self.nouns_list) - set(gt_names))
        neg_sample = np.random.choice(neg_names, size=neg_count, replace=False).tolist()
        nouns = gt_names + neg_sample
        random.shuffle(nouns)
        
        target['caption'] = '. '.join(nouns) + '.'
        
        rels = []
        for rel in target['relations']:
            rels.append(rel[2])
        rels = list(set(rels))
        neg_count = max(1, 80 - len(rels))
        neg_names = list(set(self.relations_list) - set(rels))
        neg_sample = np.random.choice(neg_names, size=neg_count, replace=False).tolist()
        rel_cap = rels + neg_sample
        target['rel_caption'] = '. '.join(rel_cap) + '.'

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


def build_flickr(image_set, args, nouns_list=None, relations_list=None):
    data_path = os.path.join(args.data_path, "flickr30k/images")

    transforms = make_coco_transforms(image_set, 
        fix_size=getattr(args, "fix_size", False),  
        strong_aug=getattr(args, "strong_aug", False), 
        args=args)

    if nouns_list is None or relations_list is None:
        with open(os.path.join(args.data_path, "flickr30k/flickr30k_nouns.txt")) as fin:
            nouns_list = [line.split(',')[0] for line in fin]
        with open(os.path.join(args.data_path, "flickr30k/flickr30k_relations.txt")) as fin:
            relations_list = [line.split(',')[0] for line in fin]

    return Flickr(data_path, 
                  os.path.join(args.data_path, "flickr30k/flickr30k_triple_grounded.json"),
                  transforms=transforms,
                  nouns_list=nouns_list,
                  relations_list=relations_list)



if __name__ == "__main__":
    import sys
    from tqdm import tqdm
    from groundingdino.util.slconfig import SLConfig

    args = SLConfig.fromfile(sys.argv[1])
    args.data_path = "data"
    dataset = build_flickr('train', args)

    for item in tqdm(dataset):
        import pdb; pdb.set_trace()
        pass
