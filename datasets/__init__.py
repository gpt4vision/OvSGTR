# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision
import copy

from .coco import build as build_coco
from .vg import build_vg

from torch.utils.data import ConcatDataset


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        rln_pretraining = getattr(args, "rln_pretraining", False)
        if rln_pretraining and image_set != "train":
            print("*"*10, "visual genome dataset (%s) will be used !" % image_set, "*"*10)
            return build_vg(image_set, args)

        return build_coco(image_set, args)


    if args.dataset_file == 'oicap':
        from .oiv6 import build_oicap
        
        rln_pretraining = getattr(args, "rln_pretraining", False)
        if rln_pretraining and image_set != "train":
            print("*"*10, "visual genome dataset (%s) will be used !" % image_set, "*"*10)
            return build_vg(image_set, args)

        print("*"*10, "Training dataset will use oi dataset!", "*"*10)
        return build_oicap(image_set, args)

    if args.dataset_file == 'vg':
        return build_vg(image_set, args)


    if args.dataset_file == 'coco_vg':
        coco_data = build_coco(image_set, args)
        vg_data = build_vg(image_set, args)

        return ConcatDataset([coco_data, vg_data])

    if args.dataset_file == 'coco_flickr30k':
        from .flickr import build_flickr

        rln_pretraining = getattr(args, "rln_pretraining", False)
        if rln_pretraining and image_set != "train":
            print("*"*10, "visual genome dataset (%s) will be used !" % image_set, "*"*10)
            return build_vg(image_set, args)

        coco_data = build_coco(image_set, args)
        flickr_data = build_flickr(image_set, args)

        return ConcatDataset([coco_data, flickr_data])

    if args.dataset_file == 'coco_flickr30k_sbucaptions':
        from .sbucaptions import build_sbucaptions
        from .flickr import build_flickr

        rln_pretraining = getattr(args, "rln_pretraining", False)
        if rln_pretraining and image_set != "train":
            print("*"*10, "visual genome dataset (%s) will be used !" % image_set, "*"*10)
            return build_vg(image_set, args)

        coco_data = build_coco(image_set, args)
        flickr_data = build_flickr(image_set, args)
        sbu_data = build_sbucaptions(image_set, args)

        return ConcatDataset([coco_data, flickr_data, sbu_data])


    raise ValueError(f'dataset {args.dataset_file} not supported')
