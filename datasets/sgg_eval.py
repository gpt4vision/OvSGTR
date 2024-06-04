# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO evaluator that works in distributed mode.

Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import os
import contextlib
import copy
import numpy as np
import torch
from collections import OrderedDict
from collections import defaultdict
from tabulate import tabulate
import itertools
from tqdm import tqdm 


#from pycocotools.cocoeval import COCOeval
from .cocoeval import COCOEval as COCOeval

from pycocotools.coco import COCO
import pycocotools.mask as mask_util

import torch.distributed as dist
from util.misc import get_rank, all_gather, get_world_size
from util import box_ops


from concurrent.futures import ThreadPoolExecutor 
import functools
import threading



from .sgg_metrics import (SGRecall, SGNoGraphConstraintRecall, SGZeroShotRecall, 
                        OvdSGZeroShotRecall, OvrSGZeroShotRecall,
                        SGNGZeroShotRecall, SGPairAccuracy, SGMeanRecall, 
                        SGNGMeanRecall, SGAccumulateRecall)

__all__ = ["SggEvaluator"]


def create_small_table(small_dict):
    """
    Create a small table using the keys of small_dict as headers. This is only
    suitable for small dictionaries.
    Args:
        small_dict (dict): a result dictionary of only a few items.
    Returns:
        str: the table as a string.
    """
    keys, values = tuple(zip(*small_dict.items()))
    table = tabulate(
        [values],
        headers=keys,
        tablefmt="pipe",
        floatfmt=".3f",
        stralign="center",
        numalign="center",
    )
    return table


def to_cpu(data):
    if isinstance(data, torch.Tensor):
        return data.cpu()
    if isinstance(data, list):
        return [to_cpu(e) for e in data]
    if isinstance(data, dict):
        return {k: to_cpu(v) for k, v in data.items()}
    return data

class SggEvaluator(object):
    def __init__(self, dataset, iou_types=("bbox", "relation"), 
                useCats=True,
                mode='sgdet',
                num_rel_category=51,
                multiple_preds=False,
                iou_thres=0.5,
                output_folder=None,
                num_workers=4,
                ovd_enabled=False,
                ovr_enabled=False
                ):
        """
            @iou_types: 'bbox'
            @mode: predcls, sgcls, sgdet
        """
        assert isinstance(iou_types, (list, tuple))

        self.dataset = dataset
        self.is_oiv6 = False
        try:
            self.is_oiv6 = self.dataset.relation_matrix is not None
        except:
            pass

        self.mode = mode 
        assert self.mode in ['predcls', 'sgcls', 'sgdet', 'det']
        self.num_rel_category = num_rel_category
        self.iou_types = iou_types
        self.iou_thres = iou_thres
        self.output_folder = output_folder
        self.ovd_enabled = ovd_enabled
        self.ovr_enabled = ovr_enabled 
       

        self.useCats = useCats
        self.multiple_preds = multiple_preds
        self.zeroshot_triplet = torch.load(os.path.join('./data/visual_genome/', 
                            "zeroshot_triplet.pytorch"), 
                            map_location=torch.device("cpu")).long().numpy()

        self.coco_gt = None
        self.coco_eval = {}
        if 'bbox' in iou_types:
            self.coco_gt = self.prepare_coco_gt(self.dataset)
            self.coco_eval['bbox'] = COCOeval(self.coco_gt, iouType='bbox')
            self.coco_eval['bbox'].useCats = self.useCats
            if self.is_oiv6:
                self.coco_eval['bbox'].params.maxDets = [1, 10, 300]

        if 'relation' in iou_types:
            self.do_sgg = True
        else:
            self.do_sgg = False 
        
        self.sgg_used_evaluators = []
        self.num_workers = num_workers
        self.lock_coco = threading.Lock() 
        self.lock_sgg = threading.Lock()
        self.reset()  

    def reset(self):
        self.img_ids = []
        self.eval_imgs = {k: [] for k in self.iou_types}

        try:
            self.sgg_result_dict, self.sgg_evaluator = self.get_sgg_evaluator(self.mode)

            self.global_container = {}
            self.global_container['zeroshot_triplet'] = self.zeroshot_triplet
            self.global_container['result_dict'] = self.sgg_result_dict
            self.global_container['mode'] = self.mode
            self.global_container['multiple_preds'] = self.multiple_preds
            self.global_container['num_rel_category'] = self.num_rel_category
            self.global_container['iou_thres'] = self.iou_thres

            self.do_sgg = True
        except:
            self.do_sgg = False 

        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)  # Add this line
        self.pending_tasks = []

        if 'bbox' in self.coco_eval:
            self.coco_eval['bbox'].gt_dt_valid = {}



    def update(self, predictions):
        predictions = to_cpu(predictions)

        """
        self.update_coco(predictions)
        self.update_relation(predictions)
        """

        update_coco_fn = functools.partial(self.update_coco, predictions)
        update_relation_fn = functools.partial(self.update_relation, predictions)

        self.pending_tasks.append(self.executor.submit(update_coco_fn))
        self.pending_tasks.append(self.executor.submit(update_relation_fn))        


    def update_coco(self, predictions):
        if 'bbox' not in self.iou_types:
            return 

        img_ids = list(np.unique(list(predictions.keys())))

        coco_res = self.prepare_coco_pred(predictions)

        with self.lock_coco:
            self.img_ids.extend(img_ids) 

            # suppress pycocotools prints
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = COCO.loadRes(self.coco_gt, coco_res) if coco_res else COCO()                

        with self.lock_coco:
            coco_eval = self.coco_eval['bbox']
            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            coco_eval.params.useCats = self.useCats 

            img_ids, eval_imgs = evaluate_coco(coco_eval)
            self.eval_imgs['bbox'].append(eval_imgs)

    def update_relation(self, predictions):
        if self.do_sgg and 'relation' in self.iou_types:
            for image_id, prediction in predictions.items():
                if 'graph' not in prediction:
                    self.do_sgg = False 
                    return 
                index = self.dataset.ids.index(image_id)
                groundtruth = {}
                gt_boxes, gt_labels, gt_edges = self.dataset.get_groundtruth(index)

                groundtruth['boxes'] = copy.deepcopy(gt_boxes)
                groundtruth['labels'] = copy.deepcopy(gt_labels)
                groundtruth['edges'] = copy.deepcopy(gt_edges)

                with self.lock_sgg:
                    _names = evaluate_relation_of_one_image(groundtruth, prediction['graph'], 
                                                            self.global_container, self.sgg_evaluator)
                    self.sgg_used_evaluators.extend(_names)
                    self.sgg_used_evaluators = list(set(self.sgg_used_evaluators))

        


    def prepare_coco_gt(self, dataset):
        coco_gt = dataset.coco
        assert coco_gt is not None
        return coco_gt



    def get_sgg_evaluator(self, mode):
        result_dict = {}
        evaluator = {}
        # tradictional Recall@K
        eval_recall = SGRecall(result_dict)
        eval_recall.register_container(mode)
        evaluator['eval_recall'] = eval_recall

        # no graphical constraint
        eval_nog_recall = SGNoGraphConstraintRecall(result_dict)
        eval_nog_recall.register_container(mode)
        evaluator['eval_nog_recall'] = eval_nog_recall

        # test on different distribution
        eval_zeroshot_recall = SGZeroShotRecall(result_dict)
        eval_zeroshot_recall.register_container(mode)
        evaluator['eval_zeroshot_recall'] = eval_zeroshot_recall

        if self.ovd_enabled:
            # ovd. zero shot
            eval_ovd_zeroshot_recall = OvdSGZeroShotRecall(result_dict)
            eval_ovd_zeroshot_recall.register_container(mode)
            evaluator['eval_ovd_zeroshot_recall'] = eval_ovd_zeroshot_recall

        if self.ovr_enabled:
            # ovr. zero shot
            eval_ovr_zeroshot_recall = OvrSGZeroShotRecall(result_dict)
            eval_ovr_zeroshot_recall.register_container(mode)
            evaluator['eval_ovr_zeroshot_recall'] = eval_ovr_zeroshot_recall



        # test on no graph constraint zero-shot recall
        eval_ng_zeroshot_recall = SGNGZeroShotRecall(result_dict)
        eval_ng_zeroshot_recall.register_container(mode)
        evaluator['eval_ng_zeroshot_recall'] = eval_ng_zeroshot_recall
        
        # used by https://github.com/NVIDIA/ContrastiveLosses4VRD for sgcls and predcls
        eval_pair_accuracy = SGPairAccuracy(result_dict)
        eval_pair_accuracy.register_container(mode)
        evaluator['eval_pair_accuracy'] = eval_pair_accuracy                

        # used for meanRecall@K
        eval_mean_recall = SGMeanRecall(result_dict, self.num_rel_category, 
                                        copy.deepcopy(self.dataset.ind_to_predicates),
                                        print_detail=True)
        eval_mean_recall.register_container(mode)
        evaluator['eval_mean_recall'] = eval_mean_recall

        # used for no graph constraint mean Recall@K
        eval_ng_mean_recall = SGNGMeanRecall(result_dict, self.num_rel_category, 
                                             copy.deepcopy(self.dataset.ind_to_predicates), 
                                             print_detail=True)
        eval_ng_mean_recall.register_container(mode)
        evaluator['eval_ng_mean_recall'] = eval_ng_mean_recall

        return result_dict, evaluator

    def prepare_coco_pred(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                coco_results.append([])
                continue

            pred_boxes = copy.deepcopy(prediction["boxes"]).cpu()
            pred_boxes = convert_to_xywh(pred_boxes).numpy()
            pred_scores = copy.deepcopy(prediction["scores"]).cpu().numpy()
            pred_labels = copy.deepcopy(prediction["labels"]).cpu().numpy()
            # add supercategories to gt
            if self.is_oiv6:     
                get_supercategory = True 
                filter_labels = True 
                index = self.dataset.ids.index(original_id)
                gt_boxes_, gt_labels_, gt_edges_ = self.dataset.get_groundtruth(index) 
                gt_boxes, gt_labels, gt_edges = copy.deepcopy(gt_boxes_), copy.deepcopy(gt_labels_), copy.deepcopy(gt_edges_)
                gt_labels_unique = np.unique(gt_labels)

                for label in gt_labels_unique:
                    super_labels = np.where(self.dataset.relation_matrix[label])[0]
                    for cls in super_labels:
                        if cls == label:
                            continue
                        idx = np.where(gt_labels == label)[0]
                        gt_boxes = np.concatenate((gt_boxes, gt_boxes[idx]))
                        extend_labels = np.full(idx.shape, cls, dtype=np.int64)
                        gt_labels = np.concatenate((gt_labels, extend_labels))

            
                img_level_anns = self.dataset.img_level_anns[original_id]
                img_labels = []
                confidences = []
                for ann in img_level_anns:
                    img_labels.append(int(ann['image_level_label']))
                    confidences.append(float(ann['confidence']))
                img_labels = np.array(img_labels, dtype=np.int64)
                allowed_classes = np.unique(
                        np.append(gt_labels, img_labels))

                pred_classes = np.unique(pred_labels)
                for pred_class in pred_classes:
                    classes = np.where(self.dataset.relation_matrix[pred_class])[0] # parent class 
                    for cls in classes:
                        if (cls in allowed_classes and cls != pred_class
                                and get_supercategory):
                            # add super-supercategory preds
                            idx = np.where(pred_labels == pred_class)[0]

                            pred_scores = np.concatenate((pred_scores, pred_scores[idx]))
                            pred_boxes = np.concatenate((pred_boxes, pred_boxes[idx]))
                            extend_labels = np.full(idx.shape, cls, dtype=np.int64)
                            pred_labels = np.concatenate((pred_labels, extend_labels))
                                
                        elif cls not in allowed_classes and filter_labels:
                            # remove unannotated preds
                            index = np.where(pred_labels != cls)[0]
                            pred_scores = pred_scores[index]
                            pred_boxes = pred_boxes[index]
                            pred_labels = pred_labels[index]



            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": pred_labels[k],
                        "bbox": box,
                        "score": pred_scores[k],
                    }
                    for k, box in enumerate(pred_boxes)
                ]
            )
        return coco_results



    def synchronize_between_processes(self):
        # Wait for the tasks to complete before proceeding
        print("Waiting for %s tasks to complete ..." % len(self.pending_tasks))
        for task in tqdm(self.pending_tasks):
            task.result()
        print("Rank:%s tasks finished." % get_rank())
        if get_world_size() > 1:
            dist.barrier()
        self.pending_tasks.clear()

        for iou_type in self.iou_types:
            if 'bbox' == iou_type:
                self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
                create_common_coco_eval(self.coco_eval[iou_type], 
                                        self.img_ids, self.eval_imgs[iou_type])

                if 'bbox' in self.coco_eval:
                    gt_dt_valid = all_gather(self.coco_eval['bbox'].gt_dt_valid)
                    tmp = {}
                    for item in gt_dt_valid:
                        for k, v in item.items():
                            if k not in tmp:
                                tmp[k] = v
                            else:
                                tmp[k]['gts'] += v['gts']
                                for i in range(len(v['dts'])):
                                    tmp[k]['dts'][i] += v['dts'][i]

                    self.coco_eval['bbox'].gt_dt_valid = tmp



    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

        if not self.do_sgg:
            return 0

        mode = self.mode
        evaluator = self.sgg_evaluator
        # calculate mean recall
        evaluator["eval_mean_recall"].calculate_mean_recall(mode)
        evaluator["eval_ng_mean_recall"].calculate_mean_recall(mode)
        
        # print result
        result_str = '=' * 100 + '\n'
        result_str += evaluator["eval_recall"].generate_print_string(mode)
        result_str += evaluator["eval_nog_recall"].generate_print_string(mode)
        result_str += evaluator["eval_zeroshot_recall"].generate_print_string(mode)
        result_str += evaluator["eval_ng_zeroshot_recall"].generate_print_string(mode)
        if self.ovd_enabled:
            result_str += evaluator["eval_ovd_zeroshot_recall"].generate_print_string(mode)
        if self.ovr_enabled:
            result_str += evaluator["eval_ovr_zeroshot_recall"].generate_print_string(mode)

        result_str += evaluator["eval_mean_recall"].generate_print_string(mode)
        result_str += evaluator["eval_ng_mean_recall"].generate_print_string(mode)
        
        if mode in ['predcls', 'sgcls']: 
            result_str += evaluator["eval_pair_accuracy"].generate_print_string(mode)
        result_str += '=' * 100 + '\n'

        #
        result_dict = self.sgg_result_dict
        if get_rank() == 0:
            print(result_str)
            if self.output_folder is not None:
                if not os.path.exists(self.output_folder):
                    os.makedirs(self.output_folder)
                torch.save(result_dict, os.path.join(self.output_folder, 'result_dict.pytorch'))

        rec_50 = float(np.mean(result_dict[self.mode + '_recall'][50]))
        zero_50 = 0
        res = {'R@50': rec_50}

        if self.ovd_enabled:
            zero_50 = float(np.mean(result_dict[self.mode + '_ovd_zeroshot_recall'][50]))
            res['zR-OvD@50'] = zero_50

        if self.ovr_enabled:
            zero_50 = float(np.mean(result_dict[self.mode + '_ovr_zeroshot_recall'][50]))
            res['zR-OvR@50'] = zero_50

        return res


    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            if get_rank() == 0:
                print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()            
            # per cat
            try:
                class_names = [e for e in self.dataset.ind_to_classes if e != '__background__']
            except:
                class_names = None 
            
            self._derive_coco_results(coco_eval, iou_type, class_names)


    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.
        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.
        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            print("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        if get_rank() == 0:
            print(
                "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
            )
        if not np.isfinite(sum(results.values())):
            print("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        ap50_per_category = []
        catIds = coco_eval.params.catIds
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image

            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

            # mean precision 
            t = np.where(coco_eval.params.iouThrs == 0.5)[0]
            ap50 = precisions[t]
            ap50 = ap50[:, :, idx, 0, -1]
            ap50 = ap50[ap50 > -1]
            ap50 = np.mean(ap50) if ap50.size else float("nan")
            # mean recall
            rec50 = coco_eval.eval["recall"][t] # T*K*A*M
            rec50 = rec50[:, idx, 0, -1]
            rec50 = rec50[rec50 > -1]
            rec50 = np.mean(rec50) if rec50.size else float("nan")

            gt_dts = coco_eval.gt_dt_valid
            cat_id = catIds[idx]
            if cat_id in gt_dts:
                gts = gt_dts[cat_id]['gts']
                dts = gt_dts[cat_id]['dts'][0]
            else:
                gts = 0
                dts = 0
            ap50_per_category.append(("{}".format(name), "%s"%gts, "%s"%dts, float(rec50*100), float(ap50*100) ))



        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        if get_rank() == 0:
            print("Per-category {} AP: \n".format(iou_type) + table)
        # AP50
        N_COLS = min(5, len(ap50_per_category)*2 )
        results_flatten = list(itertools.chain(*ap50_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "gts", "dts", "recall", "ap"]*(N_COLS//2),
            numalign="left",
        )
        if get_rank() == 0:
            print("Per-category {} AP50: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results




def evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator):
    """
    Returns:
        pred_to_gt: Matching from predicate to GT
        pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
        pred_triplet_scores: [cls_0score, relscore, cls1_score]
    """
    evaluator_names = []
    #unpack all inputs
    mode = global_container['mode']

    local_container = {}
    local_container['gt_rels'] = groundtruth['edges']

    # if there is no gt relations for current image, then skip it
    if len(local_container['gt_rels']) == 0:
        return

    local_container['gt_boxes'] = groundtruth['boxes']                  # (#gt_objs, 4)
    local_container['gt_classes'] = groundtruth['labels']              # (#gt_objs, )

    # about relations
    local_container['pred_rel_inds'] = prediction['all_node_pairs'].numpy()  # (#pred_rels, 2)
    local_container['rel_scores'] = prediction['all_relation'].numpy()          # (#pred_rels, num_pred_class)

    # about objects
    local_container['pred_boxes'] = prediction['pred_boxes'].numpy()                  # (#pred_objs, 4)
    local_container['pred_classes'] = prediction['pred_boxes_class'].numpy()     # (#pred_objs, )
    local_container['obj_scores'] = prediction['pred_boxes_score'].numpy()              # (#pred_objs, )
    

    # to calculate accuracy, only consider those gt pairs
    # This metric is used by "Graphical Contrastive Losses for Scene Graph Parsing" 
    # for sgcls and predcls
    if mode != 'sgdet':
        evaluator['eval_pair_accuracy'].prepare_gtpair(local_container)

    # to calculate the prior label based on statistics
    evaluator['eval_zeroshot_recall'].prepare_zeroshot(global_container, local_container)
    evaluator['eval_ng_zeroshot_recall'].prepare_zeroshot(global_container, local_container)

    if 'eval_ovd_zeroshot_recall' in evaluator:
        evaluator['eval_ovd_zeroshot_recall'].prepare_zeroshot(global_container, local_container)
    if 'eval_ovr_zeroshot_recall' in evaluator:
        evaluator['eval_ovr_zeroshot_recall'].prepare_zeroshot(global_container, local_container)

    if mode == 'predcls':
        local_container['pred_boxes'] = local_container['gt_boxes']
        local_container['pred_classes'] = local_container['gt_classes']
        local_container['obj_scores'] = np.ones(local_container['gt_classes'].shape[0])

    elif mode == 'sgcls':
        if local_container['gt_boxes'].shape[0] != local_container['pred_boxes'].shape[0]:
            print('Num of GT boxes is not matching with num of pred boxes in SGCLS')
    elif mode == 'sgdet' or mode == 'phrdet':
        pass
    else:
        raise ValueError('invalid mode')
    """
    elif mode == 'preddet':
        # Only extract the indices that appear in GT
        prc = intersect_2d(pred_rel_inds, gt_rels[:, :2])
        if prc.size == 0:
            for k in result_dict[mode + '_recall']:
                result_dict[mode + '_recall'][k].append(0.0)
            return None, None, None
        pred_inds_per_gt = prc.argmax(0)
        pred_rel_inds = pred_rel_inds[pred_inds_per_gt]
        rel_scores = rel_scores[pred_inds_per_gt]

        # Now sort the matching ones
        rel_scores_sorted = argsort_desc(rel_scores[:,1:])
        rel_scores_sorted[:,1] += 1
        rel_scores_sorted = np.column_stack((pred_rel_inds[rel_scores_sorted[:,0]], rel_scores_sorted[:,1]))

        matches = intersect_2d(rel_scores_sorted, gt_rels)
        for k in result_dict[mode + '_recall']:
            rec_i = float(matches[:k].any(0).sum()) / float(gt_rels.shape[0])
            result_dict[mode + '_recall'][k].append(rec_i)
        return None, None, None
    """

    if local_container['pred_rel_inds'].shape[0] == 0:
        return

    # Traditional Metric with Graph Constraint
    # NOTE: this is the MAIN evaluation function, it must be run first (several important variables need to be update)
    local_container = evaluator['eval_recall'].calculate_recall(global_container, local_container, mode)

    # No Graph Constraint
    evaluator['eval_nog_recall'].calculate_recall(global_container, local_container, mode)
    evaluator_names.append('eval_nog_recall')
    # GT Pair Accuracy
    evaluator['eval_pair_accuracy'].calculate_recall(global_container, local_container, mode)
    evaluator_names.append('eval_pair_accuracy')
    # Mean Recall
    evaluator['eval_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)
    evaluator_names.append('eval_mean_recall')
    # No Graph Constraint Mean Recall
    evaluator['eval_ng_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)
    evaluator_names.append('eval_ng_mean_recall')
    # Zero shot Recall
    evaluator['eval_zeroshot_recall'].calculate_recall(global_container, local_container, mode)
    evaluator_names.append('eval_zeroshot_recall')
    # No Graph Constraint Zero-Shot Recall
    evaluator['eval_ng_zeroshot_recall'].calculate_recall(global_container, local_container, mode)
    evaluator_names.append('eval_ng_zeroshot_recall')

    if 'eval_ovd_zeroshot_recall' in evaluator:
        evaluator['eval_ovd_zeroshot_recall'].calculate_recall(global_container, local_container, mode)
        evaluator_names.append('eval_ovd_zeroshot_recall')

    if 'eval_ovr_zeroshot_recall' in evaluator:
        evaluator['eval_ovr_zeroshot_recall'].calculate_recall(global_container, local_container, mode)
        evaluator_names.append('eval_ovr_zeroshot_recall')

    return evaluator_names

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def evaluate_coco(self):
    '''
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    '''
    # tic = time.time()
    # print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    # print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = self.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = self.computeOks
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    # toc = time.time()
    # print('DONE (t={:0.2f}s).'.format(toc-tic))
    return p.imgIds, evalImgs

def merge(img_ids, eval_imgs):
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)






