# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modules to compute the matching cost and solve the corresponding LSAP.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------


import torch, os
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


def search_query_pos(lst, query):
    start = -1
    end = -1

    for i in range(len(lst)):
        if lst[i:i+len(query)] == query and lst[i-1] in [101, 1012, 0]:
            start = i
            end = i + len(query) - 1
            break

    return start, end

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, 
                 cost_giou: float = 1, focal_alpha = 0.25,
                 **kwargs
                ):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

        self.focal_alpha = focal_alpha

        self.tokenizer = kwargs.get("tokenizer", None)

        self.has_bbox_supervision = kwargs.get("has_bbox_supervision", True)



    @torch.no_grad()
    def forward(self, outputs, targets, eps=1e-5):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        pos_seg = []
        for bid, target in enumerate(targets):
            gt_names = target['gt_names']
            all_ids = target['input_ids']
            for name in gt_names:
                ids = self.tokenizer(name + '. ').input_ids[1:-1] # ref to name +.
                start_i, end_i = search_query_pos(all_ids.tolist(), ids)
                assert start_i != end_i, "cannot find query:{} from input_ids:{}!".format(
                            name, 
                            self.tokenizer.decode(all_ids) )
                pos_seg.append((start_i, end_i))


        # Compute the classification cost.
        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        alpha = self.focal_alpha
        gamma = 2.0

        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + eps).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + eps).log())
        cost_class_all = pos_cost_class - neg_cost_class

        cost_class = torch.zeros((bs*num_queries, len(pos_seg)), device=out_prob.device)
        for idx, seg in enumerate(pos_seg):
            cost_class[:, idx] = cost_class_all[:, seg[0] : seg[1]].mean(-1)

        if self.has_bbox_supervision:
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
                
            # Compute the giou cost betwen boxes            
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        else:
            cost_bbox = cost_giou = 0

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        if self.has_bbox_supervision:
            sizes = [len(v["boxes"]) for v in targets]
        else:
            sizes = [len(v['gt_names']) for v in targets]

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class SimpleMinsumMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, focal_alpha = 0.25):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

        self.focal_alpha = focal_alpha

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]

        tgt_ids = torch.cat([v["labels"] for v in targets])

        # Compute the classification cost.
        alpha = self.focal_alpha
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        # Also concat the target labels and boxes
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            
        # Compute the giou cost betwen boxes            
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1)

        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        device = C.device
        for i, (c, _size) in enumerate(zip(C.split(sizes, -1), sizes)):
            weight_mat = c[i]
            idx_i = weight_mat.min(0)[1]
            idx_j = torch.arange(_size).to(device)
            indices.append((idx_i, idx_j))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args, **kwargs):
    matcher_type = getattr(args, "matcher_type", "HungarianMatcher")

    assert matcher_type in ['HungarianMatcher', 'SimpleMinsumMatcher'], "Unknown matcher_type: {}".format(matcher_type)
    if matcher_type == 'HungarianMatcher':
        return HungarianMatcher(
            cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha ,
            tokenizer=kwargs.get("tokenizer", None),
            has_bbox_supervision=getattr(args, "has_bbox_supervision", True)
        )
    elif matcher_type == 'SimpleMinsumMatcher':
        return SimpleMinsumMatcher(
            cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha
        )    
    else:
        raise NotImplementedError("Unknown matcher_type: {}".format(matcher_type))


if __name__ == '__main__':
    import sys

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("bert-base-uncased")

    with open(sys.argv[1], 'r') as fin:
        cats = [e.rstrip() for e in fin]

    cap = '. '.join(cats) + '.'

    all_ids = tok(cap).input_ids

    for name in cats:
        ids = tok(name +'. ').input_ids[1:-1]

        pos = search_query_pos(all_ids, ids)

        dec = tok.decode(all_ids[pos[0]:pos[1]])
        if dec != name:
            import pdb; pdb.set_trace()




