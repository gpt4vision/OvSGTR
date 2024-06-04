import torch
import torch.nn as nn
import torch.nn.functional as F
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from util import box_ops
from .utils import sigmoid_focal_loss   

from scipy.optimize import linear_sum_assignment

import os
import copy

from torchvision.ops import roi_pool

from .matcher import search_query_pos

import math
import numpy as np
import random
import bisect


    
def padding_last(logits, max_len=2048):
    logits_pad = torch.full((*logits.shape[:-1], max_len), 
                               float("-inf"), device=logits.device)
    logits_pad[..., : logits.shape[-1]] = logits

    return logits_pad

def shrink_sigmoid(x, scale=1.):
    return 1.0 / (1.0 + torch.exp(-scale * x))


def tensor_to_list(x):
    if isinstance(x, torch.Tensor):
        return x.tolist()
    return x


class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self,  matcher, weight_dict, focal_alpha, losses, 
                **kwargs):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_loss_for_edges = kwargs.get("focal_loss_for_edges", False)

        self.rln_proj = kwargs.get("rln_proj", None)
        self.rln_proj_teacher = kwargs.get("rln_proj_teacher", None)
        self.rln_classifier = kwargs.get("rln_classifier", None)
        self.rln_freq_bias = kwargs.get("rln_freq_bias", None)


        self.rln_pretraining = kwargs.get("rln_pretraining", False)
        self.tokenizer = kwargs.get("tokenizer", None)
        self.ind_to_predicates = kwargs.get("ind_to_predicates", None)
        self.global_iter = -1
        hidden_dim = kwargs.get("hidden_dim", 256)
        self.min_obj=-hidden_dim*math.log(0.9)
        self.obj_temp = kwargs.get("obj_temp", 1.3 / hidden_dim)
        self.obj_start_iter = kwargs.get("obj_start_iter", 1000)
        self.obj_threshold = kwargs.get("obj_threshold", 0.5)

        self.rel_proposals_threshold = kwargs.get("rel_proposals_threshold", 0.5)
        self.rel_proposals_threshold_enabled = kwargs.get("rel_proposals_threshold_enabled", False)

        self.rel_batch_per_image = kwargs.get("rel_batch_per_image", 64)
        self.unsupervised_distill = kwargs.get("unsupervised_distill", False)
        self.is_closed_set = (self.rln_classifier is not None)  and (not self.focal_loss_for_edges)
        self.ablation_mode = kwargs.get("ablation_mode", -1)
        self.fix_rel_batch = kwargs.get("fix_rel_batch", False)


    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_logits = outputs['pred_logits'] # (bsz, num_queries, 256)
        src_mask = src_logits == float('-inf')
        # 
        tgt_pos_seg = []
        for bid, (target, (_, J)) in enumerate(zip(targets, indices)):
            gt_names = target['gt_names']
            all_ids = target['input_ids']
            pos_seg = []
            for name in gt_names:
                ids = self.tokenizer(name +'.').input_ids[1:-1]
                start_i, end_i = search_query_pos(all_ids.tolist(), ids)
                assert start_i != end_i, "cannot find query:{} from input_ids:{}".format(ids,
                                self.tokenizer.decode(all_ids))
                pos_seg.append((start_i, end_i))

            for j in J.tolist():
                tgt_pos_seg.append(pos_seg[j])


        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)


        # set positive labels
        for bid, qid, seg in zip(idx[0], idx[1], tgt_pos_seg):
            target_classes_onehot[bid, qid, seg[0]:seg[1]].fill_(1.0)

        target_classes_onehot = target_classes_onehot[:,:,:-1]

        alpha = self.focal_alpha
        gamma = 2.0

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, 
                                     alpha=alpha, gamma=gamma, reduction='none') * src_logits.shape[1]
        loss_ce.masked_fill_(src_mask, 0.)

        nb_pos = target_classes_onehot.sum(-1)
        nb_pos[nb_pos == 0] = 1.0

        loss_ce = (loss_ce / nb_pos.unsqueeze(2)).mean(1).sum() / num_boxes
        losses = {'loss_ce': loss_ce}


        return losses


    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)


        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))

        losses['loss_giou'] = loss_giou.sum() / num_boxes

        ## calculate the x,y and h,w loss
        #with torch.no_grad():
        #    losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
        #    losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'edges': self.loss_edges,
            'obj_likelihood': self.loss_obj_likelihood,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def loss_obj_likelihood(self, outputs, targets, indices, num_boxes):
        assert "pred_obj" in outputs, "pred_obj does not exist in outputs, outputs.keys:{}".format(outputs.keys())

        idx = self._get_src_permutation_idx(indices)
        pred_obj = outputs["pred_obj"][idx]
        return  {'loss_obj_ll': torch.clamp(pred_obj, min=self.min_obj).sum()/ num_boxes}

    @torch.no_grad()
    def get_objectness(self, pred_obj):
        return torch.exp(-self.obj_temp*pred_obj)



    def loss_edges(self, outputs, targets, indices, num_boxes,
                   object_token, relation_token, rel_text_dict, 
                   rel_text_dict_t=None, outputs_t=None, indices_t=None):
        """
          compute loss for relations 
        """
        self.is_closed_set = (self.rln_classifier is not None)  and (not self.focal_loss_for_edges)
        device = outputs['pred_logits'].device
        bs, num_queries = outputs['pred_logits'].shape[:2]
        relation_wo_labels = 'edges' not in targets[0]
        relation_feature_t = None
        relation_is_str = False
        sample_cross_frame = False
        rel_tgt = []
        # check if predicates are str
        for target in targets:
            if 'edges' not in target:
                break

            if len(target['edges']) > 0 and isinstance(target['edges'][0][2], str) and (not relation_is_str):
                relation_is_str = True
                sample_cross_frame = relation_is_str
                break

        # if "edges" provided
        if not relation_wo_labels: # 
            all_edge_lbl = []
            freq_dist = []
            batch_ids = []

            sid, oid = [], []
            s_labels, o_labels = [], []
            sid_t, oid_t = [], []
            for bid, target in enumerate(targets):
                tgt_edges = target['edges']
                if len(tgt_edges) == 0:
                    continue 
                
                matched = {}
                for src, dst in zip(indices[bid][0].tolist(), indices[bid][1].tolist()):
                    matched[dst] = src
                # negatives
                num_pos = num_total = 0
                n = len(matched)
                full_adj = torch.ones((n,n))-torch.diag(torch.ones(n))
                for edge in tensor_to_list(tgt_edges):
                    if edge[0] in matched and edge[1] in matched:
                        full_adj[edge[0], edge[1]] = 0

                # teacher's matched nodes (optional)
                if outputs_t is not None:
                    matched_t = {}
                    for src, dst in zip(indices_t[bid][0].tolist(), indices_t[bid][1].tolist()):
                        matched_t[dst] = src

                
                # positives 
                if not self.is_closed_set and self.fix_rel_batch:
                    pos_num_per_img = int(self.rel_batch_per_image * 0.25)
                    if len(tgt_edges) > pos_num_per_img:
                        if isinstance(tgt_edges, list):
                            tgt_edges = random.sample(tgt_edges, pos_num_per_img)
                        else:
                            tgt_edges = tgt_edges[torch.randperm(len(tgt_edges))][:pos_num_per_img]

                for edge in tensor_to_list(tgt_edges):
                    if edge[0] in matched and edge[1] in matched:
                        sid.append([bid, matched[edge[0]]])
                        oid.append([bid, matched[edge[1]]])
                        if relation_is_str:
                            rel_tgt.append(edge[2])
                        else:
                            all_edge_lbl.append(edge[2])

                        num_pos += 1
                        #full_adj[edge[0], edge[1]] = 0

                        if self.rln_freq_bias is not None:
                            s_labels.append(target['labels'][edge[0]])
                            o_labels.append(target['labels'][edge[1]])

                        if outputs_t is not None:
                            sid_t.append([bid, matched_t[edge[0]]])
                            oid_t.append([bid, matched_t[edge[1]]])

                # negatives 
                neg_edges = torch.nonzero(full_adj)
                if self.is_closed_set or (relation_is_str and not self.fix_rel_batch):
                    num_neg_per_img = max(1, num_pos * 3)  
                else:
                    num_neg_per_img = max(1, self.rel_batch_per_image - num_pos) 

                if len(neg_edges) >= num_neg_per_img:
                    if outputs_t is not None and not self.unsupervised_distill:
                        # sample
                        hs_obj_t = outputs_t['hs_obj'][bid]
                        nsid, noid = [], []
                        for edge in neg_edges.tolist():
                            nsid.append(matched_t[edge[0]])
                            noid.append(matched_t[edge[1]])
                        nsid = torch.as_tensor(nsid)
                        noid = torch.as_tensor(noid)
                        
                        feat = torch.cat((hs_obj_t[nsid], hs_obj_t[noid],
                                          outputs_t['hs_rln'][bid].flatten(1).repeat(nsid.shape[0], 1)),
                                          1)
                        with torch.no_grad():
                            feat = self.rln_proj_teacher(feat)
                            encoded_text = rel_text_dict_t['encoded_text'][bid]
                            feat = (feat @ encoded_text.T).sigmoid()

                        feat_score = feat.max(-1)[0]

                        if self.rel_proposals_threshold_enabled:
                            keep1 = torch.where(feat_score > self.rel_proposals_threshold)[0]
                            keep2 = feat_score.topk(num_neg_per_img)[1]
                            keep = torch.cat((keep1, keep2)).unique()
                        else:
                            keep = feat_score.topk(num_neg_per_img)[1]

                        neg_edges = neg_edges[keep.to(neg_edges.device)]
                    else:
                        if sample_cross_frame:
                            idx_ = torch.randperm(neg_edges.shape[0])[: max(1, num_pos*3)]
                            neg_edges = neg_edges[idx_]
                        else:
                            idx_ = torch.randperm(neg_edges.shape[0])[:num_neg_per_img]
                            neg_edges = neg_edges[idx_]
                
                for edge in neg_edges.tolist():
                    sid.append([bid, matched[edge[0]]])
                    oid.append([bid, matched[edge[1]]])

                    if self.rln_freq_bias is not None:
                        s_labels.append(target['labels'][edge[0]])
                        o_labels.append(target['labels'][edge[1]])

                    if relation_is_str:
                        rel_tgt.append('[UNK]')
                    else:
                        all_edge_lbl.append(0)

                    num_total += 1
                    if outputs_t is not None:
                        sid_t.append([bid, matched_t[edge[0]]])
                        oid_t.append([bid, matched_t[edge[1]]])

                # sample within a batch if needed
                if sample_cross_frame:
                    sample_num = num_neg_per_img - len(neg_edges)
                    if sample_num > 0 and len(targets) > 1:
                        p = np.array([1]*len(targets))
                        p[bid] = 0
                        p = p / p.sum() 
                        sample_bid = np.random.choice(range(len(targets)), sample_num, p=p)
                        sample_tid = np.random.choice(range(object_token.shape[1]), sample_num)
                         
                        for c_bid, c_tid in zip(sample_bid, sample_tid):
                            if len(tgt_edges) > 0:
                                 pos_edge = random.choice(tgt_edges)
                                 pos_edge = (matched[pos_edge[0]], matched[pos_edge[1]])
                            else: 
                                 if len(matched) > 0:
                                     pos_edge = np.random.choice(list(matched.keys()), 2)
                                     pos_edge = (matched[pos_edge[0]], matched[pos_edge[1]])
                                 else:
                                     pos_edge = np.random.choice(range(object_token.shape[1]), 2)

                            if random.random() > 0.5:
                                 sid.append([bid, pos_edge[0]])
                                 oid.append([c_bid, c_tid])
                            else:
                                 oid.append([bid, pos_edge[1]])
                                 sid.append([c_bid, c_tid])
                            rel_tgt.append('[UNK]')
                            num_total += 1 

                num_total += num_pos
                batch_ids.extend([bid] * num_total)

            assert len(sid) == len(oid) and len(sid) > 0, " Error: len(sid):%s, len(oid):%s" %(len(sid), len(oid))
            sid = torch.as_tensor(sid)
            oid = torch.as_tensor(oid)
            if not relation_is_str:
                all_edge_lbl = torch.as_tensor(all_edge_lbl)

            batch_ids = torch.as_tensor(batch_ids)
            if self.ablation_mode == 'wo_rln':
                relation_feature = torch.cat((object_token[sid[:, 0], sid[:, 1]],
                                 object_token[oid[:, 0], oid[:, 1]]), 1)
            elif self.ablation_mode == 'avg_rln':
                relation_feature = 0
                for k in range(relation_token.shape[1]):
                    relation_feature += self.rln_proj(torch.cat((object_token[sid[:, 0], sid[:, 1]],
                                 object_token[oid[:, 0], oid[:, 1]],
                                 relation_token[batch_ids][:, k, :]), 1))

                relation_feature /= relation_token.shape[1]
            else:
                relation_feature = torch.cat((object_token[sid[:, 0], sid[:, 1]],
                                 object_token[oid[:, 0], oid[:, 1]],
                                 relation_token[batch_ids].flatten(1)), 1)

            if self.rln_freq_bias is not None:
                s_labels = torch.as_tensor(s_labels).to(object_token.device)
                o_labels = torch.as_tensor(o_labels).to(object_token.device)
                freq_dist.append(self.rln_freq_bias( \
                            torch.stack((s_labels, o_labels), 1)
                        ))
            if outputs_t is not None:
                sid_t = torch.as_tensor(sid_t)
                oid_t = torch.as_tensor(oid_t)
                relation_feature_t = torch.cat((outputs_t['hs_obj'][sid_t[:, 0], sid_t[:, 1]],
                                                outputs_t['hs_obj'][oid_t[:, 0], oid_t[:, 1]],
                                                outputs_t['hs_rln'][batch_ids].flatten(1)), 1)

        else: # open-set
            rel_tgt = []
            batch_ids = []
            sid, oid = [], []

            for bid, target in enumerate(targets):
                if len(target['relations']) == 0:
                    continue

                grounded = {}
                cur_num = 0
                for src, dst in zip(indices[bid][0], indices[bid][1]):
                    grounded[dst.item()] = src.item()

                q_nouns = target['gt_names']
                for rel in target['relations']:
                    if rel[0] not in q_nouns or rel[1] not in q_nouns:
                        continue 
                    si = q_nouns.index(rel[0])
                    oi = q_nouns.index(rel[1])

                    rel_tgt.append(rel[2])
                    sid.append([bid, grounded[si]])
                    oid.append([bid, grounded[oi]])
                    cur_num += 1
                    
                # random sample negatives
                n = len(q_nouns)
                full_adj = torch.ones((n, n))-torch.diag(torch.ones(n))
                for rel in target['relations']:
                    if rel[0] not in q_nouns or rel[1] not in q_nouns:
                        continue 
                    si = q_nouns.index(rel[0])
                    oi = q_nouns.index(rel[1])
                    full_adj[si, oi] = 0
                neg_edges = torch.nonzero(full_adj)

                num_neg_per_img = max(1, len(target['relations']) * 3)
                if len(neg_edges) >= num_neg_per_img:
                    idx_ = torch.randperm(neg_edges.shape[0])[:num_neg_per_img]
                    neg_edges = neg_edges[idx_,:]

                for item in neg_edges.tolist():
                    sid.append([bid, grounded[item[0]]])
                    oid.append([bid, grounded[item[1]]])
                    rel_tgt.append('[UNK]')
                    cur_num += 1

                # random sampling from other images in a batch 
                if len(neg_edges) < num_neg_per_img: 
                    for rel in target['relations']:
                        if rel[0] not in q_nouns or rel[1] not in q_nouns:
                            continue 
                        si = q_nouns.index(rel[0])
                        oi = q_nouns.index(rel[1])

                        next_id = bid
                        assert len(targets) >= 1, "batch size must be greather than 1!"
                        if len(targets) == 1:
                            continue

                        while next_id == bid:
                            next_id = random.randint(0, len(targets)-1)

                        rid = random.randint(0, object_token.shape[1]-1)
                        if random.randint(0, 1) == 0:
                            sid.append([bid, grounded[si]])
                            oid.append([next_id,  rid])
                        else:
                            sid.append([next_id, rid])
                            oid.append([bid, grounded[oi]])

                        rel_tgt.append('[UNK]')
                        cur_num += 1

                batch_ids.extend([bid] * cur_num)

            batch_ids = torch.as_tensor(batch_ids)
            sid = torch.as_tensor(sid)
            oid = torch.as_tensor(oid)
            relation_feature = torch.cat((object_token[sid[:, 0], sid[:, 1]],
                                          object_token[oid[:, 0], oid[:, 1]],
                                          relation_token[batch_ids].flatten(1)
                                          ), 1)

        assert len(relation_feature) > 0, "No relation features !"

        # random permute
        _idx_ = torch.randperm(len(relation_feature))
        batch_ids = batch_ids[_idx_]
        relation_feature = relation_feature[_idx_]
        if self.ablation_mode == 'avg_rln':
            pass
        else:
            relation_feature = self.rln_proj(relation_feature)

        if relation_feature_t is not None:
            with torch.no_grad():
                relation_feature_t = self.rln_proj_teacher(relation_feature_t[_idx_])

        if (not relation_wo_labels) and (not relation_is_str):
            all_edge_lbl = all_edge_lbl[_idx_]
        else:
            rel_tgt = [rel_tgt[e] for e in _idx_.tolist()]

        # loss 
        if self.rln_classifier is not None:
            assert not relation_wo_labels, "rln_classifier should be None for open vocabulary !"
            rel_logits = self.rln_classifier(relation_feature)
            if self.rln_freq_bias is not None:
                freq_dist = torch.cat(freq_dist, 0)[_idx_]
                rel_logits += freq_dist



            rel_tgt_onehot = torch.zeros([rel_logits.shape[0], rel_logits.shape[1]],
                                          dtype =rel_logits.dtype, 
                                          layout=rel_logits.layout, 
                                          device=rel_logits.device)

            all_edge_lbl = all_edge_lbl.to(rel_logits.device)
            rel_tgt_onehot.scatter_(-1, all_edge_lbl.unsqueeze(-1), 1)
        else:
            encoded_text = rel_text_dict['encoded_text'][batch_ids]
            text_mask = rel_text_dict['text_token_mask'][batch_ids]
            input_ids = rel_text_dict['input_ids'][batch_ids]


            rel_logits  = torch.einsum("a d, a b d -> a b", relation_feature, encoded_text)
            rel_logits.masked_fill_(~text_mask, float('-inf'))
            # padding to max_text_len
            rel_logits = padding_last(rel_logits, 512) # 512 ? 2048
            rel_tgt_onehot = torch.zeros_like(rel_logits)

            if relation_feature_t is not None and not self.unsupervised_distill:
                with torch.no_grad():
                    encoded_text_t = rel_text_dict_t['encoded_text'][batch_ids]
                    rel_logits_t  = torch.einsum("a d, a b d -> a b", relation_feature_t, encoded_text_t)
                    rel_logits_t.masked_fill_(~text_mask, float('-inf'))
                    rel_logits_t = padding_last(rel_logits_t, rel_logits.shape[-1])

                rel_logits_t = shrink_sigmoid(rel_logits_t, 2.0) # suitable for OvR
                #rel_logits_t.sigmoid_()

            if relation_wo_labels or relation_is_str:
                for ii, name in enumerate(rel_tgt):
                    if name == '[UNK]':
                        continue 

                    ids = self.tokenizer(name + '.').input_ids[1:-1]
                    all_ids = input_ids[ii]

                    start_i, end_i = search_query_pos(all_ids.tolist(), ids)
                    assert start_i != end_i, "cannot find query:{} from input_ids: {}".format(
                            name, self.tokenizer.decode(all_ids))
                    assert start_i < rel_tgt_onehot.shape[1] and end_i < rel_tgt_onehot.shape[1], "start_i:{}, end_i:{}, tgt shape:{}".format(start_i, end_i, rel_tgt_onehot.shape)

                    rel_tgt_onehot[ii, start_i: end_i] = 1.0

            else:
                for ii, label in enumerate(all_edge_lbl.tolist()):
                    name = self.ind_to_predicates[label]
                    if name == '[UNK]' and relation_feature_t is not None and not self.unsupervised_distill: # use teacher's output
                        rel_tgt_onehot[ii] = rel_logits_t[ii]
                        continue 

                    if name == '[UNK]':
                        continue 

                    ids = self.tokenizer(name +'.').input_ids[1:-1]
                    all_ids = input_ids[ii]
                    start_i, end_i = search_query_pos(all_ids.tolist(), ids)
                    assert start_i != end_i, "cannot find query:{} from input_ids".format(ids)

                    rel_tgt_onehot[ii, start_i: end_i] = 1.0



        rel_num = torch.as_tensor(rel_tgt_onehot.shape[0], device=rel_tgt_onehot.device) 
        #rel_num = rel_tgt_onehot.sum()
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(rel_num)
        rel_num = torch.clamp(rel_num / get_world_size(), min=1).item()

                
        if not self.focal_loss_for_edges:  # CE
            loss = F.cross_entropy(rel_logits, all_edge_lbl, reduction='sum') / rel_num
            losses = dict(loss_edges=loss)
        else: # focal loss
            alpha, gamma = 0.25, 2.0
            eps = 1e-5 
            rel_prob = rel_logits.sigmoid().clamp(min=eps, max=1.0-eps)
            rel_mask = (rel_logits != float('-inf')).float()
          
            rel_weight = (rel_tgt_onehot > 0.5).sum(1, keepdim=True)
            rel_weight[rel_weight == 0] = 1.0

            pos_loss = - torch.log(rel_prob) * ((1.0 - rel_prob)** gamma) * rel_tgt_onehot * rel_mask / rel_weight
            neg_loss = - torch.log(1.0 -rel_prob) * (rel_prob**gamma) * (1.0 - rel_tgt_onehot) * rel_mask  / rel_weight

            pos_loss = pos_loss.sum() 
            neg_loss = neg_loss.sum() 


            loss = (pos_loss + neg_loss) / rel_num

            losses = dict(loss_edges=loss)
            with torch.no_grad():
                losses['loss_edges_pos'] = pos_loss.detach() /  rel_num
                losses['loss_edges_neg'] = neg_loss.detach() /  rel_num

            if outputs_t is not None and self.unsupervised_distill:
                ids = torch.where(all_edge_lbl == 0)[0]
                loss_distill = F.l1_loss(relation_feature[ids], relation_feature_t[ids], reduction='sum') / rel_num

                losses['loss_distill'] = loss_distill


        losses['rel_batch'] = torch.as_tensor(rel_num).to(losses['loss_edges'].device)
        if os.environ.get("DEBUG") == '1':
            import pdb; pdb.set_trace()

        return losses

    def _focal_loss(self, logits, tgt_onehot, gamma=2.0, eps=1e-5):
        prob = logits.sigmoid().clamp(min=eps, max=1.0-eps)
        mask = (logits != float('-inf')).float()

        pos_loss = -torch.log(prob) * torch.pow(1.0 - prob, gamma) * tgt_onehot * mask
        neg_loss = -torch.log(1.0-prob) * torch.pow(prob, gamma) * (1 - tgt_onehot) * mask

        num_pos = tgt_onehot.sum()
        if num_pos == 0:
            return neg_loss.sum()

        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos

        return loss


    def forward(self, outputs, targets, outputs_t=None, return_indices=False, 
                global_iter=-1):

        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        input_ids = outputs['input_ids']
        for bid, target in enumerate(targets):
            target['input_ids'] = input_ids[bid]


        self.global_iter = global_iter



        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        device=next(iter(outputs.values())).device


        indices = self.matcher(outputs_without_aux, targets)
        
        if outputs_t is not None:
            outputs_without_aux_t = {k: v for k, v in outputs_t.items() if k != 'aux_outputs'}
            indices_t = self.matcher(outputs_without_aux_t, targets)
        else:
            indices_t = None

        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        # prepare for dn loss
        dn_meta = outputs['dn_meta'] if 'dn_meta' in outputs else None 

        if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            output_known_lbs_bboxes,single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['boxes']) > 0:
                    t = torch.arange(0, len(targets[i]['boxes']) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            output_known_lbs_bboxes=dn_meta['output_known_lbs_bboxes']
            l_dict = {}
            for loss in self.losses:
                if 'edges' == loss:
                    continue 
                if 'obj_likelihood' == loss:
                    continue 

                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes*scalar,**kwargs))

            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            pass

        for loss in self.losses:
            kwargs = {}
            if 'edges' in loss:
                kwargs = {'object_token': outputs['hs_obj'],
                          'relation_token': outputs['hs_rln'],
                          'rel_text_dict': outputs.get('rel_text_dict', None),
                          'rel_text_dict_t': outputs_t.get('rel_text_dict', None) if outputs_t is not None else None 
                          }
                if outputs_t is not None:
                    kwargs['outputs_t'] = outputs_t
                    kwargs['indices_t'] = indices_t

            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                if outputs_t is not None:
                    aux_outputs_t = outputs_t['aux_outputs'][idx]
                    indices_t = self.matcher(aux_outputs_t, targets)

                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    if 'edges' in loss:
                        kwargs = {'object_token': aux_outputs['hs_obj'],
                                  'relation_token': aux_outputs['hs_rln'],
                                  'rel_text_dict': outputs.get('rel_text_dict', None),
                                  'rel_text_dict_t': outputs_t.get('rel_text_dict', None) if outputs_t is not None else None 
                                 }
                        if outputs_t is not None:
                            kwargs['outputs_t'] = aux_outputs_t
                            kwargs['indices_t'] = indices_t

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
                    l_dict={}
                    for loss in self.losses:
                        if 'edges' == loss:
                            continue 

                        kwargs = {}
                        if 'labels' in loss:
                            kwargs = {'log': False}

                        l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes*scalar,
                                                                 **kwargs))

                    l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    pass

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)

            if outputs_t is not None:
                interm_outputs_t = outputs_t['interm_outputs']
                indices_t = self.matcher(interm_outputs_t, targets)

            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                if 'obj_likelihood' == loss:
                    continue 
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}

                if 'edges' in loss:
                    kwargs = {'object_token': interm_outputs['hs_obj'],
                              'relation_token': interm_outputs['hs_rln'],
                              'rel_text_dict': outputs.get('rel_text_dict', None), 
                              'rel_text_dict_t': outputs_t.get('rel_text_dict', None) if outputs_t is not None else None 
                             }
                    if outputs_t is not None:
                        kwargs['outputs_t'] = interm_outputs_t
                        kwargs['indices_t'] = indices_t

                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # enc output loss
        if 'enc_outputs' in outputs:
            for i, enc_outputs in enumerate(outputs['enc_outputs']):
                indices = self.matcher(enc_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    if 'obj_likelihood' == loss:
                        continue 
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    if 'edges' == loss:
                        continue 

                    l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def prep_for_dn(self,dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups,pad_size=dn_meta['num_dn_group'],dn_meta['pad_size']
        assert pad_size % num_dn_groups==0
        single_pad=pad_size//num_dn_groups

        return output_known_lbs_bboxes,single_pad,num_dn_groups
