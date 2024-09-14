# ------------------------------------------------------------------------
# modified from Grounding DINO
# url: https://github.com/gpt4vision/OvSGTR
# Copyright (c) 2024 Joseph Chen. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# 
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import os
import copy
from typing import List

import math
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import batched_nms

import torch.distributed as dist

from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.misc import (
    NestedTensor,
    accuracy,
    get_world_size,
    interpolate,
    inverse_sigmoid,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
    all_gather 
)

from groundingdino.util.utils import get_phrases_from_posmap
from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span



from ..registry import MODULE_BUILD_FUNCS
from .backbone import build_backbone
from .bertwarper import (
    BertModelWarper,
    generate_masks_with_special_tokens,
    generate_masks_with_special_tokens_and_transfer_map,
)
from .transformer import build_transformer
from .utils import MLP, ContrastiveEmbed 

from .utils import FrequencyBias

# 
import bisect
from .matcher import build_matcher 
from .losses import SetCriterion 
from .dn_components import prepare_for_cdn,dn_post_process

from .graph_infer import graph_infer

MAX_WORDS_LEN = 151





class GroundingDINO(nn.Module):
    """This is the Cross-Attention Detector module that performs object detection"""
    def __init__(
        self,
        backbone,
        transformer,
        num_queries,
        aux_loss=False,
        iter_update=False,
        query_dim=2,
        num_feature_levels=1,
        nheads=8,
        # two stage
        two_stage_type="no",  # ['no', 'standard']
        dec_pred_bbox_embed_share=True,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=True,
        num_patterns=0,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        dn_labelbook_size=100,
        text_encoder_type="bert-base-uncased",
        sub_sentence_present=True,
        max_text_len=256,
        num_classes=2000,
        # sgg
        do_sgg=False,
        sgg_mode="full",
        num_rln_cat=51,
        rln_pretraining=False,
        num_rln_queries=1,
        rln_freq_bias=None,
        focal_loss_for_edges=False,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.max_text_len = max_text_len
        self.sub_sentence_present = sub_sentence_present
        self.do_sgg = do_sgg
        self.rln_pretraining = rln_pretraining
        self.sgg_mode = sgg_mode


        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size
        self.label_enc = nn.Embedding(dn_labelbook_size+1, hidden_dim)

        # bert
        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)

        self.bert = get_tokenlizer.get_pretrained_language_model(text_encoder_type)
        self.bert.pooler.dense.weight.requires_grad_(False)
        self.bert.pooler.dense.bias.requires_grad_(False)
        self.bert = BertModelWarper(bert_model=self.bert)


        self.feat_map = nn.Linear(self.bert.config.hidden_size, self.hidden_dim, bias=True)
        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)
       	
        # sgg 
        if self.do_sgg:
            if self.sgg_mode != 'full':
                self.rln_text_proj = nn.Linear(self.bert.config.hidden_size, self.hidden_dim, bias=True)
            else:
                self.rln_text_proj = None

        # special tokens
        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == "no", "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = ContrastiveEmbed(max_text_len=self.max_text_len) # 2048


        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)
            ]
        class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        if self.do_sgg:
            self.num_rln_queries = num_rln_queries
            self.rln_embed = nn.Embedding(self.num_rln_queries, self.hidden_dim*2)
            self.transformer.decoder.rln_embed = self.rln_embed 

            extra_dim = 0 #if not self.rln_add_coords else self.hidden_dim
            input_dim = self.hidden_dim*3 + extra_dim

            if self.sgg_mode == 'full':
                self.rln_proj = MLP(input_dim, 2048, self.hidden_dim, 2) # 2048->1024
                self.rln_classifier = nn.Linear(self.hidden_dim, num_rln_cat)

                self.rln_freq_bias = FrequencyBias(rln_freq_bias) if rln_freq_bias else None
                if rln_freq_bias:
                    nn.init.constant_(self.rln_classifier.bias, 0.)
                elif focal_loss_for_edges:
                    prior_prob = 0.1
                    bias_value = -math.log((1 - prior_prob) / prior_prob)

                    self.rln_classifier.weight.data.fill_(0.) 
                    nn.init.constant_(self.rln_classifier.bias, bias_value)

            else:
                self.rln_proj = MLP(input_dim, 2048, self.hidden_dim, 2)
                self.rln_classifier = None 
        else:
            self.rln_proj = None
            self.rln_classifier = None 

        # two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)


    def encode_captions(self, captions, device, encode_relation=False):
        text_dict = {
            "encoded_text": [], #encoded_text,  # bs, 195, d_model
            "text_token_mask": [], #text_token_mask,  # bs, 195
            "position_ids": [], #position_ids,  # bs, 195
            "text_self_attention_masks": [], #text_self_attention_masks,  # bs, 195,195
            "input_ids": []
        }
        cap_len = [len(e[:-1].split('.')) for e in captions]
        captions = [e[:-1].split('.') for e in captions]

        for ii in range((max(cap_len) + MAX_WORDS_LEN-1) // MAX_WORDS_LEN):
            sub_captions = [e[ii*MAX_WORDS_LEN: (ii+1)*MAX_WORDS_LEN] for e in captions]
            sub_captions = [('. '.join(e) + '. ') for e in sub_captions]
            
            # encoder texts
            tokenized = self.tokenizer(sub_captions, padding="longest", return_tensors="pt").to(
                device
            )

            (
                text_self_attention_masks,
                position_ids,
                cate_to_token_mask_list,
            ) = generate_masks_with_special_tokens_and_transfer_map(
                tokenized, self.specical_tokens, self.tokenizer
            )

            if text_self_attention_masks.shape[1] > self.max_text_len:
                print("Warning: text_self_attention_masks.shape[1]:{} > max_text_len:{}".format(
                      text_self_attention_masks.shape[1], self.max_text_len))

                text_self_attention_masks = text_self_attention_masks[
                    :, : self.max_text_len, : self.max_text_len
                ]
                position_ids = position_ids[:, : self.max_text_len]
                tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
                tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
                tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]

            # extract text embeddings
            if self.sub_sentence_present:
                tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
                tokenized_for_encoder["attention_mask"] = text_self_attention_masks
                tokenized_for_encoder["position_ids"] = position_ids
            else:
                tokenized_for_encoder = tokenized

            bert_output = self.bert(**tokenized_for_encoder)  # bs, 195, 768

            if encode_relation:
                assert self.rln_text_proj is not None, "rln_text_proj cannot be None !"
                encoded_text = self.rln_text_proj(bert_output["last_hidden_state"])  # bs, 195, d_model
            else:
                encoded_text = self.feat_map(bert_output["last_hidden_state"])  # bs, 195, d_model


            text_token_mask = tokenized.attention_mask.bool()  # bs, 195
            # text_token_mask: True for nomask, False for mask
            # text_self_attention_masks: True for nomask, False for mask

            if encoded_text.shape[1] > self.max_text_len:
                print("Warning: encoded_text.shape[1]:{} > max_text_len:{}".format(encoded_text.shape[1],
                        self.max_text_len))
                encoded_text = encoded_text[:, : self.max_text_len, :]
                text_token_mask = text_token_mask[:, : self.max_text_len]
                position_ids = position_ids[:, : self.max_text_len]
                text_self_attention_masks = text_self_attention_masks[
                    :, : self.max_text_len, : self.max_text_len
                ]

            text_dict['encoded_text'].append(encoded_text)
            text_dict['text_token_mask'].append(text_token_mask)
            text_dict['position_ids'].append(position_ids)
            text_dict['text_self_attention_masks'].append(text_self_attention_masks)
            text_dict['input_ids'].append(tokenized.input_ids)

        text_dict['encoded_text'] = torch.cat(text_dict['encoded_text'], 1)
        text_dict['text_token_mask'] = torch.cat(text_dict['text_token_mask'], 1)
        text_dict['position_ids'] = torch.cat(text_dict['position_ids'], 1)
        text_dict['input_ids'] = torch.cat(text_dict['input_ids'], 1)
        attn_mask = torch.zeros((text_dict['encoded_text'].shape[0], 
                                 text_dict['encoded_text'].shape[1],
                                 text_dict['encoded_text'].shape[1]), 
                                 device=device, dtype=torch.bool)

        last_idx = 0
        for ii, mask in enumerate(text_dict['text_self_attention_masks']):
            attn_mask[:, last_idx: last_idx + mask.shape[1], last_idx: last_idx+mask.shape[1]] = \
                    mask
            last_idx += mask.shape[1]

        text_dict['text_self_attention_masks'] = attn_mask
        return text_dict


    def forward(self, samples: NestedTensor, targets: List = None, **kw):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x num_classes]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, width, height). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if targets is None:
            captions = kw["captions"]
            rel_captions = kw['rel_captions'] if 'rel_captions' in kw else []
        else:
            captions = [t["caption"] for t in targets]
            rel_captions = [t['rel_caption'] for t in targets] if 'rel_caption' \
                            in targets[0] else [] 


        # text features
        text_dict = self.encode_captions(captions, samples.device, encode_relation=False)

        if self.do_sgg and len(rel_captions) == 0 and self.sgg_mode != 'full':
            raise Exception("rel_caption cannot be None !")

        rel_text_dict = None 
        if self.do_sgg and self.sgg_mode != 'full':
            rel_text_dict = self.encode_captions(rel_captions, samples.device, encode_relation=True)

        concat_rel_text =  True #if os.environ.get("DEBUG") == '1' else True 
        if rel_text_dict is not None and concat_rel_text:
            sep_len = (text_dict['encoded_text'].shape[1], rel_text_dict['encoded_text'].shape[1])

            for k in text_dict.keys():
                if k != 'text_self_attention_masks':
                    text_dict[k] = torch.cat((text_dict[k], rel_text_dict[k]), 1)

            attn_mask = torch.zeros((text_dict['encoded_text'].shape[0], 
                                 text_dict['encoded_text'].shape[1],
                                 text_dict['encoded_text'].shape[1]), 
                                 device=samples.device, dtype=torch.bool)
            attn_mask[:, :sep_len[0], :sep_len[0]] = text_dict['text_self_attention_masks']
            attn_mask[:, sep_len[0]:, sep_len[0]:] = rel_text_dict['text_self_attention_masks']
            text_dict['text_self_attention_masks'] = attn_mask

        # visual features
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
            backbone_feature = None 


        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        # dn part 
        use_dn = self.dn_number > 0 and targets is not None 
        if use_dn:
            input_query_label, input_query_bbox, attn_mask, dn_meta =\
                prepare_for_cdn(dn_args=(targets, self.dn_number, self.dn_label_noise_ratio, self.dn_box_noise_scale),
                                training=self.training,num_queries=self.num_queries,num_classes=self.num_classes,
                                hidden_dim=self.hidden_dim, label_enc=self.label_enc)
        else:
            #assert targets is None
            input_query_bbox = input_query_label = attn_mask = dn_meta = None        


        hs, hs_rln, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
            srcs, masks, input_query_bbox, poss, 
            input_query_label, attn_mask, text_dict,
        )

        # split via sep_len
        if rel_text_dict is not None and concat_rel_text:
            rel_text_dict = {}
            for k in text_dict.keys():
                if k != 'text_self_attention_masks' and k != 'sep_len':
                    tmp = text_dict[k].split(sep_len, 1)
                    text_dict[k] = tmp[0]
                    rel_text_dict[k] = tmp[1]

            rel_text_dict['text_self_attention_masks'] = text_dict['text_self_attention_masks'][:, sep_len[0]:, sep_len[0]:]
            text_dict['text_self_attention_masks'] = text_dict['text_self_attention_masks'][:, :sep_len[0], :sep_len[0]]


        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        # output
        outputs_class = torch.stack(
            [
                layer_cls_embed(layer_hs, text_dict)
                for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
            ]
        )
        if self.dn_number > 0 and dn_meta is not None:
            hs_obj = [e[:, dn_meta['pad_size']:, :] for e in hs]
            outputs_class, outputs_coord_list = dn_post_process( \
                            outputs_class, outputs_coord_list,
                            dn_meta, self.aux_loss, self._set_aux_loss)
        else:
            hs_obj = hs


        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord_list[-1]}

        # for intermediate outputs
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list)


        # for encoder output
        if hs_enc is not None:
            # prepare intermediate outputs
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1], text_dict)
            out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord }
            out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': init_box_proposal, 
                                                      }

        if dn_meta is not None:
            out['dn_meta'] = dn_meta

        out['input_ids'] = text_dict['input_ids'] #tokenized['input_ids']
        if self.do_sgg:
            out['hs_rln'] = hs_rln[-1]
            out['hs_obj'] = hs_obj[-1]
            out['rel_text_dict'] = rel_text_dict

            if 'interm_outputs' in out:
                out['interm_outputs']['hs_obj'] = hs_obj[-1]
                out['interm_outputs']['hs_rln'] = hs_rln[-1]

            if self.aux_loss:
                for idx, item in enumerate(out['aux_outputs']):
                    item['hs_obj'] = hs_obj[idx]
                    item['hs_rln'] = hs_rln[idx]

        
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_select=100, nms_iou_threshold=-1, 
                 do_sgg=False, relation_thresh=0.5,
                 test_overlap=False, use_gt_box=False,
                 **kwargs
                 ) -> None:
        super().__init__()
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold
        self.do_sgg = do_sgg
        self.relation_thresh = relation_thresh
        self.test_overlap = test_overlap
        self.use_gt_box = use_gt_box 
        self.rln_proj = kwargs.get("rln_proj", None)
        self.detections_per_img = kwargs.get("detections_per_img", 100)
        self.score_threshold = kwargs.get("score_threshold", 0)
        self.temperature = kwargs.get("temperature", 1)

        self.rln_classifier = kwargs.get("rln_classifier", None)
        self.rln_freq_bias = kwargs.get("rln_freq_bias", None)

        #
        self.use_text_labels = kwargs.get("use_text_labels", False)
        self.tokenizer = kwargs.get("tokenizer", None)
        self.name2classes = kwargs.get("name2classes", None)
        self.name2predicates = kwargs.get("name2predicates", None)
        self.max_text_len = kwargs.get("max_text_len", 2048)
        self.matcher = kwargs.get("matcher", None)

        if self.use_text_labels:
            assert self.tokenizer is not None, " tokenzier should not be None when use text labels !"
            self.text_threshold = 1e-3
            self._positive_map = None



    def get_positive_map(self, max_text_len=2048, cat_list=None):
        if self._positive_map is not None:
            return self._positive_map

        if cat_list is None:
            assert self.name2classes is not None, "self.name2classes should not be None!"
            cat_list = [e for e in self.name2classes.keys()]
            use_name2class = True
        else:
            use_name2class = False 

        captions, cat2tokenspan = build_captions_and_token_span(cat_list, True)
        tokenspanlist = [cat2tokenspan[cat] for cat in cat_list]
        positive_map = create_positive_map_from_span(
                    self.tokenizer(captions), tokenspanlist, max_text_len)  # N, dim. normed            
        
        if use_name2class:
            max_label = max(self.name2classes.values()) + 1
            new_pos_map = torch.zeros((max_label, positive_map.shape[1]))
            for ii, (name, label) in enumerate(self.name2classes.items()):
                new_pos_map[label] = positive_map[ii]
        else:
            new_pos_map = positive_map

        self._positive_map = new_pos_map        
        return self._positive_map

    
    def __repr__(self):
        return f"{self.__class__.__name__}(num_select={self.num_select},\n\t  nms_iou_threshold={self.nms_iou_threshold}, \n\t score_threshold={self.score_threshold},\n\t detections_per_img={self.detections_per_img},\n\t do_sgg={self.do_sgg},  relation_thresh={self.relation_thresh}, \n\t test_overlap={self.test_overlap}, \n\t use_gt_box={self.use_gt_box}, \n\t use_text_labels={self.use_text_labels}, \n\t max_text_len={self.max_text_len})"


    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False, test=False, cat_list=None, gt_dicts=None):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        device = out_logits.device

        fake_predcls = False 
        if gt_dicts is not None:
            fake_predcls = True 
            assert outputs['hs_obj'].shape[0] == 1 and len(gt_dicts), 'predcls only supports batch_size=1 !'


        if self.do_sgg:
            obj_token = outputs['hs_obj']
            rln_token = outputs['hs_rln']


        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        batch_size = out_logits.shape[0]
        if self.use_text_labels:
            pos_maps = self.get_positive_map(self.max_text_len, cat_list).to(prob.device)
            prob_to_label = prob @ pos_maps.T
            prob =  prob_to_label

            num_cat = prob.shape[2]
            topk_values, topk_indexes = torch.topk(prob.view(batch_size, -1), num_select, dim=1)
            scores = topk_values
            topk_boxes = topk_indexes // num_cat
            labels = topk_indexes % num_cat

            out_bbox = torch.gather(out_bbox, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
            boxes = out_bbox
            if self.do_sgg:
                dim = obj_token.shape[-1]
                obj_token = torch.gather(obj_token, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, dim))

            # prob_to_token -> prob_to_label
            """
            boxes = torch.zeros((batch_size, num_select, 4), device=prob.device)
            labels = torch.zeros((batch_size, num_select), dtype=torch.long)
            scores = torch.zeros((batch_size, num_select))
            if self.do_sgg:
                topk_obj_token = torch.zeros((batch_size, num_select, 
                                  obj_token.shape[-1]), device=obj_token.device)
            for bid in range(batch_size):
                tokenized = {'input_ids': outputs['input_ids'][bid]}
                sep_idx = [i for i in range(len(tokenized['input_ids'])) 
                              if tokenized['input_ids'][i] in [101, 102, 1012]]


                phrases = []
                for logit in prob[bid]:
                    max_idx = logit.argmax()
                    insert_idx = bisect.bisect_left(sep_idx, max_idx)
                    right_idx = sep_idx[insert_idx]
                    left_idx = sep_idx[insert_idx - 1]
                    phrases.append(get_phrases_from_posmap(logit > self.text_threshold, 
                                                           tokenized, self.tokenizer, 
                                                           left_idx, right_idx))
         
                score = prob[bid].max(-1)[0]
                # top-k 
                topk_score, idx = torch.topk(score, num_select, -1)
                topk_phrases = [phrases[e] for e in idx.cpu().tolist()]

                scores[bid] = topk_score
                boxes[bid] = out_bbox[bid][idx]
                labels[bid] = torch.as_tensor([self.name2classes[e] if e in self.name2classes else 0  for e in topk_phrases])
                if self.do_sgg:
                    topk_obj_token[bid] = obj_token[bid][idx]

            if self.do_sgg:
                obj_token = topk_obj_token
            """
        else:
            num_cat = prob.shape[2]
            topk_values, topk_indexes = torch.topk(prob.view(batch_size, -1), num_select, dim=1)
            scores = topk_values
            topk_boxes = topk_indexes // num_cat
            labels = topk_indexes % num_cat

            out_bbox = torch.gather(out_bbox, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
            boxes = out_bbox
            if self.do_sgg:
                dim = obj_token.shape[-1]
                obj_token = torch.gather(obj_token, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, dim))

        # 
        if not_to_xyxy:
            pass
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(boxes)


        if test:
            print("PostProcess test=True!")
            assert not not_to_xyxy
            boxes[:,:,2:] = boxes[:,:,2:] - boxes[:,:,:2]

        
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        if fake_predcls:
            boxes = gt_dicts[0]['gt_boxes'].unsqueeze(0) 
            boxes = box_ops.box_cxcywh_to_xyxy(boxes)
            boxes = boxes * scale_fct[:, None, :]

            labels = gt_dicts[0]['gt_labels'].unsqueeze(0)
            scores = torch.ones_like(labels).float()

            src_ids, dst_ids = gt_dicts[0]['ids']
            tmp = [-1] * len(dst_ids)
            for src_, dst_ in zip(src_ids.tolist(), dst_ids.tolist()):
                tmp[dst_] = src_
            tmp = torch.as_tensor(tmp)
            obj_token = outputs['hs_obj'][:, tmp]


        # nms if required
        if self.nms_iou_threshold > 0:
            item_indices = [batched_nms(b, s, l, iou_threshold=self.nms_iou_threshold) for b,s,l in zip(boxes, scores, labels)]

            item_indices = [e[:self.detections_per_img] for e in item_indices]

            if self.do_sgg:
                results = [{'scores': s[i], 'labels': l[i], 
                           'boxes': b[i], 'obj_token': ot[i]} for s, l, b, ot, i in zip(
                               scores, labels, boxes, obj_token, item_indices)]
            else:
                results = [{'scores': s[i], 'labels': l[i], 'boxes': b[i]} for s, l, b, i in zip(scores, labels, boxes, item_indices)]
        else:
            
            if self.do_sgg:
                results = [{'scores': s[:self.detections_per_img], 
                            'labels': l[:self.detections_per_img], 
                            'boxes': b[:self.detections_per_img], 
                            'obj_token': ot[:self.detections_per_img]} for s, l, b, ot in zip(scores, labels, boxes, obj_token)]
            else:
                results = [{'scores': s[:self.detections_per_img], 
                            'labels': l[:self.detections_per_img], 
                            'boxes': b[:self.detections_per_img]} for s, l, b in zip(scores, labels, boxes)]

        if self.score_threshold > 0:
            def _score_filter(result, threshold):
                keep_idx = result['scores'] >  threshold
                for k in result.keys():
                    result[k] = result[k][keep_idx]

                return result 

            results = [_score_filter(res, self.score_threshold) for res in results]

        # sgg prediction
        if self.do_sgg:
            _ = [result.update({'rln_token': rln_token[batch_id]}) 
                    for batch_id, result in enumerate(results)]

            _ = [result.update({'scale_fct': scale_fct[batch_id]}) 
                    for batch_id, result in enumerate(results)]
                     

            sgg_out = graph_infer(results, self.rln_proj, self.rln_classifier, 
                                  self.rln_freq_bias,
                                  outputs['rel_text_dict'],
                                  self.name2predicates, 
                                  self.tokenizer, 
                                  use_sigmoid=True,
                                  use_classifier=self.rln_classifier is not None,
                                  save_features=False
                                  )

            for batch, res in enumerate(results):
                res.update({'graph': sgg_out[batch]})
                del res['rln_token']
                del res['obj_token']

        return results

@MODULE_BUILD_FUNCS.registe_with_name(module_name="groundingdino")
def build_groundingdino(args):
    backbone = build_backbone(args)
    frozen_backbone = getattr(args, "frozen_backbone", False)
    if frozen_backbone:
        print("*"*10, " Freeze backbone !", "*"*10)
        for name, p in backbone.named_parameters():
            p.requires_grad = False 

    transformer = build_transformer(args)

    dn_labelbook_size = args.dn_labelbook_size
    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    sub_sentence_present = args.sub_sentence_present


    do_sgg = getattr(args, "do_sgg", False)
    rln_pretraining = getattr(args, "rln_pretraining", False)
    sgg_mode =getattr(args, "sgg_mode", "full")
    num_rln_cat = getattr(args, "num_rln_cat", 51)

    rln_freq_bias = getattr(args, "rln_freq_bias", None)
    if rln_freq_bias is not None:
        rln_freq_bias = torch.load(rln_freq_bias)


    model = GroundingDINO(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        num_patterns=args.num_patterns,
        dn_number=args.dn_number,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
        text_encoder_type=args.text_encoder_type,
        sub_sentence_present=sub_sentence_present,
        max_text_len=args.max_text_len,
        do_sgg=do_sgg,
        sgg_mode=sgg_mode, 
        num_rln_cat=num_rln_cat,
        rln_pretraining=rln_pretraining,
        rln_freq_bias=rln_freq_bias,
        num_rln_queries=getattr(args, "num_rln_queries", 1),
        focal_loss_for_edges=getattr(args, "focal_loss_for_edges", False),
    )

    # matcher
    matcher = build_matcher(args, tokenizer=model.tokenizer)

    # loss fn
    # prepare weight dict
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef

    if do_sgg:
        weight_dict['loss_edges'] = args.edge_loss_coef
        weight_dict['loss_edges_pos'] = 1.0
        weight_dict['loss_edges_neg'] = 1.0
        weight_dict['loss_edges_fg'] = 1.0
        weight_dict['loss_distill'] = getattr(args, "distill_loss_coef", 0.1)


    clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

    
    # for DN training
    if args.use_dn:
        weight_dict['loss_ce_dn'] = args.cls_loss_coef
        weight_dict['loss_bbox_dn'] = args.bbox_loss_coef
        weight_dict['loss_giou_dn'] = args.giou_loss_coef

    use_masks = getattr(args, "masks", False)
    if use_masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    clean_weight_dict = copy.deepcopy(weight_dict)

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in clean_weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if args.two_stage_type != 'no':
        interm_weight_dict = {}
        try:
            no_interm_box_loss = args.no_interm_box_loss
        except:
            no_interm_box_loss = False
        _coeff_weight_dict = {
            'loss_ce': 1.0,
            'loss_bbox': 1.0 if not no_interm_box_loss else 0.0,
            'loss_giou': 1.0 if not no_interm_box_loss else 0.0,
            'loss_edges': 1.0,
            'loss_edges_pos': 1.0,
            'loss_edges_neg': 1.0,
            'loss_edges_fg': 1.0,
            'loss_obj_ll': 1.0,
            'loss_distill': 1.0, 
        }
        try:
            interm_loss_coef = args.interm_loss_coef
        except:
            interm_loss_coef = 1.0
        interm_weight_dict.update({k + f'_interm': v * interm_loss_coef * _coeff_weight_dict[k] for k, v in clean_weight_dict_wo_dn.items()})
        weight_dict.update(interm_weight_dict)

    losses = ['labels', 'boxes' ]
    if use_masks:
        losses += ["masks"]
    if do_sgg:
        losses += ["edges"]



    num_classes = getattr(args, "num_classes", 2000)
    
    has_bbox_supervision = getattr(args, "has_bbox_supervision", True)
    if not has_bbox_supervision:
        losses.remove("boxes")
        print("Attention please : No bbox supervision !")
        
        freeze_bbox = []
        for name, p in model.named_parameters():
            if 'bbox_embed' in name:
                p.requires_grad = False
                freeze_bbox.append(name)
        print("*"*10, " freeze bbox:", freeze_bbox)



    criterion = SetCriterion(matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, losses=losses,
                             rln_pretraining=rln_pretraining,
                             tokenizer=model.tokenizer, 
                             focal_loss_for_edges=getattr(args, "focal_loss_for_edges", False),
                             rel_proposals_threshold=getattr(args, "rel_proposals_threshold", 0.5),
                             rel_proposals_threshold_enabled=getattr(args, "rel_proposals_threshold_enabled", False),
                             unsupervised_distill=getattr(args, "unsupervised_distill", False),
                             fix_rel_batch=getattr(args, "fix_rel_batch", False),
                             rel_batch_per_image=getattr(args, "rel_batch_per_image", 64),
                             )

    # post 
    use_text_labels = getattr(args, "use_text_labels", True)
    postprocessors = {'bbox': PostProcess(num_select=args.num_select, 
                        nms_iou_threshold=args.nms_iou_threshold,
                        do_sgg=do_sgg,
                        use_text_labels=use_text_labels,
                        tokenizer = model.tokenizer,
                        detections_per_img=getattr(args, "detections_per_img", 100),
                        temperature=getattr(args, "obj_temp", 1.0) / args.hidden_dim,
                        max_text_len=getattr(args, "max_text_len", 2048),
                        use_gt_box=getattr(args, "use_gt_box", False)
                    )}
    if postprocessors['bbox'].use_gt_box:
        print("*"*10, " PostProcessing use GT Boxes !")
        postprocessors['bbox'].matcher = matcher

    device = torch.device(args.device)
    postprocessors['bbox'].to(device)

    # frozen bert
    for name, p in model.named_parameters():
        if 'bert' in name:
            p.requires_grad = False 

    if args.dn_number <= 0:
        model.label_enc.weight.requires_grad = False 
        print("disable label_enc !")


    # only feat_map and new added parameters will be trainable
    if getattr(args, "frozen_old_params", False):
        for name, p in model.named_parameters():
            if 'rln' in name or 'relation' in name or 'dropout12' in name or 'norm12' in name:
                continue
            p.requires_grad = False 

    

    return model, criterion, postprocessors
