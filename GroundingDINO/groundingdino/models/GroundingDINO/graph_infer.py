# modified from https://github.com/suprosanna/relationformer/blob/scene_graph/inference.py
import torch
import numpy as np

from typing import List, Dict
import copy


def graph_infer(outputs : List[Dict], 
                rln_proj, rln_classifier,  
                rln_freq_bias,
                text_dict, 
                name2predicates, 
                tokenizer,
                use_sigmoid=False,
                use_classifier=False,
                save_features=False):
    dst = []
    if rln_freq_bias is not None:
        use_sigmoid = False 

    for batch_id, output in enumerate(outputs):
        obj_token = output['obj_token'] # (#obj, dim)
        rln_token = output['rln_token'] # (#query, dim)

        if not use_classifier and text_dict is not None:
            encoded_text = text_dict['encoded_text'][batch_id]
            text_mask = text_dict['text_token_mask'][batch_id]
            input_ids = text_dict['input_ids'][batch_id]
            sep_idx = [i for i in range(len(input_ids)) if input_ids[i] in [101, 102, 1012]]

        boxes = copy.deepcopy(output['boxes']) #(#obj, 4)
        scores = copy.deepcopy(output['scores']) # (#obj)
        labels = copy.deepcopy(output['labels']) # (#obj)

        node_id = torch.nonzero(labels).squeeze() 

        obj_token = obj_token[node_id]
        pred_classes = labels[node_id]
        pred_cls_score = scores[node_id]

        pred_boxes = boxes[node_id]
        pred_boxes_score = pred_cls_score
        pred_boxes_class = pred_classes


        if node_id.dim() !=0 and node_id.nelement() != 0 and node_id.shape[0]>1:
            # all possible node pairs in all token ordering
            tmp = torch.arange(len(node_id))
            node_pairs = torch.cat((torch.combinations(tmp),
                                    torch.combinations(tmp)[:,[1,0]]),0)

            id_rel = torch.tensor(list(range(len(node_id))))
            node_pairs_rel = torch.cat((torch.combinations(id_rel),torch.combinations(id_rel)[:,[1,0]]),0)

            # feature
            relation_feat = torch.cat((
                                           obj_token[node_pairs[:, 0], :],
                                           obj_token[node_pairs[:, 1], :],
                                           rln_token.flatten().repeat(len(node_pairs), 1), 
                                           ),
                                           dim=1)
            relation_feat = rln_proj(relation_feat)

            if use_classifier:
                relation_logits = rln_classifier(relation_feat)
                if rln_freq_bias is not None:
                    bias = rln_freq_bias( \
                             torch.stack((pred_classes[node_pairs[:, 0]], 
                                          pred_classes[node_pairs[:, 1]]), 1))

                    relation_logits += bias
            else:
                relation_logits = torch.einsum("a d, b d -> a b", relation_feat, encoded_text)
                relation_logits.masked_fill(~text_mask, float('-inf'))


            all_node_pairs = node_pairs_rel.cpu()
            if use_sigmoid:
                relation_prob = relation_logits.sigmoid().detach().cpu()
            else:
                relation_prob = relation_logits.softmax(-1).detach().cpu()
            
            if use_classifier:
                all_relation = relation_prob
            else:
                all_relation = torch.zeros((relation_prob.shape[0], len(name2predicates)))
                for ii in range(1, len(sep_idx)):
                    right_idx = sep_idx[ii]
                    left_idx = sep_idx[ii-1] + 1
                    if left_idx >= right_idx:
                        continue
                    name = tokenizer.decode(input_ids[left_idx:right_idx])
                    all_relation[:, name2predicates[name]] = relation_prob[:, left_idx:right_idx].mean(-1)



            # sort by score: relation score * subject score * object score
            rel_score = all_relation[:, 1:].max(1)[0]

            obj_score0 = pred_boxes_score[all_node_pairs[:, 0]]
            obj_score1 = pred_boxes_score[all_node_pairs[:, 1]]
            rel_score = rel_score.to(obj_score0.device) * obj_score0 * obj_score1

            rel_idx = rel_score.sort(descending=True)[1].to(all_relation.device)
            all_relation = all_relation[rel_idx]
            all_node_pairs = all_node_pairs[rel_idx]

        else:
            assert node_id.nelement() == 1, "#obj != 1"

            print("Warning: #obj==1!")
            all_node_pairs = torch.zeros(1, 2).long()
            all_relation = torch.zeros(1, 51) # 
            relation_feat = None
            pred_boxes = pred_boxes.view(1, -1).repeat(2, 1)
            pred_boxes_score = pred_boxes_score.view(1, -1).repeat(2, 1)
            pred_boxes_class = pred_boxes_class.view(1, -1).repeat(2, 1)
            #pred_boxes_score.fill_(0.)

        out = {}
        if all_relation is not None:
            out['node_id'] = node_id.cpu()
            out['pred_boxes'] = pred_boxes.cpu()
            out['pred_boxes_score'] = pred_boxes_score.cpu()
            out['pred_boxes_class'] = pred_boxes_class.cpu()

            out['all_node_pairs'] = all_node_pairs
            out['all_relation'] = all_relation
            if save_features and relation_feat is not None:
                out['rln_features'] = relation_feat.data.cpu()

        dst.append(out)

    return dst
