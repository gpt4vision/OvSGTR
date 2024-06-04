# -*- coding: utf-8 -*-
'''
@File    :   visualizer.py
@Time    :   2022/04/05 11:39:33
@Author  :   Shilong Liu 
@Contact :   liusl20@mail.tsinghua.edu.cn; slongliu86@gmail.com
Modified from COCO evaluator
'''

import os, sys
from textwrap import wrap
import torch
import numpy as np
import cv2
import datetime

import copy
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from pycocotools import mask as maskUtils
from matplotlib import transforms
import networkx as nx

def renorm(img: torch.FloatTensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) \
        -> torch.FloatTensor:
    # img: tensor(3,H,W) or tensor(B,3,H,W)
    # return: same as img
    assert img.dim() == 3 or img.dim() == 4, "img.dim() should be 3 or 4 but %d" % img.dim() 
    if img.dim() == 3:
        assert img.size(0) == 3, 'img.size(0) shoule be 3 but "%d". (%s)' % (img.size(0), str(img.size()))
        img_perm = img.permute(1,2,0)
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        img_res = img_perm * std + mean
        return img_res.permute(2,0,1)
    else: # img.dim() == 4
        assert img.size(1) == 3, 'img.size(1) shoule be 3 but "%d". (%s)' % (img.size(1), str(img.size()))
        img_perm = img.permute(0,2,3,1)
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        img_res = img_perm * std + mean
        return img_res.permute(0,3,1,2)

class ColorMap():
    def __init__(self, basergb=[255,255,0]):
        self.basergb = np.array(basergb)
    def __call__(self, attnmap):
        # attnmap: h, w. np.uint8.
        # return: h, w, 4. np.uint8.
        assert attnmap.dtype == np.uint8
        h, w = attnmap.shape
        res = self.basergb.copy()
        res = res[None][None].repeat(h, 0).repeat(w, 1) # h, w, 3
        attn1 = attnmap.copy()[..., None] # h, w, 1
        res = np.concatenate((res, attn1), axis=-1).astype(np.uint8)
        return res


class COCOVisualizer():
    def __init__(self) -> None:
        self._colors = None
        pass

    def visualize(self, img, tgt, caption=None, colors=None, dpi=300, savedir=None, show_in_console=True, title=None, close_axis=False, relations=None):
        """
        img: tensor(3, H, W)
        tgt: make sure they are all on cpu.
            must have items: 'image_id', 'boxes', 'size'
        """
        plt.figure(dpi=dpi)
        plt.rcParams['font.size'] = '8'

        if relations is not None:
            fig, axs = plt.subplots(2, 1, figsize=(12, 8))
            ax = axs[0]
            axr = axs[1]
        else:
            ax = plt.gca()

        img = renorm(img).permute(1, 2, 0)
        ax.imshow(img)
        self.addtgt(ax, tgt, colors)

        if relations is not None:
            #fig, ax = plt.subplots(figsize=(10, 6))
            G = nx.DiGraph()

            # Extract nodes and edges from relation tuples
            edges = [tuple(r[:2]) for r in relations]
            nodes = list(np.unique(np.array(edges)))

            # Create dictionaries for labels
            labeldict = {node: node for node in nodes}
            #edge_labeldict = {(edge[0], edge[1]): f"{edge[2]} ({edge[3]:.2f})" for edge in relations}
            edge_labeldict = {(edge[0], edge[1]): f"{edge[2]}" for edge in relations}

            # Add nodes and edges to the graph
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)

            # Get positions for nodes
            pos = nx.circular_layout(G, scale=0.35)


            # Draw the graph
            d = nx.degree(G)

            nx.draw(G, pos, ax=axr, labels=labeldict,
                    node_size=[len(i[1]) * 60 for i in labeldict.items()],
                    node_color='lightcoral', edge_color='mediumorchid',
                    width=1.5,  font_size=8, with_labels=True)

            # Draw edge labels
            nx.draw_networkx_edge_labels(G, pos, ax=axr, edge_labels=edge_labeldict,
                                         label_pos=0.5, font_size=8, rotate=False)



        if title is not None:
            plt.title(title)

        if close_axis:
            ax.axis('off')

        if show_in_console:
            plt.show()

        if savedir is not None:
            if caption is None:
                #savename = '{}/{}-{}.png'.format(savedir, int(tgt['image_id']), str(datetime.datetime.now()).replace(' ', '-'))
                savename = '{}/{}.jpg'.format(savedir, int(tgt['image_id']) )
            else:
                savename = '{}/{}-{}-{}.png'.format(savedir, caption, int(tgt['image_id']), str(datetime.datetime.now()).replace(' ', '-'))
            print("savename: {}".format(savename))
            os.makedirs(os.path.dirname(savename), exist_ok=True)
            plt.savefig(savename, bbox_inches='tight', pad_inches=0)
        plt.close()

    def addtgt(self, ax, tgt, colors=None):
        """
        - tgt: dict. args:
            - boxes: num_boxes, 4. xywh, [0,1].
            - box_label: num_boxes.
        """
        assert 'boxes' in tgt
        #ax = plt.gca()
        H, W = tgt['size'].tolist() 
        numbox = tgt['boxes'].shape[0]

        color = []
        polygons = []
        boxes = []
        for ii, box in enumerate(tgt['boxes'].cpu()):
            unnormbbox = box * torch.Tensor([W, H, W, H])
            unnormbbox[:2] -= unnormbbox[2:] / 2
            [bbox_x, bbox_y, bbox_w, bbox_h] = unnormbbox.tolist()
            boxes.append([bbox_x, bbox_y, bbox_w, bbox_h])
            poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
            np_poly = np.array(poly).reshape((4,2))
            polygons.append(Polygon(np_poly))
            if colors is None:
                c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
            else:
                if ii >= len(colors):
                    colors = list(colors)
                    c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
                    colors.append(c)

                c = colors[ii]

            color.append(c)
        
        self._colors = copy.deepcopy(color) 
        p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.1)
        ax.add_collection(p)
        p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
        ax.add_collection(p)


        if 'box_label' in tgt:
            assert len(tgt['box_label']) == numbox, f"{len(tgt['box_label'])} = {numbox}, "
            for idx, bl in enumerate(tgt['box_label']):
                _string = str(bl)
                bbox_x, bbox_y, bbox_w, bbox_h = boxes[idx]
                # ax.text(bbox_x, bbox_y, _string, color='black', bbox={'facecolor': 'yellow', 'alpha': 1.0, 'pad': 1})
                ax.text(bbox_x, bbox_y, _string, color='black', bbox={'facecolor': color[idx], 'alpha': 0.6, 'pad': 1})

        if 'caption' in tgt:
            ax.set_title(tgt['caption'], wrap=True)


