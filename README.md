# OvSGTR
Official Implementation of "Expanding Scene Graph Boundaries: Fully Open-vocabulary Scene Graph Generation via Visual-Concept Alignment and Retention" (
    [Paper](https://arxiv.org/pdf/2311.10988)
)

![OvSGTR](figures/OvSGTR.png)

## Setup
For simplicity, you can directly run ```bash install.sh```, which includes the following steps: 

0. install pytorch 1.9.1 and other dependencies, e.g., 
```
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html # this might need to be changed due to cuda driver version 

pip install -r requirements.txt
```

1. install GroundingDINO and download pre-trained weights
```
cd GroundingDINO && python3 setup.py install 
```
```
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O $PWD/GroundingDINO/weights/groundingdino_swint_ogc.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth -O $PWD/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth
```



## Dataset 
```
- VG150
- COCO
```
prepare the dataset under the folder ```data``` with the [instruction](data/data.md)

## Closed-set SGG
For training OvSGTR (w. Swin-T) on VG150, running with this command 
```
bash scripts/DINO_train_dist.sh vg ./config/GroundingDINO_SwinT_OGC_full.py  ./data  ./logs/ovsgtr_vg_swint_full ./GroundingDINO/weights/groundingdino_swint_ogc.pth
```
or 
```
bash scripts/DINO_train_dist.sh vg ./config/GroundingDINO_SwinB_full.py  ./data  ./logs/ovsgtr_vg_swinb_full ./GroundingDINO/weights/groundingdino_swinb_cogcoor.pth 
```
for using Swin-B backbone.
you might need to change the default devices of ```CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7```  in the script. 
Notice that the actual batch size = batch size (default 4 in config files) * num gpus. 


![benchmark on Closed-set SGG](figures/closed-sgg.png)

### Checkpoints

<!-- insert a table -->
<table>
  <thead>
    <tr style="text-align: right;">
      <th>backbone</th>
      <th>R@20/50/100</th>
      <th>Checkpoint</th>
      <th>Config</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Swin-T</td>
      <td> 26.97 / 35.82 / 41.38 </td>
      <td><a href="">link</a></td>
      <td>config/GroundingDINO_SwinT_OGC_full.py</td>
    </tr>
    <tr>
      <td>Swin-B</td>
      <td>  </td>
      <td><a href="">link</a>
      <td> </td>
    </tr>
  </tbody>
</table> 

## OvD-SGG 


## OvR-SGG

## OvD+R-SGG



## Acknowledgement
Thank [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) and [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) for their awesome code and models. 


## Citation
Please cite OvSGTR in your publications if it helps your research:
```
@article{chen2023expanding,
  title={Expanding Scene Graph Boundaries: Fully Open-vocabulary Scene Graph Generation via Visual-Concept Alignment and Retention},
  author={Chen, Zuyao and Wu, Jinlin and Lei, Zhen and Zhang, Zhaoxiang and Chen, Changwen},
  journal={arXiv preprint arXiv:2311.10988},
  year={2023}
}
```