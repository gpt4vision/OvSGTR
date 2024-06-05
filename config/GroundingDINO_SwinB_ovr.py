modelname = "groundingdino"
backbone = "swin_B_384_22k"
position_embedding = "sine"
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
backbone_freeze_keywords = None
enc_layers = 6
dec_layers = 6
pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 900
query_dim = 4
num_patterns = 0
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4
two_stage_type = "standard"
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
transformer_activation = "relu"
dec_pred_bbox_embed_share = True
dn_box_noise_scale = 1.0
dn_label_noise_ratio = 0.5
dn_label_coef = 1.0
dn_bbox_coef = 1.0
embed_init_tgt = True
dn_labelbook_size = 2000
max_text_len = 512 #256
text_encoder_type = "bert-base-uncased"
use_text_enhancer = True
use_fusion_layer = True
use_checkpoint = True
use_transformer_ckpt = True
use_text_cross_attention = True
text_dropout = 0.0
fusion_dropout = 0.0
fusion_droppath = 0.1
sub_sentence_present = True


# train
frozen_weights = None 
frozen_backbone = True 

lr =  1e-5 
lr_rln_mult = 10
param_dict_type = 'default'
lr_backbone = 1e-05
lr_text_backbone = 1e-05
lr_backbone_names = ['backbone.0']
lr_linear_proj_names = ['reference_points', 'sampling_offsets']
lr_linear_proj_mult = 0.1
ddetr_lr_param = False
batch_size = 4
weight_decay = 0.0001
epochs = 12
lr_drop = 11
save_checkpoint_interval = 1
clip_max_norm = 0.1
onecyclelr = False
multi_step_lr = False
lr_drop_list = [20, 23] 

aux_loss = True
set_cost_class = 1.0
set_cost_bbox = 5.0
set_cost_giou = 2.0

cls_loss_coef = 2.0
mask_loss_coef = 1.0
dice_loss_coef = 1.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0
interm_loss_coef = 1.0
no_interm_box_loss = False
focal_alpha = 0.25

# for dn
use_dn = True
dn_number  = 100
masks = False
use_text_labels = True 


vg_roidb_key = 'split_GLIPunseen'
num_select = 300
nms_iou_threshold = 0.5

has_bbox_supervision = True 
rln_pretraining = False 
do_sgg = True 
num_rln_cat = 51
num_rln_queries = 1
edge_loss_coef = 1.0

rln_add_coords = True

rln_freq_bias = None
sgg_mode = 'ovr'

sg_ovd_mode = False
sg_ovr_mode = True
focal_loss_for_edges = True


detections_per_img = 100
use_distill = True
unsupervised_distill=True
distill_loss_coef = 0.1 

#teacher_weight='checkpoints/vg-pretrain-coco-swinb.pth'
