config=$2
data_path=$3
output_dir=$4
checkpoint=$5

#CUDA_VISIBLE_DEVICES=0 

#DEBUG=1 
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py \
  --output_dir $output_dir \
	-c $config --data_path $data_path \
	--eval \
	--resume $checkpoint \
	--dataset_file $1 \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0   use_test_set=True  use_gt_box=False
