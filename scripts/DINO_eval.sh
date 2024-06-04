config=$2
data_path=$3
output=$4
checkpoint=$5


#DEBUG=2
CUDA_VISIBLE_DEVICES=0 python main.py \
  --output_dir $output \
	-c $config --data_path $data_path  \
	--eval --resume $checkpoint --dataset_file $1 \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False use_test_set=True #use_gt_box=True
