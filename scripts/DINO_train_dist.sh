

data_type=$1
config=$2
data_path=$3
out=$4



CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py \
	--output_dir $out -c $config --data_path $data_path \
	--dataset_file $data_type \
	--pretrain_model_path $5  \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0

