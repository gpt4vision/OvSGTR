
DEBUG=1 CUDA_VISIBLE_DEVICES=5 python main.py \
	--output_dir $4 -c $2 --data_path $3 \
	--dataset_file $1  \
	--pretrain_model_path $5 \
	--num_workers 0 \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0 eval_before_train=False  \
	batch_size=4
