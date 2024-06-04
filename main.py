# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
import numpy as np
import copy

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.multiprocessing as mp
import torch.distributed as dist 

from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import ModelEma, BestMetricHolder
import util.misc as utils

import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, test

import wandb
from collections import OrderedDict

from groundingdino.models.GroundingDINO import build_groundingdino

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


#os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'




def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--data_path', type=str, default='/comp_robot/cv_public_dataset/')
    parser.add_argument('--anno_train_file', type=str, default='oiv6_train_triple.jsonl')
    parser.add_argument('--anno_val_file', type=str, default='val_bbox_cap.jsonl')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')

    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--half_dtype', default="bfloat16", type=str)

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    
    return parser


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict, "ERROR: modelname:{} not in models:{}".format(args.modelname, MODULE_BUILD_FUNCS._module_dict)

    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors

def main(gpu, ngpus_per_node, args):
    args.gpu = gpu 
    #utils.init_distributed_mode(args)
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])

        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend='nccl', init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    torch.cuda.set_device(args.gpu)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.set_deterministic(True)

    # load cfg file and update the args
    if args.rank == 0:
        print("Loading config file from {}".format(args.config_file))

    #time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)


    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # update some new args temporally
    if not getattr(args, 'use_ema', None):
        args.use_ema = False
    if not getattr(args, 'debug', None):
        args.debug = False

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="detr")
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')

    wandb_logger = None
    if utils.get_rank() == 0 and not args.eval:
        if os.environ.get("DEBUG") not in ['1', '2']:
            wandb_logger = wandb.init(config=args, project='dino-'+args.dataset_file)

    if not hasattr(args, "frozen_weights"):
        args.frozen_weights = None


    if args.rank == 0:
        print(args)

    device = torch.device(args.device)

    if args.distributed:
        dist.barrier()

    # build model
    model, criterion, postprocessors = build_model_main(args)
    model_without_ddp = model
    if utils.get_rank() == 0:
        print("PostProcessors:", postprocessors)

    # load frozen weights if needed
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"

        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    rln_proj = getattr(model, "rln_proj", None)
    rln_classifier = getattr(model, "rln_classifier", None)
    rln_freq_bias = getattr(model, "rln_freq_bias", None)

    if utils.get_rank() == 0:
        logger.info("rank:{}, rln_proj:{}, rln_classifier:{}, \
                     rln_freq_bias:{}".format(utils.get_rank(), \
                        rln_proj, rln_classifier, rln_freq_bias,
                        ))

    try:
        if utils.get_rank() == 0 and args.frozen_backbone:
            logger.info("Frozen backbone!")
    except:
        pass


    wo_class_error = False
    model = model.cuda()
    criterion = criterion.cuda()

    # ema
    if args.use_ema:
        ema_m = ModelEma(model, args.ema_decay)
    else:
        ema_m = None

    # load pre-trained weights
    if (not args.resume) and args.pretrain_model_path:
        logger.info("*"*10 + "Loading weights from pretrained model:%s ..." % args.pretrain_model_path)
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() 
                                if check_keep(k, _ignorekeywordlist)})
        pre_w = _tmp_st['bert.embeddings.word_embeddings.weight']
        pad_l = model_without_ddp.bert.embeddings.word_embeddings.weight.shape[0] - pre_w.shape[0]

        if pad_l > 0:
            logger.info("Resize bert embeddings")
            pad_w = model_without_ddp.bert.embeddings.word_embeddings.weight[pre_w.shape[0]:].to(pre_w.device)
            pre_w = torch.cat((pre_w, pad_w), 0)
            _tmp_st['bert.embeddings.word_embeddings.weight'] = pre_w

        missing, unexpected = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info("Miss keys:{}".format(missing))
        logger.info("Unexpected keys:{}".format(unexpected))

        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)        

        rln_text_proj = getattr(model_without_ddp, "rln_text_proj", None)
        rln_text_proj_exist = False 
        for k in _tmp_st.keys():
            if 'rln_text_proj' in k:
                rln_text_proj_exist = True
                break

        if rln_text_proj is not None and (not args.eval) and not rln_text_proj_exist:
            feat_map_state = model_without_ddp.feat_map.state_dict()
            rln_text_proj.load_state_dict(feat_map_state) # initialize rln_text_proj with feat_map
            logger.info("*"*10 + "Copy weights from feat_map to rln_text_proj" + "*"*10)

    if getattr(args, "use_distill", False):
        logger.info("*"*10 + "use_distill=True!")
        model_t = copy.deepcopy(model)
        teacher_weight = getattr(args, "teacher_weight", None)
        if teacher_weight is not None:
            logger.info("Loading Teacher weight:{}".format(teacher_weight))
            checkpoint = torch.load(teacher_weight, map_location='cpu')['model']
            missing, unexpected = model_t.load_state_dict(utils.clean_state_dict(checkpoint))
            logger.info("Teacher Missing keys:{}".format(missing))
            logger.info("Teacher unexpected keys:{}".format(unexpected))

        model_t.eval()
        rln_proj_teacher = model_t.rln_proj
    else:
        model_t = None 
        rln_proj_teacher = None

    if args.distributed:
        dist.barrier()
        if args.find_unused_params:
            print("*"*10, "Warning: find_unused_parameters = True !")
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module

    if args.distributed:
        if rln_proj is not None:
            rln_proj = DDP(rln_proj, device_ids=[args.gpu], 
                          find_unused_parameters=args.find_unused_params)

        if rln_classifier is not None:
            rln_classifier = DDP(rln_classifier, device_ids=[args.gpu],
                                 find_unused_parameters=args.find_unused_params)

        if rln_freq_bias is not None:
            rln_freq_bias = DDP(rln_freq_bias, device_ids=[args.gpu],
                                 find_unused_parameters=args.find_unused_params)





    criterion.rln_proj = rln_proj
    criterion.rln_proj_teacher = rln_proj_teacher
    criterion.rln_classifier =  rln_classifier
    criterion.rln_freq_bias = rln_freq_bias
    postprocessors['bbox'].rln_proj = rln_proj
    postprocessors['bbox'].rln_classifier = rln_classifier
    postprocessors['bbox'].rln_freq_bias = rln_freq_bias

    frozen_p = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            frozen_p.append(name)

    logger.info("frozen params:{}".format(frozen_p))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of trainable params:'+str(n_parameters))
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("number of total params:" + str(total_params))
    logger.info("params:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

    param_dicts = get_param_dict(args, model_without_ddp)

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    

    dataset_train = build_dataset(image_set='train', args=args)
    
    use_test_set = getattr(args, "use_test_set", False)
    if use_test_set:
        print("*"*10, " Use test set !", "*"*10)

    dataset_val = build_dataset(image_set='val' if not use_test_set else 'test', args=args)
    if utils.get_rank() == 0:
        logger.info("len(dataset_train)=%s, len(dataset_val)=%s" %(len(dataset_train), len(dataset_val)) )

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train, drop_last=True)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train, sampler_val = None, None


    try:
        is_oiv6 = args.dataset_file == 'oicap'
    except:
        is_oiv6 = False

    data_loader_train = DataLoader(dataset_train, 
                                   batch_size=args.batch_size, 
                                   sampler=sampler_train,
                                   shuffle=(sampler_train is None), 
                                   collate_fn=utils.collate_fn, 
                                   num_workers=args.num_workers if not is_oiv6 else 2, # > 0 may OOM
                                   pin_memory=True)

    data_loader_val = DataLoader(dataset_val, batch_size=1, 
                                 sampler=sampler_val,
                                 drop_last=False, shuffle=False, 
                                 collate_fn=utils.collate_fn, 
                                 num_workers=args.num_workers,
                                 pin_memory=True
                                 )

    if args.onecyclelr:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(data_loader_train), epochs=args.epochs, pct_start=0.2)
    elif args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)


    output_dir = Path(args.output_dir)
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
    if args.resume: # resume 
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        missing, unexpected = model_without_ddp.load_state_dict(
                                        utils.clean_state_dict(checkpoint['model']), strict=False)

        logger.info("RESUME -- Miss keys:{}".format(missing))
        logger.info("RESUME -- Unexpected keys:{}".format(unexpected))
        if 'epoch' in checkpoint:
            logger.info("RESUME -- epoch:{}".format(checkpoint['epoch']))

        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)                

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1


    if args.eval:
        if args.distributed:
            dist.barrier()

        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
        if args.output_dir and coco_evaluator is not None:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()} }
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if wandb_logger is not None:
            wandb_logger.finish()

        return



    if args.distributed:
        dist.barrier()

    start_time = time.time()
    best_map_holder = BestMetricHolder(use_ema=args.use_ema)


    eval_before_train = getattr(args, "eval_before_train", True)
    if os.environ.get("DEBUG") == '1':
        eval_before_train = False 

    if eval_before_train:
        print("*"*10, " eval before training ...")
        # eval before training
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
            wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
        )
        map_regular = test_stats['coco_eval_bbox'][0]
        if wandb_logger is not None:
            if 'R@50' in test_stats:
                wandb_logger.log({'R@50': test_stats['R@50']})
            if 'zR-OvD@50' in test_stats:
                wandb_logger.log({'zR-OvD@50': test_stats['zR-OvD@50']})
            if 'zR-OvR@50' in test_stats:
                wandb_logger.log({'zR-OvR@50': test_stats['zR-OvR@50']})

            wandb_logger.log({'epoch': -1, 'eval': map_regular})

    if args.distributed:
        dist.barrier()
    print("*"*10, " start training ...")
    args.global_iter = -1

    # train
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, 
            args=args, logger=(logger if args.save_log else None), ema_m=ema_m, wandb_logger=wandb_logger, 
            model_t=model_t)

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']

        if not args.onecyclelr:
            lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                if args.use_ema:
                    weights.update({
                        'ema_model': ema_m.module.state_dict(),
                    })
                utils.save_on_master(weights, checkpoint_path)
                
        # eval
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
            wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
        )
        map_regular = test_stats['coco_eval_bbox'][0]
        if 'R@50' in test_stats:
            _isbest = best_map_holder.update(test_stats['R@50'], epoch, is_ema=False)
        else:
            _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)

        if wandb_logger is not None:
            if 'R@50' in test_stats:
                wandb_logger.log({'R@50': test_stats['R@50']})
            if 'zR-OvD@50' in test_stats:
                wandb_logger.log({'zR-OvD@50': test_stats['zR-OvD@50']})
            if 'zR-OvR@50' in test_stats:
                wandb_logger.log({'zR-OvR@50': test_stats['zR-OvR@50']})

            wandb_logger.log({'epoch': epoch, 'eval': map_regular})

        if _isbest:
            checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
        }

        # eval ema
        if args.use_ema:
            ema_test_stats, ema_coco_evaluator = evaluate(
                ema_m.module, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            )
            log_stats.update({f'ema_test_{k}': v for k,v in ema_test_stats.items()})
            map_ema = ema_test_stats['coco_eval_bbox'][0]
            _isbest = best_map_holder.update(map_ema, epoch, is_ema=True)
            if _isbest:
                checkpoint_path = output_dir / 'checkpoint_best_ema.pth'
                utils.save_on_master({
                    'model': ema_m.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        log_stats.update(best_map_holder.summary())

        ep_paras = {
                'epoch': epoch,
                'n_parameters': n_parameters
            }
        log_stats.update(ep_paras)
        try:
            log_stats.update({'now_time': str(datetime.datetime.now())})
        except:
            pass
        
        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats['epoch_time'] = epoch_time_str

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # remove the copied files.
    copyfilelist = vars(args).get('copyfilelist')
    if copyfilelist and args.local_rank == 0:
        from datasets.data_util import remove
        for filename in copyfilelist:
            print("Removing: {}".format(filename))
            remove(filename)

    if wandb_logger is not None:
        wandb_logger.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    ngpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 1
    args.distributed = ngpus_per_node > 1 # always use ddp for multiple gpus


    if args.distributed:
        args.world_size = args.world_size * ngpus_per_node

        port_id = 10000 + np.random.randint(0, 1000)
        args.dist_url = 'tcp://127.0.0.1:' + str(port_id)
        mp.spawn(main, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main(args.rank, ngpus_per_node, args)

