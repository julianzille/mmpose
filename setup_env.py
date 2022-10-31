## Most code/functions are extracted from Open-MMLab library:
## https://github.com/open-mmlab
## https://github.com/open-mmlab/mmdetection
## https://github.com/open-mmlab/mmpose

import mmcv
from mmcv import Config, DictAction
import mmpose
import torch, torchvision
import time
import os
import shutil
import os.path as osp
import numpy as np
import pickle

import time
import warnings
import copy

from collections import OrderedDict, defaultdict
import torch.distributed as dist
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import get_git_hash
from mmpose import __version__
from mmpose.apis import init_random_seed, train_model
from mmpose.models import build_posenet
from mmpose.utils import collect_env, get_root_logger, setup_multi_processes

from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmpose.apis import multi_gpu_test, single_gpu_test
from mmpose.datasets import build_dataloader, build_dataset

try:
    from mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import wrap_fp16_model
    
from matplotlib import pprint
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result, process_mmdet_results)
from mmdet.apis import inference_detector, init_detector
from xtcocotools.coco import COCO

local_runtime = False
import cv2
from matplotlib import pyplot as plt
from mmpose.core.bbox.transforms import bbox_xywh2xyxy
from mmpose.datasets import DatasetInfo
import matplotlib
from mmpose.apis import (collect_multi_frames, get_track_id,
                         inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_tracking_result)

from mmpose.core import Smoother
from mmpose.apis import (collect_multi_frames, extract_pose_sequence,
                         get_track_id, inference_pose_lifter_model,
                         inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_3d_pose_result)

from mmpose.core import Smoother
from mmpose.datasets import DatasetInfo
import pandas as pd
from PIL import Image
import os
import pandas as pd
import random
import math
import json


# Train from config file
def train_2D_cfg(config,pretr=True):
    cfg=Config.fromfile(config)
    if pretr==False:
        cfg.model['pretrained']=None
    elif pretr==True:
        cfg.work_dir=cfg.work_dir+'_pretr'
        
    mmcv.mkdir_or_exist(cfg.work_dir)
    cfg.log_name = time.strftime('%d-%m-', time.localtime())+"ep"+str(cfg.total_epochs)
    cfg.log_file = os.path.join(cfg.work_dir, f'{cfg.log_name}.log')
    
    autoscale_lr = False # automatically scale lr with the number of gpus
    launcher = 'none' # Job launcher. ['none', 'pytorch', 'slurm', 'mpi']
    deterministic = False # Whether to set deterministic options for CUDNN backend.
    diff_seed = False # Whether or not set different seeds for different ranks
    seed = 0 

    if autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    if launcher == 'none':
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(
                f'We treat {cfg.gpu_ids} as gpu-ids, and reset to '
                f'{cfg.gpu_ids[0:1]} as gpu-ids to avoid potential error in '
                'non-distribute training time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # set multi-process settings
    setup_multi_processes(cfg)

    # init the logger before other steps
    logger = get_root_logger(log_file=cfg.log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()

    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    # logger.info('Environment info:\n' + dash_line + env_info + '\n' +
    #             dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    # logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(seed)
    seed = seed + dist.get_rank() if diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {deterministic}')
    set_random_seed(seed, deterministic=deterministic)
    cfg.seed = seed
    meta['seed'] = seed

    # build dataset
    train_set = [build_dataset(cfg.data.train)]

    # build model
    model = build_posenet(cfg.model)

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        train_set.append(build_dataset(val_dataset))

    if cfg.checkpoint_config is not None:
        # save mmpose version, config file content
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmpose_version=__version__ + get_git_hash(digits=7),
            # config=cfg.pretty_text,
        )

    timestamp = time.strftime('%d-%m-', time.localtime())+"ep"+str(cfg.total_epochs)

    # train model
    train_model(
        model,
        train_set, 
        cfg, 
        distributed=distributed, 
        validate=True, 
        timestamp=timestamp,
        meta=meta)

def test_pose_estimator(wrk_dir,config,pretr):
    cfg=Config.fromfile(config)
    if pretr==False:
        ckpt=wrk_dir+'/latest.pth'
        cfg.model.pretrained=None
    elif pretr==True:
        ckpt=wrk_dir+'_pretr/latest.pth'
        cfg.work_dir=cfg.work_dir+'_pretr'
        
    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    distributed = False

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    # step 1: give default values and override (if exist) from cfg.data
    loader_cfg = {
        **dict(seed=cfg.get('seed'), drop_last=False, dist=distributed),
        **({} if torch.__version__ != 'parrots' else dict(
               prefetch_num=2,
               pin_memory=False,
           )),
        **dict((k, cfg.data[k]) for k in [
                   'seed',
                   'prefetch_num',
                   'pin_memory',
                   'persistent_workers',
               ] if k in cfg.data)
    }
    # step2: cfg.data.test_dataloader has higher priority
    test_loader_cfg = {
        **loader_cfg,
        **dict(shuffle=False, drop_last=False),
        **dict(workers_per_gpu=cfg.data.get('workers_per_gpu', 1)),
        **dict(samples_per_gpu=cfg.data.get('samples_per_gpu', 1)),
        **cfg.data.get('test_dataloader', {})
    }
    
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    model = build_posenet(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, ckpt, map_location='cpu')
    model = fuse_conv_bn(model)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader)
    # print(outputs)
    rank, _ = get_dist_info()
    eval_config = cfg.get('evaluation', {})

    results = dataset.evaluate(outputs, cfg.work_dir, **eval_config)
    for k, v in sorted(results.items()):
        print(f'{k}: {v}')
        
    
def load_pickle(pickle_file):
    """
    Loads a dictionary from a saved skeleton .pickle file
    """
    with open(pickle_file, 'rb') as handle:
        data = pickle.load(handle)

    return(data)

def demo_pose_lifter(det_cfg,det_ckpt,pose_cfg,pose_ckpt,poselift_cfg,poselift_ckpt,vid):
       
    return_heatmap=False
    output_layer_names = None
    use_multi_frames=False
    rebase_keypoint_height=True
    num_instances=1
    pose_results_list = []
    next_id = 0
    pose_results = []
    save_out_video=True
    
    kp3d=np.empty((0,4),float)
    
    smoother = Smoother(filter_cfg='configs/_base_/filters/one_euro.py',
                        keypoint_key='keypoints',
                        keypoint_dim=2)
    
    det_model = init_detector(det_cfg,det_ckpt)
    
    pose_model=init_pose_model(pose_cfg,pose_ckpt)
    pose_dataset = pose_model.cfg.data['test']['type']
    pose_dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    pose_dataset_info=DatasetInfo(pose_dataset_info)    
    
    poselift_model=init_pose_model(poselift_cfg,poselift_ckpt)
    poselift_dataset = poselift_model.cfg.data['test']['type']
    poselift_dataset_info = poselift_model.cfg.data['test'].get('dataset_info', None)
    poselift_dataset_info = DatasetInfo(poselift_dataset_info)
    
    video = mmcv.VideoReader(vid)
    # length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames=len(video)
    
    print("\n2D Detection and Pose Estimation:")
    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
        pose_results_last = pose_results
        # get the detection results of current frame
        # the resulting box is (x1, y1, x2, y2)
        det_results = inference_detector(det_model, cur_frame)
        # keep the person class bounding boxes.
        det_results = process_mmdet_results(det_results, 1)
        
        pose_results, _ = inference_top_down_pose_model(
            pose_model,
            cur_frame,
            det_results,
            format='xyxy',
            dataset=pose_dataset,
            bbox_thr=0.9,
            dataset_info=pose_dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)
        
        
        pose_results, next_id = get_track_id(
            pose_results,
            pose_results_last,
            next_id,
            use_oks=True,
            use_one_euro=True,
            tracking_thr=0.1,
            sigmas=pose_dataset_info.sigmas)
        
        # pose_results = smoother.smooth(pose_results)
        # print(next_id)
        pose_results_list.append(copy.deepcopy(pose_results))
    
    if save_out_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = video.fps
        writer = None
    
    # Re-arrange keypoints:
    for pose_results in pose_results_list:
        for res in pose_results:
            # del res['track_id']
            keypoints = res['keypoints']
            keypoints_new = np.zeros((20, keypoints.shape[1]), dtype=keypoints.dtype)
            keypoints_new=keypoints[[2,1,0,3,
                                    23,4,18,17,
                                    8,9,19,
                                    5,6,20,
                                    14,15,21,
                                    11,12,22]] 
            res['keypoints']=keypoints_new
    
    data_cfg = poselift_model.cfg.data_cfg
    # print(pose_results_list)
    
    # print(np.shape(pose_results_list))
    
    print("\nPose Lifting:")
    #Pose det results: iterate over each frame
    for i, pose_results in enumerate(mmcv.track_iter_progress(pose_results_list)): 
        # extract and pad input pose2d sequence
        # returns 2D kp and BBox for 27 frames. len = 27
        pose_results_2d = extract_pose_sequence( 
            pose_results_list,
            frame_idx=i,
            causal=data_cfg.causal,
            seq_len=data_cfg.seq_len,
            step=data_cfg.seq_frame_interval)
        
        # if i<15:
        #     print(pose_results_2d)
            
        # pose_results_2d = smoother.smooth(pose_results_2d)

        # 2D-to-3D pose lifting
        #'keypoints': 2D keypoints for 27 frames, 'keypoints_3d': 3D keypoints for target frame
        poselift_results = inference_pose_lifter_model( 
            poselift_model,
            pose_results_2d=pose_results_2d,
            dataset=poselift_dataset,
            dataset_info=poselift_dataset_info,
            with_track_id=True,
            image_size=video.resolution,
            norm_pose_2d=False)
        # if i==10:
        #     print(poselift_results)
        # print(len(poselift_results)) #Norm = true
                
        poselift_results_vis = []
        for idx, res in enumerate(poselift_results):
            keypoints_3d = res['keypoints_3d']
            
            # exchange y,z-axis, and then reverse the direction of x,z-axis
            # keypoints_3d = keypoints_3d[..., [0, 2, 1]]
            keypoints_3d[..., 0] = -keypoints_3d[..., 0]
            # keypoints_3d[..., 2] = -keypoints_3d[..., 2]
            # rebase height (z-axis)
            
            if rebase_keypoint_height:
                keypoints_3d[..., 2] -= np.min(
                    keypoints_3d[..., 2], axis=-1, keepdims=True)
            res['keypoints_3d'] = keypoints_3d
            # add title
            det_res = pose_results[idx]
            instance_id = det_res['track_id']
            res['title'] = f'Prediction ({instance_id})'
            
            # only visualize the target frame
            res['keypoints'] = det_res['keypoints']
            res['bbox'] = det_res['bbox']
            res['track_id'] = instance_id
            poselift_results_vis.append(res)

        if len(poselift_results)==0:
            # print(poselift_results)
            kp3d=np.append(kp3d,np.zeros((20,4)),axis=0)
        else:    
            kp3d=np.append(kp3d,poselift_results_vis[0]['keypoints_3d'],axis=0)
        
        # Visualization
        if num_instances < 0:
            num_instances = len(poselift_results_vis)
        
        img_vis = vis_3d_pose_result(
            poselift_model,
            result=poselift_results_vis,
            img=video[i],#i
            dataset=poselift_dataset,
            dataset_info=poselift_dataset_info,
            out_file=None,
            num_instances=num_instances,
            show=False)
        
        # print(poselift_results_vis[0])
        
        if save_out_video:
            if writer is None:
                writer = cv2.VideoWriter(os.path.join('vis_results',
                         f'lift_{os.path.basename(vid)}'), fourcc, fps, (img_vis.shape[1], img_vis.shape[0]))
            writer.write(img_vis) 
            
        
    if save_out_video:
        writer.release()
    kp3d=kp3d.reshape(frames,20,4)
    return kp3d
        
def add_cameras(n,calib_file,cameras_dict):
    calib_dict=mmcv.load(calib_file)
    
    for i in range(n):
        R=np.array(calib_dict['cameras'][i]['r'])
        T=np.array(calib_dict['cameras'][i]['t'])
        K=np.array(calib_dict['cameras'][i]['k'][:2])
        k=np.array(calib_dict['cameras'][i]['d'][:2])
        k=np.append(k,[k[1]/2],axis=0)
        p=np.array(calib_dict['cameras'][i]['d'][-2:])
        w=calib_dict['camera_resolution'][0]
        h=calib_dict['camera_resolution'][1]
        
        date=calib_file.split('/')[2]
        
        if 'top' in calib_file.split('/'):
            name=f'{date}_1_cam{i+1}'
            idx=f'{date}_1{i+1}'.replace('_','')
        elif 'bottom' in calib_file.split('/'):
            name=f'{date}_2_cam{i+1}'
            idx=f'{date}_2{i+1}'.replace('_','')
        else:
            assert calib_file.split('/')[2].split('_')[0]=='2019'
            name=f'{date}_cam{i+1}'
            idx=f'{date}_0{i+1}'.replace('_','')
            
        cameras_dict[idx]=dict(R=R,T=T,K=K,k=k,p=p,w=w,h=h,name=name)
        cameras_dict[idx]['id']=idx
        
    return cameras_dict

def get_cam(camera_id,cameras):
    for item in list(cameras.keys()):
        copy=item.replace('_','')
        copy=copy.replace('cam','0')
        if copy==camera_id:
            
            return cameras[item]
        else:
            raise Exception("Check camera dict")

def _get_pose_stats(kps):
    """Get statistic information `mean` and `std` of pose data.
    Args:
        kps (ndarray): keypoints in shape [..., K, C] where K and C is
            the keypoint category number and dimension.
    Returns:
        mean (ndarray): [K, C]
    """
    assert kps.ndim > 2
    K, C = kps.shape[-2:]
    kps = kps.reshape(-1, K, C)
    mean = kps.mean(axis=0)
    std = kps.std(axis=0)
    return mean, std

def get_centers_scales(bboxes,scale_factor=1.2):
    centers = np.stack([(bboxes[:, 0] + bboxes[:, 2]) / 2,(bboxes[:, 1] + bboxes[:, 3]) / 2], axis=1)
    scales = scale_factor * np.max(bboxes[:, 2:] - bboxes[:, :2], axis=1) / 200
    return centers, scales

def get_fte_anns(date,cheetah,action,cam):
    #2D Anns
    anns2d=pd.read_csv(f'data/acino_3d/{date}/{cheetah}/{action}/fte_pw/'+cam.split('.')[0]+'_fte.csv')
    anns2d.columns=anns2d.columns+anns2d.iloc[0] # Make keypoint labels column headers
    anns2d=anns2d.iloc[1:,1:]
    kp2d=[]
    
    for img in range(len(anns2d)):
        for kp in range(0,len(anns2d.columns),3):
            if math.isnan(float(anns2d.iloc[img,kp])): 
                kp2d.append(0)
                kp2d.append(0)
                kp2d.append(bool(0))
            else:
                kp2d.append(float(anns2d.iloc[img,kp]))           
                kp2d.append(float(anns2d.iloc[img,kp+1]))
                kp2d.append(bool(1))
                
    kp2d=np.array(kp2d,dtype=float).reshape(len(anns2d),20,3)
    bboxes = np.stack([np.min(kp2d[:, :, 0], axis=1),np.min(kp2d[:, :, 1], axis=1),np.max(kp2d[:, :, 0], axis=1),np.max(kp2d[:, :, 1], axis=1)],axis=1)
    return kp2d,bboxes

def generate_stats(kp2darr,kp3darr,dirr):
    
    kp3darr = kp3darr[..., :3]  # remove visibility
    mean_3d, std_3d = _get_pose_stats(kp3darr)

    kp2darr = kp2darr[..., :2]  # remove visibility
    mean_2d, std_2d = _get_pose_stats(kp2darr)

    
    # centered around root
    # the root keypoint is 0-index
    kps_3d_rel = kp3darr[..., 1:, :] - kp3darr[..., :1, :]
    mean_3d_rel, std_3d_rel = _get_pose_stats(kps_3d_rel)

    kps_2d_rel = kp2darr[..., 1:, :] - kp2darr[..., :1, :]
    mean_2d_rel, std_2d_rel = _get_pose_stats(kps_2d_rel)

    stats = {
        'joint3d_stats': {
            'mean': mean_3d,
            'std': std_3d
        },
        'joint2d_stats': {
            'mean': mean_2d,
            'std': std_2d
        },
        'joint3d_rel_stats': {
            'mean': mean_3d_rel,
            'std': std_3d_rel
        },
        'joint2d_rel_stats': {
            'mean': mean_2d_rel,
            'std': std_2d_rel
        }
    }

    for name, stat_dict in stats.items():
        out_file = f'data/acino_3d/annotations/{dirr}/{name}.pkl'


        with open(out_file, 'wb') as f:
            pickle.dump(stat_dict, f)
        print(f'Create statistic data file: {out_file}')

def get_anns(cam_id, cheetah, action, cam, date, start_frame, end_frame, pose_model,pose_dataset,pose_dataset_info,det_model,smoother):
    ''' Calculate centers, scales and estimate 2D keypoints using top-down pose-estimator (pose-detect + '''
    
    #DLC 2D Anns:
    kp2d_fte, bboxes_fte = get_fte_anns(date,cheetah,action,cam)
    
    #Save frames
    imgnames=[]
    bboxes_res=np.array([])
    
    kp2d_res=np.empty((0,3), float)
    
    video=mmcv.VideoReader(f'data/acino_3d/{date}/{cheetah}/{action}/{cam}')
    
    next_id=0
    pose_results=[]
        
    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
        if frame_id>=start_frame and frame_id<end_frame: 
            
            pose_results_last=pose_results
            
            imgname=f'{cheetah}'+date.split('_')[1]+(date.split('_')[2])+f'_{action}.{cam_id}_{frame_id}.jpg'
            mmcv.imwrite(video[frame_id],f'data/acino_3d/images/{imgname}')  # save frame as JPEG file      
            imgnames.append(imgname)
            
            det_results=inference_detector(det_model,cur_frame)
            cheetah_results=process_mmdet_results(det_results,1)
            
            pose_results,returned_outputs=inference_top_down_pose_model(
                pose_model,
                cur_frame,
                cheetah_results,
                # bbox_thr=0.3,
                format='xyxy',
                dataset=pose_dataset,
                dataset_info=pose_dataset_info,
                return_heatmap=False,
                outputs=None)
            
            pose_results, next_id = get_track_id(
                    pose_results,
                    pose_results_last,
                    next_id,
                    use_oks=True,
                    tracking_thr=0.3,
                    sigmas=pose_dataset_info.sigmas)
            
            pose_results = smoother.smooth(pose_results)
            # if frame_id==10:
            #     print(returned_outputs)
            assert pose_results != []
                          
            bboxes_res=np.append(bboxes_res,pose_results[0]['bbox'][:4])
            keypoints=pose_results[0]['keypoints']
            keypoints_new=keypoints[[2,1,0,3,
                                23,4,18,17,
                                8,9,19,
                                5,6,20,
                                14,15,21,
                                11,12,22]]
            kp2d_res=np.append(kp2d_res,keypoints_new,axis=0)
                
            # else:
            #     bboxes_res=np.append(bboxes_res,bboxes_fte[frame_id-start_frame])
            #     kp2d_res=np.append(kp2d_res,kp2d_fte[frame_id-start_frame],axis=0)
            
    
    imgnames=np.array(imgnames)        
    
    #2D Anns
    kp2d_res=kp2d_res.reshape(len(kp2d_res)//20 ,20,3)        
    bboxes_res=bboxes_res.reshape(len(bboxes_res)//4,4)
    
    scale_factor=1.2
    centers_res,scales_res= get_centers_scales(bboxes_res,scale_factor)
    centers_fte,scales_fte= get_centers_scales(bboxes_fte,scale_factor)

    #3D Anns
    anns3d=mmcv.load(f'data/acino_3d/{date}/{cheetah}/{action}/fte_pw/fte.pickle')
    kp3d=anns3d['positions']
    kp3d = np.array(kp3d,dtype=float).reshape(len(anns3d['positions']),20,3)
    kp3d = np.concatenate([kp3d, np.ones((len(kp3d), 20, 1))],
                                axis=2)
    
    assert len(imgnames)==kp3d.shape[0]
    assert kp3d.shape[0]==kp2d_res.shape[0]
    assert kp2d_res.shape==kp2d_fte.shape
    
    return imgnames,centers_res,scales_res,kp2d_res,kp3d,centers_fte,scales_fte,kp2d_fte

def _parse_h36m_imgname(imgname):
    subj, rest = osp.basename(imgname).split('_', 1)
    action, rest = rest.split('.', 1)
    camera, rest = rest.split('_', 1)
    return subj,action,camera

def load_pickle(pickle_file):
    """
    Loads a dictionary from a saved skeleton .pickle file
    """
    with open(pickle_file, 'rb') as handle:
        data = pickle.load(handle)

    return(data)

def train_test_lifter(config,eps,anns2d,causal,frames,train=False,test=True):
    cfg=Config.fromfile(config)
    cfg.total_epochs=eps
    
    cfg.data['train']['ann_file']=f'data/acino_3d/annotations/{anns2d}/acino3d_train.npz'
    cfg.data['val']['ann_file']=f'data/acino_3d/annotations/{anns2d}/acino3d_val.npz'
    cfg.data['test']['ann_file']=f'data/acino_3d/annotations/{anns2d}/acino3d_test.npz'
    
    if causal:
        caus='causal'
    else:
        caus='noncausal'
    
    cfg.model['backbone']['causal']=causal
    cfg.work_dir=f'work_dirs/acino_poselift_{anns2d}_{caus}_{frames}frames'

    cfg.log_name = time.strftime('%d-%m-', time.localtime())+"ep"+str(cfg.total_epochs)
    cfg.log_file = os.path.join(cfg.work_dir, f'{cfg.log_name}.log')
    mmcv.mkdir_or_exist(cfg.work_dir)
    cfg.data_cfg['causal']=causal
    # set random seeds
    seed=0
    seed = init_random_seed(seed)
    
    set_random_seed(seed, deterministic=False)
    cfg.seed = seed
    
    if train:
        train_lifter(cfg)
    if test:
        test_poselifter(cfg,f'{cfg.work_dir}/latest.pth')
    
def train_lifter(cfg):
    # set multi-process settings
    setup_multi_processes(cfg)
    # init the logger before other steps
    logger = get_root_logger(log_file=cfg.log_file, log_level=cfg.log_level)
    
    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()

    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    # logger.info('Environment info:\n' + dash_line + env_info + '\n' +
    #             dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: False')
    # logger.info(f'Config:\n{cfg.pretty_text}')
    logger.info(f'Set random seed to {cfg.seed}, '
                f'deterministic: False')
    meta['seed'] = cfg.seed

    # build dataset
    train_set = [build_dataset(cfg.data.train)]

    # build model
    model = build_posenet(cfg.model)
    timestamp = time.strftime('%d-%m-', time.localtime())+"ep"+str(cfg.total_epochs)
    
    # print(cfg.lr_config)
    train_model(
        model,
        train_set, 
        cfg, 
        distributed=False, 
        validate=True, 
        timestamp=timestamp,
        meta=meta)
    
    
    
def test_poselifter(cfg,ckpt):
    
    # cfg=Config.fromfile(config)
    
    distributed = False
    fuse_convbn=True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    # step 1: give default values and override (if exist) from cfg.data
    loader_cfg = {
        **dict(seed=cfg.get('seed'), drop_last=False, dist=distributed),
        **({} if torch.__version__ != 'parrots' else dict(
               prefetch_num=2,
               pin_memory=False,
           )),
        **dict((k, cfg.data[k]) for k in [
                   'seed',
                   'prefetch_num',
                   'pin_memory',
                   'persistent_workers',
               ] if k in cfg.data)
    }
    # step2: cfg.data.test_dataloader has higher priority
    test_loader_cfg = {
        **loader_cfg,
        **dict(shuffle=False, drop_last=False),
        **dict(workers_per_gpu=cfg.data.get('workers_per_gpu', 1)),
        **dict(samples_per_gpu=cfg.data.get('samples_per_gpu', 1)),
        **cfg.data.get('test_dataloader', {})
    }
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    model = build_posenet(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, ckpt, map_location='cpu')

    if fuse_convbn:
        model = fuse_conv_bn(model)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader)

    rank, _ = get_dist_info()
    eval_config = cfg.get('evaluation', {})
    # eval_config = merge_configs(eval_config, dict(metric='3dpck'))

    if rank == 0:
        # if out:
        
        print(f'\nwriting results to {cfg.work_dir}/test_results.json')
        mmcv.dump(outputs, f'{cfg.work_dir}/test_results.json')

        results = dataset.evaluate(outputs, cfg.work_dir, **eval_config)
        # print(results)
        for k, v in sorted(results.items()):
            print(f'{k}: {v}')
            
def topdown_mmtrack(det_config,det_checkpoint,pose_config,pose_checkpoint,vid):
    save_out_video = True
    use_multi_frames=False
    online=True
    smooth=True
    smooth_filter_cfg='configs/_base_/filters/one_euro.py'
    device='cuda:0'
    bbox_thr=0.7
    kpt_thr=0.5
    use_oks_tracking=True
    tracking_thr=0.3
    euro=False
    det_cat_id=1
    show=False

    print('Initializing model...')
    det_model = init_detector(
        det_config, det_checkpoint, device=device.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        pose_config, pose_checkpoint, device=device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)
    # print(dataset_info.skeleton)
    # read video
    video = mmcv.VideoReader(vid)
    assert video.opened, f'Failed to load video file {vid}'

    if save_out_video:
        fps = video.fps
        size = (video.width, video.height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join('vis_results',
                         f'track_{os.path.basename(vid)}'), fourcc,fps, size)

    # frame index offsets for inference, used in multi-frame inference setting
    if use_multi_frames:
        assert 'frame_indices_test' in pose_model.cfg.data.test.data_cfg
        indices = pose_model.cfg.data.test.data_cfg['frame_indices_test']

    # build pose smoother for temporal refinement
    if euro:
        warnings.warn(
            'Argument --euro will be deprecated in the future. '
            'Please use --smooth to enable temporal smoothing, and '
            '--smooth-filter-cfg to set the filter config.',
            DeprecationWarning)
        smoother = Smoother(
            filter_cfg='configs/_base_/filters/one_euro.py', keypoint_dim=2)
    elif smooth:
        smoother = Smoother(filter_cfg=smooth_filter_cfg, keypoint_dim=2)
    else:
        smoother = None

    # whether to return heatmap, optional
    return_heatmap = False

    # return the output of some desired layers,
    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    next_id = 0
    pose_results = []
    print('Running inference...')
    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
        pose_results_last = pose_results

        # get the detection results of current frame
        # the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(det_model, cur_frame)

        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, det_cat_id)

        if use_multi_frames:
            frames = collect_multi_frames(video, frame_id, indices,
                                          online)

        # test a single image, with a list of bboxes.
        pose_results, _ = inference_top_down_pose_model(
            pose_model,
            frames if use_multi_frames else cur_frame,
            person_results,
            bbox_thr=bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=True,
            outputs=output_layer_names)
        
        # get track id for each person instance
        pose_results, next_id = get_track_id(
            pose_results,
            pose_results_last,
            next_id,
            use_oks=use_oks_tracking,
            tracking_thr=tracking_thr,
            sigmas=dataset_info.sigmas)

        # post-process the pose results with smoother

        pose_results = smoother.smooth(pose_results)
        if frame_id==10:
            print(frame_id)
        # print(returned_outputs)
        # show the results
        vis_frame = vis_pose_result(
            pose_model,
            cur_frame,
            pose_results,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=kpt_thr,
            show=False)

        if show:
            cv2.imshow('Frame', vis_frame)

        if save_out_video:
            videoWriter.write(vis_frame)

        if show and cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if save_out_video:
        videoWriter.release()
        
def bb_intersection_over_union(boxA, boxB):
    ##obtained from https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc
    
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou