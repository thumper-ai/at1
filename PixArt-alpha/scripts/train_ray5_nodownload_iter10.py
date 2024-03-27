import os
import sys
import types
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import argparse
import datetime
import time
import warnings
warnings.filterwarnings("ignore")  # ignore warning
import torch
import torch.nn as nn
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from diffusers.models import AutoencoderKL
from torch.utils.data import RandomSampler
from mmcv.runner import LogBuffer
from copy import deepcopy

from diffusion import IDDPM
from diffusion.utils.checkpoint import save_checkpoint, load_checkpoint
from diffusion.utils.dist_utils import synchronize, get_world_size, clip_grad_norm_
from diffusion.data.builder import build_dataset, build_dataloader, set_data_root
from diffusion.model.builder import build_model
from diffusion.utils.logger import get_root_logger
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow
from diffusion.utils.optimizer import build_optimizer, auto_scale_lr
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.data_sampler import AspectRatioBatchSampler, BalancedAspectRatioBatchSampler
from ray import train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, Checkpoint, CheckpointConfig, RunConfig
import ray 
# import numpy as np
import pyarrow 
import tempfile

def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = 'true'
    os.environ["FSDP_AUTO_WRAP_POLICY"] = 'TRANSFORMER_BASED_WRAP'
    os.environ["FSDP_BACKWARD_PREFETCH"] = 'BACKWARD_PRE'
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = 'T2IDiTBlock'


def ema_update(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)



def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("config", type=str, help="config")
    parser.add_argument("--cloud", action='store_true', default=False, help="cloud or local machine")
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the dir to save logs and models')
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--n_gpu_workers', type=int, default=4)

    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = read_config(args.config)
    config["args"]= args
    ray.init("auto")
    os.umask(0) 

    def download_dataset(head=False):
        import s3fs
        import pyarrow 
        imgs = []
        os.umask(0) 
        # os.umask(0o000)  # file permission: 666; dir permission: 777
        s3 = s3fs.S3FileSystem(#anon=True, 
        #                        use_listings_cache=False,
                            key='****************',
        #                        key='*****************',
                            secret='****************',
        #                        secret='*****************',
                            endpoint_url='ENTER_URL_HERE', version_aware=True)
        captions = s3.find('akash-thumper-v1-t5-data')
        # captions= np.array(captions)
        # captions= [captions[i] for i in range(20)]s
        if not os.path.exists("/home/ray/ray_results"):
            os.makedirs('/home/ray/ray_results/', int('777', base=8) ,exist_ok=True)
        else:
            os.chmod('/home/ray/ray_results/', int('777', base=8))


        if not os.path.exists("/home/ray/train_data/caption_feature_wmask/"):
            os.makedirs('/home/ray/train_data/caption_feature_wmask/', int('777', base=8) ,exist_ok=True)
        if not os.path.exists("/home/ray/train_data/images/"):
            os.makedirs('/home/ray/train_data/images/', int('777', base=8) , exist_ok=True )
        if not os.path.exists("/home/ray/train_data/partition/"):
            os.makedirs('/home/ray/train_data/partition/', int('777', base=8)  ,exist_ok=True)
          
        if not os.path.exists("/home/ray/train_data/img_vae_feature/train_vae_256/noflip/"):
            os.makedirs('/home/ray/train_data/img_vae_feature/train_vae_256/noflip/', int('777', base=8)  ,exist_ok=True)         
        # print("found caps", captions)
        
        # os.system(f"s5cmd cp akash-thumper-v1-t5-data/*.npz /home/ray/train_data/caption_feature_wmask/ ")
        # os.system(f"s5cmd cp akash-thumper-v1-training-images/*.jpg /home/ray/train_data/images/ ")

        # for i, cap in enumerate(captions):
        #     imgfilepath = cap.replace("akash-thumper-v1-t5-data", "akash-thumper-v1-training-images").replace(".npz", ".jpg")
        #     imgfilepath_save = f"/home/ray/train_data/images/{os.path.basename(imgfilepath)}"
        #     captionfeaturefilepath_save = f'/home/ray/train_data/caption_feature_wmask/{os.path.basename(imgfilepath).replace(".npz", ".jpg")}'
        #     if os.path.exists(imgfilepath_save):
        #         imgs.append(imgfilepath_save)
        #     if i % 5000 ==0:
        #         print(f"{len(imgs)} of {len(captions)}")

        for i, cap in enumerate(captions):
            imgfilepath = cap.replace("akash-thumper-v1-t5-data", "akash-thumper-v1-training-images").replace(".npz", ".jpg")
            imgfilepath_save = f"/home/ray/train_data/images/{os.path.basename(imgfilepath)}"
            vaefilepath = f"akash-thumper-v1-vae-data/{os.path.basename(cap)}".replace(".npz", ".npy")
            vaefilepath_save = f"/home/ray/train_data/img_vae_feature/train_vae_256/noflip/{os.path.basename(vaefilepath)}"
            if head:
                if s3.exists(cap) and s3.exists(vaefilepath) and s3.exists(imgfilepath):
                    try:
                # if s3.exists(cap) and os.path.exists( imgfilepath_save) :
                        print(cap, imgfilepath, imgfilepath_save)
                        if not head:
                            s3.get(cap, f"/home/ray/train_data/caption_feature_wmask/")
                            s3.get(imgfilepath,  imgfilepath_save) #f"/home/ray/train_data/images/"
                            print(f"vae {vaefilepath} {vaefilepath_save}")
                            s3.get(vaefilepath,  vaefilepath_save)    
                        
                        # s3.get(imgfilepath, f"/home/ray/train_data/images/" )
                        if i % 10:
                            print(f"{i} of {len(captions)}")
                        imgs.append(imgfilepath_save)
                    except Exception as e:
                        print(e)
            else:
                imgs.append(imgfilepath_save)

        # imgs = np.array(imgs)
        # np.savetxt('/home/ray/train_data/partition/part0.txt', imgs)
        if os.path.exists('/home/ray/train_data/partition/part0.txt'):
            try:
                os.remove('/home/ray/train_data/partition/part0.txt')
            except Exception as e:
                print(e)

        with open('/home/ray/train_data/partition/part0.txt', 'w') as f:
            for img in imgs:
                f.write(f"/home/ray/train_data/images/{img}\n")
                # f.write(f"/home/ray/train_data/images/{img[0:8]}/{img}\n")
        

    def train_func(config):
        if args.work_dir is not None:
            # update configs according to CLI args if args.work_dir is not None
            config.work_dir = args.work_dir
        if args.cloud:
            config.data_root = '/data/data'
            # config.data_root = '/data/data'

        if args.resume_from is not None:
            config.resume_from = dict(
                checkpoint=args.resume_from,
                load_ema=False,
                resume_optimizer=True,
                resume_lr_scheduler=True)
        if args.debug:
            config.log_interval = 20
            config.train_batch_size = 8
            config.valid_num = 100

        os.umask(0)  # file permission: 666; dir permission: 777
        os.makedirs(config.work_dir, exist_ok=True)


        if not os.path.exists('/home/ray/models'):
            print("downloading models")
            if not os.path.exists('/home/ray/models'):
                os.mkdir('/home/ray/models')
            # download_s3_folder2(bucket_name='akash-thumper-v1-files', s3_folder="models" , local_dir="/home/ray/models" )
            os.system("cd /home/ray/models && git lfs clone https://huggingface.co/PixArt-alpha/PixArt-alpha" )
            # os.system("cd /home/ray/models && git lfs clone https://huggingface.co/liuhaotian/llava-v1.5-7b")
        else:
            print("found model ")
          
        if not os.path.exists('/home/ray/models/PixArt-alpha/t5-v1_1-xxl/config.json'):
            try:
            # download_s3_folder2(bucket_name='akash-thumper-v1-files', s3_folder="models" , local_dir="/home/ray/models" )
                os.system("cd /home/ray/models && git lfs clone https://huggingface.co/PixArt-alpha/PixArt-alpha")
            except:
                os.system("cd /home/ray/models && git lfs pull https://huggingface.co/PixArt-alpha/PixArt-alpha")

            # os.system("cd /home/ray/models && git lfs pull https://huggingface.co/liuhaotian/llava-v1.5-7b")
        else:
            print("found model ")
        
        import s3fs
        s3 = s3fs.S3FileSystem(#anon=True, 
            #                        use_listings_cache=False,
                                key='****************',
            #                        key='*****************',
                                secret='****************',
            #                        secret='*****************',
                                endpoint_url='ENTER_URL_HERE', version_aware=True)
        s3.get('akash-thumper-v1-checkpoints/TorchTrainer_2023-12-13_19-05-02/TorchTrainer_93b28_00000_0_2023-12-13_19-05-03/checkpoint_000001/epoch_28_step_700.pth',
        "/home/ray/ray_results/")
        # download_dataset()

        init_handler = InitProcessGroupKwargs()
        init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug
        # Initialize accelerator and tensorboard logging
        if config.use_fsdp:
            init_train = 'FSDP'
            from accelerate import FullyShardedDataParallelPlugin
            from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
            set_fsdp_env()
            fsdp_plugin = FullyShardedDataParallelPlugin(state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),)
        else:
            init_train = 'DDP'
            fsdp_plugin = None

        even_batches = True
        if config.multi_scale:
            even_batches=False,
        #accelerator==0.19.0
        # accelerator = Accelerator(
        #     mixed_precision=config.mixed_precision,
        #     gradient_accumulation_steps=config.gradient_accumulation_steps,
        #     log_with="tensorboard",
        #     logging_dir=os.path.join(config.work_dir, "logs"),
        #     fsdp_plugin=fsdp_plugin,
        #     even_batches=even_batches,
        #     kwargs_handlers=[init_handler]
        # )
        accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="tensorboard",
            # project_dir=os.path.join(config.work_dir, "logs"),
            project_dir=config.work_dir,

            fsdp_plugin=fsdp_plugin,
            even_batches=even_batches,
            kwargs_handlers=[init_handler]
        )


        logger = get_root_logger(os.path.join(config.work_dir, 'train_log.log'))

        config.seed = init_random_seed(config.get('seed', None))
        set_random_seed(config.seed)

        if accelerator.is_main_process:
            config.dump(os.path.join(config.work_dir, 'config.py'))

        # logger.info(f"Config: \n{config.pretty_text}")
        logger.info(f"World_size: {get_world_size()}, seed: {config.seed}")
        logger.info(f"Initializing: {init_train} for training")
        image_size = config.image_size  # @param [256, 512]
        latent_size = int(image_size) // 8
        pred_sigma = getattr(config, 'pred_sigma', True)
        learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma
        model_kwargs={"window_block_indexes": config.window_block_indexes, "window_size": config.window_size,
                    "use_rel_pos": config.use_rel_pos, "lewei_scale": config.lewei_scale, 'config':config}

        # build models
        train_diffusion = IDDPM(str(config.train_sampling_steps), learn_sigma=learn_sigma, pred_sigma=pred_sigma, snr=config.snr_loss)
        model = build_model(config.model,
                            config.grad_checkpointing,
                            config.get('fp32_attention', False),
                            input_size=latent_size,
                            learn_sigma=learn_sigma,
                            pred_sigma=pred_sigma,
                            **model_kwargs).train()
        # model = ray.train.torch.prepare_model(model)

        logger.info(f"{model.__class__.__name__} Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
        model_ema = deepcopy(model).eval()

        if config.load_from is not None:
            missing, unexpected = load_checkpoint(config.load_from, model, load_ema=config.get('load_ema', False))
            if accelerator.is_main_process:
                print('Warning Missing keys: ', missing)
                print('Warning Unexpected keys', unexpected)

        ema_update(model_ema, model, 0.)
        if not config.data.load_vae_feat:
            vae = AutoencoderKL.from_pretrained( "/home/ray/models/PixArt-alpha/sd-vae-ft-ema"
                                                # config.vae_pretrained
                                                , local_files_only=True)
            vae = vae.to(ray.train.torch.get_device())

        # prepare for FSDP clip grad norm calculation
        if accelerator.distributed_type == DistributedType.FSDP:
            for m in accelerator._models:
                m.clip_grad_norm_ = types.MethodType(clip_grad_norm_, m)

        # build dataloader
        set_data_root(config.data_root)
        dataset = build_dataset(config.data, resolution=image_size, aspect_ratio_type=config.aspect_ratio_type)
        if config.multi_scale:
            batch_sampler = AspectRatioBatchSampler(sampler=RandomSampler(dataset), dataset=dataset,
                                                    batch_size=2, #config.train_batch_size
                                                    aspect_ratios=dataset.aspect_ratio, drop_last=True,
                                                    ratio_nums=dataset.ratio_nums, config=config, valid_num=config.valid_num)
            # used for balanced sampling
            # batch_sampler = BalancedAspectRatioBatchSampler(sampler=RandomSampler(dataset), dataset=dataset,
            #                                                 batch_size=config.train_batch_size, aspect_ratios=dataset.aspect_ratio,
            #                                                 ratio_nums=dataset.ratio_nums)
            # train_dataloader = build_dataloader(dataset, batch_sampler=batch_sampler, num_workers=config.num_workers)
            train_dataloader = build_dataloader(dataset, batch_sampler=batch_sampler, num_workers=2)
        else:
            train_dataloader = build_dataloader(dataset, num_workers=2, batch_size=176,
                                                 shuffle=False)
            # train_dataloader = build_dataloader(dataset, num_workers=config.num_workers, batch_size=config.train_batch_size, shuffle=True)

        # build optimizer and lr scheduler
        lr_scale_ratio = 1
        if config.get('auto_lr', None):
            lr_scale_ratio = auto_scale_lr(config.train_batch_size * get_world_size() * config.gradient_accumulation_steps,
                                        config.optimizer,
                                        **config.auto_lr)
        optimizer = build_optimizer(model, config.optimizer)
        lr_scheduler = build_lr_scheduler(config, optimizer, train_dataloader, lr_scale_ratio)

        timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

        if accelerator.is_main_process:
            accelerator.init_trackers(f"tb_{timestamp}")

        start_epoch = 0
        if config.resume_from is not None and config.resume_from['checkpoint'] is not None:
            start_epoch, missing, unexpected = load_checkpoint(**config.resume_from,
                                                            model=model,
                                                            model_ema=model_ema,
                                                            optimizer=optimizer,
                                                            lr_scheduler=lr_scheduler,
                                                            )

            if accelerator.is_main_process:
                print('Warning Missing keys: ', missing)
                print('Warning Unexpected keys', unexpected)
        # Prepare everything
        # There is no specific order to remember, you just need to unpack the
        # objects in the same order you gave them to the prepare method.
        model, model_ema = accelerator.prepare(model, model_ema)
        optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)
        if config.get('debug_nan', False):
            DebugUnderflowOverflow(model)
            logger.info('NaN debugger registered. Start to detect overflow during training.')



        time_start, last_tic = time.time(), time.time()
        log_buffer = LogBuffer()

        start_step = start_epoch * len(train_dataloader)
        global_step = 0
        total_steps = len(train_dataloader) * config.num_epochs

        load_vae_feat = getattr(train_dataloader.dataset, 'load_vae_feat', False)
        # Now you train the model
        for epoch in range(start_epoch + 1, config.num_epochs + 1):
            data_time_start= time.time()
            data_time_all = 0
            for step, batch in enumerate(train_dataloader):
                data_time_all += time.time() - data_time_start
                if load_vae_feat:
                    z = batch[0]
                else:
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=config.mixed_precision == 'fp16'):
                            posterior = vae.encode(batch[0]).latent_dist
                            if config.sample_posterior:
                                z = posterior.sample()
                            else:
                                z = posterior.mode()
                clean_images = z * config.scale_factor
                y = batch[1]
                y_mask = batch[2]
                data_info = batch[3]

                # Sample a random timestep for each image
                bs = clean_images.shape[0]
                timesteps = torch.randint(0, config.train_sampling_steps, (bs,), device=clean_images.device).long()
                grad_norm = None
                with accelerator.accumulate(model):
                    # Predict the noise residual
                    optimizer.zero_grad()
                    loss_term = train_diffusion.training_losses(model, clean_images, timesteps, model_kwargs=dict(y=y, mask=y_mask, data_info=data_info))
                    loss = loss_term['loss'].mean()
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.gradient_clip)
                    optimizer.step()
                    lr_scheduler.step()
                    if accelerator.sync_gradients:
                        ema_update(model_ema, model, config.ema_rate)

                lr = lr_scheduler.get_last_lr()[0]
                logs = {"loss": accelerator.gather(loss).mean().item()}
                if grad_norm is not None:
                    logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
                log_buffer.update(logs)
                if (step + 1) % config.log_interval == 0 or (step + 1) == 1:
                    t = (time.time() - last_tic) / config.log_interval
                    t_d = data_time_all / config.log_interval
                    avg_time = (time.time() - time_start) / (global_step + 1)
                    eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - start_step - global_step - 1))))
                    eta_epoch = str(datetime.timedelta(seconds=int(avg_time * (len(train_dataloader) - step - 1))))
                    # avg_loss = sum(loss_buffer) / len(loss_buffer)
                    log_buffer.average()
                    info = f"Step/Epoch [{(epoch-1)*len(train_dataloader)+step+1}/{epoch}][{step + 1}/{len(train_dataloader)}]:total_eta: {eta}, " \
                        f"epoch_eta:{eta_epoch}, time_all:{t:.3f}, time_data:{t_d:.3f}, lr:{lr:.3e}, s:({model.module.h}, {model.module.w}), "
                    info += ', '.join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                    logger.info(info)
                    last_tic = time.time()
                    log_buffer.clear()
                    data_time_all = 0
                logs.update(lr=lr)
                accelerator.log(logs, step=global_step + start_step)

                if (global_step + 1) % 1000 == 0 and config.s3_work_dir is not None:
                    logger.info(f"s3_work_dir: {config.s3_work_dir}")

                global_step += 1
                data_time_start= time.time()

                synchronize()
                # After each epoch you optionally sample some demo images with evaluate() and save the model
                # if accelerator.is_main_process:
                #     with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                #         if ((epoch - 1) * len(train_dataloader) + step + 1) % config.save_model_steps == 0:
                #             os.umask(0o000)  # file permission: 666; dir permission: 777
                #             save_checkpoint(temp_checkpoint_dir,
                #                             epoch=epoch,
                #                             step=(epoch - 1) * len(train_dataloader) + step + 1,
                #                             model=accelerator.unwrap_model(model),
                #                             model_ema=accelerator.unwrap_model(model_ema),
                #                             optimizer=optimizer,
                #                             lr_scheduler=lr_scheduler
                #                             )
                synchronize()

            synchronize()
            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if accelerator.is_main_process:
                if epoch % config.save_model_epochs == 0 or epoch == config.num_epochs or epoch % 2 ==0:
                    os.umask(0)  # file permission: 666; dir permission: 777
                    save_dir = f"/home/ray/ray_results/checkpoint_{epoch}/"
                    os.makedirs(save_dir)
                    # with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                    save_checkpoint(save_dir,
                                    epoch=epoch,
                                    step=(epoch - 1) * len(train_dataloader) + step + 1,
                                    model=accelerator.unwrap_model(model),
                                    model_ema=accelerator.unwrap_model(model_ema),
                                    optimizer=optimizer,
                                    lr_scheduler=lr_scheduler
                                    )
                    ray.train.report({"loss": loss.item()}, checkpoint=Checkpoint.from_directory(save_dir))
            # if accelerator.is_main_process:
            #     if epoch % config.save_model_epochs == 0 or epoch == config.num_epochs or epoch % 2 ==0:
            #         os.umask(0)  # file permission: 666; dir permission: 777
            #         save_dir = f"/home/ray/ray_results/checkpoint_{epoch}/"
            #         os.makedirs(save_dir)
            #         # with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            #         save_checkpoint(save_dir,
            #                         epoch=epoch,
            #                         step=(epoch - 1) * len(train_dataloader) + step + 1,
            #                         model=accelerator.unwrap_model(model),
            #                         model_ema=accelerator.unwrap_model(model_ema),
            #                         optimizer=optimizer,
            #                         lr_scheduler=lr_scheduler
            #                         )
            #         ray.train.report({"loss": loss.item(), "step":(epoch - 1) * len(train_dataloader) + step + 1} )
                        # print(f"uploading {temp_checkpoint_dir}")
                        # os.system(f"b2 sync {temp_checkpoint_dir} b2://akash-thumper-v1-checkpoints --skipNewer")

            synchronize()

    import s3fs
    s3 = s3fs.S3FileSystem(#anon=True, 
            #                        use_listings_cache=False,
                                key='****************',
            #                        key='*****************',
                                secret='****************',
            #                        secret='*****************',
                                endpoint_url='ENTER_URL_HERE', version_aware=True)
    s3.get('akash-thumper-v1-checkpoints/TorchTrainer_2023-12-13_19-05-02/TorchTrainer_93b28_00000_0_2023-12-13_19-05-03/checkpoint_000001/epoch_28_step_700.pth',
        "/home/ray/ray_results/")
    # download_dataset(head = True)
    # [4] Configure scaling and resource requirements.
    scaling_config = ScalingConfig(num_workers=config["args"].n_gpu_workers, use_gpu=True)

    # [5] Launch distributed training job.
    # trainer = TorchTrainer(train_func, scaling_config=scaling_config)
    trainer = TorchTrainer(
        train_func,
        train_loop_config=config,
        run_config=RunConfig(
        storage_path=f"s3://akash-thumper-v1-checkpoints?endpoint_override={os.environ['S3_ENDPOINT_URL']}",
        # checkpoint_config=CheckpointConfig(
        #     num_to_keep=200,
        #     checkpoint_score_attribute="loss",
        #     checkpoint_score_order="min",
        # ),
    ),
        # datasets=ray_datasets,
        # dataset_config=DataConfig(datasets_to_split=["train", "validation"]),
        scaling_config=ScalingConfig(num_workers=args.n_gpu_workers, use_gpu=True, resources_per_worker={"GPU": 1, "CPU":2 }
                                     )
    )

    result = trainer.fit()

    # TorchTrainer_2023-12-13_02-12-35/TorchTrainer_24002_00000_0_2023-12-13_02-12-37/checkpoint_000001/epoch_8_step_1072.pth

    # akash-thumper-v1-checkpoints/TorchTrainer_2023-12-13_11-44-23/TorchTrainer_044c1_00000_0_2023-12-13_11-44-23/checkpoint_000001/epoch_16_step_400.pth
 #akash-thumper-v1-checkpoints/TorchTrainer_2023-12-13_13-42-22/TorchTrainer_800b0_00000_0_2023-12-13_13-42-23/checkpoint_000001/epoch_20_step_500.pth
#akash-thumper-v1-checkpoints/TorchTrainer_2023-12-13_17-15-17/TorchTrainer_3f046_00000_0_2023-12-13_17-15-19/checkpoint_000001/epoch_24_step_600.pth
#akash-thumper-v1-checkpoints/TorchTrainer_2023-12-13_19-05-02/TorchTrainer_93b28_00000_0_2023-12-13_19-05-03/checkpoint_000001/epoch_28_step_700.pth
#akash-thumper-v1-checkpoints/TorchTrainer_2023-12-14_08-42-31/TorchTrainer_c76b4_00000_0_2023-12-14_08-42-33/checkpoint_000001/epoch_32_step_800.pth