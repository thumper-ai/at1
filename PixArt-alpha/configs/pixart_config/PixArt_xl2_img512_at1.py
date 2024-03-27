_base_ = ['../PixArt_xl2_internal.py']
data_root = '/mnt/sabrent/at1_test'
image_list_json = ['info.json',]

data = dict(type='InternalData', root='/mnt/sabrent/at1_test', image_list_json=image_list_json, transform='default_train', load_vae_feat=True)
image_size = 512

# model setting
window_block_indexes = []
window_size=0
use_rel_pos=False
model = 'PixArt_XL_2'
fp32_attention = True
load_from = None
vae_pretrained = "/home/logan/thumperai/PixArt-alpha/output/pretrained_models/sd-vae-ft-ema"
lewei_scale = 1.0

    # parser.add_argument('--caption_filepath', default='/home/logan/thumperai/test.csv', type=str)
    # parser.add_argument('--vae_dir', default="/home/logan/thumperai/PixArt-alpha/output/pretrained_models/sd-vae-ft-ema", type=str)
    # parser.add_argument('--t5_dir', default="/home/logan/thumperai/PixArt-alpha/output/pretrained_models/t5-v1_1-xxl", type=str)
    # parser.add_argument('--img2img_dir', default="/mnt/sabrent/cc0img", type=str)

# training setting
use_fsdp=False   # if use FSDP mode
num_workers=1
train_batch_size = 1 # 32
num_epochs = 200 # 3
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.01
optimizer = dict(type='AdamW', lr=2e-5, weight_decay=3e-2, eps=1e-10)
lr_schedule_args = dict(num_warmup_steps=1000)

eval_sampling_steps = 200
log_interval = 20
save_model_epochs=1
work_dir = 'output/debug'
