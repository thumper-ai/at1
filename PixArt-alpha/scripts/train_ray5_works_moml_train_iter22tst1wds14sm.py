import copy, os

import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, asdict

import wandb
import torchvision.utils as vutils
import tempfile

from accelerate import Accelerator
import torchvision

import torch 
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# from diffusers.models import AutoencoderKL
from diffusers import AutoencoderKL
from diffusers.utils import import_utils
from diffusers.utils.torch_utils import randn_tensor
import diffusers

from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, Checkpoint, CheckpointConfig, RunConfig

# from tld.denoiser import Denoiser
# from tld.diffusion import DiffusionGenerator

import torch
from torch import nn
from einops.layers.torch import Rearrange

# from tld.transformer_blocks import DecoderBlock, MLPSepConv, SinusoidalEmbedding

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

import torch
from tqdm import tqdm
import ray 

class DiffusionGenerator:
    def __init__(self, model, vae, device, model_dtype=torch.float32):
        self.model = model
        self.vae = vae
        self.device = device
        self.model_dtype = model_dtype


    # @torch.no_grad()
    # def generate(self, 
    #              n_iter=30, 
    #              labels=None, #embeddings to condition on
    #              num_imgs=16, 
    #              class_guidance=3,
    #              seed=10,  #for reproducibility
    #              scale_factor=8, #latent scaling before decoding - should be ~ std of latent space
    #              img_size=32, #height, width of latent
    #              sharp_f=0.1, 
    #              bright_f=0.1, 
    #              exponent=1,
    #              seeds=None):
    #     """Generate images via reverse diffusion."""
    #     noise_levels = (1 - torch.pow(torch.arange(0, 1, 1 / n_iter), exponent)).tolist()
    #     new_img = self.initialize_image(seeds, num_imgs, img_size, seed)

    #     labels = torch.cat([labels, torch.zeros_like(labels)])
    #     self.model.eval()

    #     for i in tqdm(range(len(noise_levels) - 1)):
    #         curr_noise, next_noise = noise_levels[i], noise_levels[i + 1]
    #         noises = torch.full((2*num_imgs, 1), curr_noise)

    #         x0_pred = self.model(torch.cat([new_img, new_img]),
    #                              noises.to(self.device, self.model_dtype),
    #                              labels.to(self.device, self.model_dtype))

    #         x0_pred = self.apply_classifier_free_guidance(x0_pred, num_imgs, class_guidance)

    #         new_img = ((curr_noise - next_noise) * x0_pred + next_noise * new_img) / curr_noise

    #     x0_pred[:, 3, :, :] += sharp_f
    #     x0_pred[:, 0, :, :] += bright_f

    #     x0_pred_img = self.vae.decode((x0_pred*scale_factor).half())[0].cpu()
    #     return x0_pred_img, x0_pred


    @torch.no_grad()
    def generate(self, 
                 n_iter=30, 
                 labels=None, #embeddings to condition on
                 num_imgs=16, 
                 class_guidance=3,
                 seed=10,  #for reproducibility
                 scale_factor=8, #latent scaling before decoding - should be ~ std of latent space
                 img_size=32, #height, width of latent
                 sharp_f=0.1, 
                 bright_f=0.1, 
                 exponent=1,
                 seeds=None):
        """Generate images via reverse diffusion."""
        print('labelsize', labels.size())
        noise_levels = (1 - torch.pow(torch.arange(0, 1, 1 / n_iter), exponent)).tolist()
        new_img = self.initialize_image(seeds, num_imgs, img_size, seed)

        labels = torch.cat([labels, torch.zeros_like(labels)])
        self.model.eval()

        for i in tqdm(range(len(noise_levels) - 1)):
            curr_noise, next_noise = noise_levels[i], noise_levels[i + 1]
            noises = torch.full((2*num_imgs, 1), curr_noise)

            x0_pred = self.model(torch.cat([new_img, new_img]),
                                 noises.to(self.device, self.model_dtype),
                                 labels.to(self.device, self.model_dtype))

            x0_pred = self.apply_classifier_free_guidance(x0_pred, num_imgs, class_guidance)

            new_img = ((curr_noise - next_noise) * x0_pred + next_noise * new_img) / curr_noise

        x0_pred[:, 3, :, :] += sharp_f
        x0_pred[:, 0, :, :] += bright_f

        x0_pred_img = self.vae.decode((x0_pred*scale_factor).half())[0].cpu()
        return x0_pred_img, x0_pred

    def initialize_image(self, seeds, num_imgs, img_size, seed):
        """Initialize the seed tensor."""
        if seeds is None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            return torch.randn(num_imgs, 4, img_size, img_size, dtype=self.model_dtype, 
                               device=self.device, generator=generator)
        else:
            return seeds.to(self.device, self.model_dtype)

    def apply_classifier_free_guidance(self, x0_pred, num_imgs, class_guidance):
        """Apply classifier-free guidance to the predictions."""
        x0_pred_label, x0_pred_no_label = x0_pred[:num_imgs], x0_pred[num_imgs:]
        return class_guidance * x0_pred_label + (1 - class_guidance) * x0_pred_no_label


class SinusoidalEmbedding(nn.Module):
    def __init__(self, emb_min_freq=1.0, emb_max_freq=1000.0, embedding_dims=32):
        super(SinusoidalEmbedding, self).__init__()

        frequencies = torch.exp(
            torch.linspace(np.log(emb_min_freq), np.log(emb_max_freq),
                embedding_dims // 2))

        self.register_buffer('angular_speeds', 2.0 * torch.pi * frequencies)

    def forward(self, x):
        embeddings = torch.cat([torch.sin(self.angular_speeds * x),
                                torch.cos(self.angular_speeds * x)], dim=-1)
        return embeddings

class MHAttention(nn.Module):
    def __init__(self, is_causal=False, dropout_level=0, n_heads=4):
        super().__init__()
        self.is_causal = is_causal
        self.dropout_level = dropout_level
        self.n_heads = n_heads

    def forward(self, q, k, v, attn_mask=None):

        assert q.size(-1) == k.size(-1)
        assert k.size(-2) == v.size(-2)

        q, k, v = [rearrange(x, 'bs n (d h) -> bs h n d', h=self.n_heads) for x in [q,k,v]]
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        out = nn.functional.scaled_dot_product_attention(q, k, v,
                                                          attn_mask=attn_mask,
                                                          is_causal=self.is_causal,
                                                          dropout_p=self.dropout_level if self.training else 0)

        out = rearrange(out, 'bs h n d -> bs n (d h)', h=self.n_heads)

        return out

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, is_causal=False, dropout_level=0, n_heads=4):
        super().__init__()
        self.qkv_linear = nn.Linear(embed_dim, 3*embed_dim, bias=False)
        self.mha = MHAttention(is_causal, dropout_level, n_heads)

    def forward(self, x):
        q, k, v = self.qkv_linear(x).chunk(3, dim=2)
        return self.mha(q, k, v)

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, is_causal=False, dropout_level=0, n_heads=4):
        super().__init__()
        self.kv_linear = nn.Linear(embed_dim, 2*embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mha = MHAttention(is_causal, dropout_level, n_heads)

    def forward(self, x, y):
        q = self.q_linear(x)
        k, v = self.kv_linear(y).chunk(2, dim=2)
        return self.mha(q,k,v)

class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_multiplier, dropout_level):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_multiplier * embed_dim),
            nn.GELU(),
            nn.Linear(mlp_multiplier * embed_dim, embed_dim),
            nn.Dropout(dropout_level)
        )

    def forward(self, x):
        return self.mlp(x)

class MLPSepConv(nn.Module):
    def __init__(self, embed_dim, mlp_multiplier, dropout_level):
        """see: https://github.com/ofsoundof/LocalViT"""
        super().__init__()
        self.mlp = nn.Sequential(
            #this Conv with kernel size 1 is equivalent to the Linear layer in a "regular" transformer MLP
            nn.Conv2d(embed_dim, mlp_multiplier*embed_dim, kernel_size=1, padding='same'),
            nn.Conv2d(mlp_multiplier*embed_dim, mlp_multiplier*embed_dim, kernel_size=3,
                      padding='same', groups=mlp_multiplier*embed_dim), #<- depthwise conv
            nn.GELU(),
            nn.Conv2d(mlp_multiplier*embed_dim, embed_dim, kernel_size=1, padding='same'),
            nn.Dropout(dropout_level)
            )

    def forward(self, x):
        w = h = int(np.sqrt(x.size(1))) #only square images for now
        x = rearrange(x, 'bs (h w) d -> bs d h w', h=h, w=w)
        x = self.mlp(x)
        x = rearrange(x, 'bs d h w -> bs (h w) d')
        return x

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, is_causal, mlp_multiplier, dropout_level, mlp_class=MLP):
        super().__init__()
        self.self_attention = SelfAttention(embed_dim, is_causal, dropout_level, n_heads=embed_dim//64)
        self.cross_attention = CrossAttention(embed_dim, is_causal=False, dropout_level=0, n_heads=4)
        self.mlp = mlp_class(embed_dim, mlp_multiplier, dropout_level)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x, y):
        x = self.norm1(self.self_attention(x) + x)
        x = self.norm2(self.cross_attention(x, y) + x)
        x = self.norm3(self.mlp(x) + x)
        return x

class DenoiserTransBlock(nn.Module):
    def __init__(self, patch_size, img_size, embed_dim, dropout, n_layers, mlp_multiplier=4, n_channels=4):
        super().__init__()

        self.patch_size = patch_size
        self.img_size = img_size
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.n_layers = n_layers
        self.mlp_multiplier = mlp_multiplier

        seq_len = int((self.img_size/self.patch_size)*((self.img_size/self.patch_size)))
        patch_dim = self.n_channels*self.patch_size*self.patch_size

        self.patchify_and_embed = nn.Sequential(
                                       nn.Conv2d(self.n_channels, patch_dim, kernel_size=self.patch_size, stride=self.patch_size),
                                       Rearrange('bs d h w -> bs (h w) d'),
                                       nn.LayerNorm(patch_dim),
                                       nn.Linear(patch_dim, self.embed_dim),
                                       nn.LayerNorm(self.embed_dim)
                                       )

        self.rearrange2 = Rearrange('b (h w) (c p1 p2) -> b c (h p1) (w p2)',
                                   h=int(self.img_size/self.patch_size),
                                   p1=self.patch_size, p2=self.patch_size)


        self.pos_embed = nn.Embedding(seq_len, self.embed_dim)
        self.register_buffer('precomputed_pos_enc', torch.arange(0, seq_len).long())

        self.decoder_blocks = nn.ModuleList([DecoderBlock(embed_dim=self.embed_dim,
                                            mlp_multiplier=self.mlp_multiplier,
                                            #note that this is a non-causal block since we are 
                                            #denoising the entire image no need for masking
                                            is_causal=False,
                                            dropout_level=self.dropout,
                                            mlp_class=MLPSepConv)
                                              for _ in range(self.n_layers)])

        self.out_proj = nn.Sequential(nn.Linear(self.embed_dim, patch_dim),
                                                self.rearrange2)


    def forward(self, x, cond):
        x = self.patchify_and_embed(x)
        pos_enc = self.precomputed_pos_enc[:x.size(1)].expand(x.size(0), -1)
        x = x+self.pos_embed(pos_enc)

        for block in self.decoder_blocks:
            x = block(x, cond)

        return self.out_proj(x)

class Denoiser(nn.Module):
    def __init__(self,
                 image_size, noise_embed_dims, patch_size, embed_dim, dropout, n_layers,
                 text_emb_size=768):
        super().__init__()

        self.image_size = image_size
        self.noise_embed_dims = noise_embed_dims
        self.embed_dim = embed_dim

        self.fourier_feats = nn.Sequential(SinusoidalEmbedding(embedding_dims=noise_embed_dims),
                                           nn.Linear(noise_embed_dims, self.embed_dim),
                                           nn.GELU(),
                                           nn.Linear(self.embed_dim, self.embed_dim)
                                           )

        self.denoiser_trans_block = DenoiserTransBlock(patch_size, image_size, embed_dim, dropout, n_layers)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.label_proj = nn.Linear(text_emb_size, self.embed_dim)

    def forward(self, x, noise_level, label):

        noise_level = self.fourier_feats(noise_level).unsqueeze(1)

        label = self.label_proj(label).unsqueeze(1)

        noise_label_emb = torch.cat([noise_level, label], dim=1) #bs, 2, d
        noise_label_emb = self.norm(noise_label_emb)

        x = self.denoiser_trans_block(x,noise_label_emb)

        return x
    
def test_outputs():
    model = Denoiser(image_size=16, noise_embed_dims=128, patch_size=2, embed_dim=256, dropout=0.1, n_layers=6)
    x = torch.rand(8, 4, 16, 16)
    noise_level = torch.rand(8, 1)
    label = torch.rand(8, 768)

    with torch.no_grad():
        output = model(x, noise_level, label)

    assert output.shape == torch.Size([8, 4, 16, 16])
    print("Basic tests passed.")

# def eval_gen(diffuser, labels):
#     class_guidance=4.5
#     seed=10
#     print("labels", labels.size())
#     out, _ = diffuser.generate(labels= labels.squeeze(), #torch.repeat_interleave(labels, 8, dim=0),
#                                         num_imgs=64,
#                                         class_guidance=class_guidance,
#                                         seed=seed,
#                                         n_iter=40,
#                                         exponent=1,
#                                         sharp_f=0.1,
#                                         )

#     out = to_pil((vutils.make_grid((out+1)/2, nrow=8, padding=4)).float().clip(0, 1))
#     out.save(f'emb_val_cfg:{class_guidance}_seed:{seed}.png')

#     return out

def eval_gen(diffuser, labels):
    class_guidance=4.5
    seed=10
    print("labels", labels.size())
    out, _ = diffuser.generate(labels= labels.squeeze(), #torch.repeat_interleave(labels, 8, dim=0),
                                        num_imgs=labels.size()[0],
                                        class_guidance=class_guidance,
                                        seed=seed,
                                        n_iter=40,
                                        exponent=1,
                                        sharp_f=0.1,
                                        )

    out = to_pil((vutils.make_grid((out+1)/2, nrow=8, padding=4)).float().clip(0, 1))
    out.save(f'emb_val_cfg:{class_guidance}_seed:{seed}.png')

    return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters_per_layer(model):
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()} parameters")

to_pil = torchvision.transforms.ToPILImage()

def update_ema(ema_model, model, alpha=0.999):
    with torch.no_grad():
        for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(model_param.data, alpha=1-alpha)


@dataclass
class ModelConfig:
    embed_dim: int = 512
    n_layers: int = 6
    clip_embed_size: int = 768
    scaling_factor: int = 8
    patch_size: int = 2
    image_size: int = 32 
    n_channels: int = 4
    dropout: float = 0
    mlp_multiplier: int = 4
    batch_size: int = 128
    class_guidance: int = 3
    lr: float = 3e-4
    n_epoch: int = 100
    alpha: float = 0.999
    noise_embed_dims: int = 128
    diffusion_n_iter: int = 35
    from_scratch: bool = True
    run_id: str = None
    model_name: str = None
    beta_a: float = 0.75
    beta_b: float = 0.75
    data_dir:str =""
    model_dir:str=""

@dataclass
class DataConfig:
    latent_path: str #path to a numpy file containing latents
    text_emb_path: str
    val_path: str


def train_func(config):
    from diffusers import AutoencoderKL
    from diffusers.utils import import_utils
    from diffusers.utils.torch_utils import randn_tensor
    import diffusers
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir, exist_ok=True)
    
    os.environ["WANDB_API_KEY"]= 'b710005fcef0b6bb3ae5a33ce530af5b93a93844'
    os.environ["WANDB_ENTITY"]= 'lcerkovnik'
    os.environ["WANDB_PROJECT"]=  'rakt-test'
    # if not os.path.exists("/home/ray/train_data/wds_at1--0.tar"):
    #     s3.get("s3://akash-thumper-v1-wds-256/wds_at1--0.tar", "/home/ray/train_data/")

    if not os.path.exists(config.data_dir):
        os.makedirs(config.data_dir, exist_ok=True)

    for a in range(0, 2210):
        if not os.path.exists(f"{config.data_dir}/wds_at1--{a}.tar"):
            s3.get(f"s3://akash-thumper-v3-wds-256/wds_at1--{a}.tar", config.data_dir)    


    accelerator = Accelerator(mixed_precision="fp16", log_with="wandb")

    accelerator.print("Loading Data:")

    train_data_shard = ray.train.get_dataset_shard("train")
    train_loader = train_data_shard.iter_torch_batches(
        prefetch_batches =  1, 
        device = accelerator.device, drop_last=True,
        # collate_fn= collate,
        batch_size=1, dtypes={"image_latents.pyd": torch.float16, "text_encodings.pyd": torch.float16} #dtypes={"vae": torch.float32, "txt_features": torch.float32, "attention_mask": torch.int32}
    )

    # train_dataset_len = int(round(config["dataset_len"] // ray.train.get_context().get_world_size()))
    # print(f"creating datashard len {train_dataset_len } / { config['dataset_len'] } ")


    vae = diffusers.models.AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", cache_dir= config.model_dir,
                                    torch_dtype=torch.float16)
    
    # if accelerator.is_main_process:
    vae = vae.to(accelerator.device)
   
    model = Denoiser(image_size=config.image_size, noise_embed_dims=config.noise_embed_dims,
                 patch_size=config.patch_size, embed_dim=config.embed_dim, dropout=config.dropout,
                 n_layers=config.n_layers)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    if not config.from_scratch:
        accelerator.print("Loading Model:")
        wandb.restore(config.model_name, run_path=f"lcerkovnik/at1256/runs/{config.run_id}",
                      replace=True)
        full_state_dict = torch.load(config.model_name)
        model.load_state_dict(full_state_dict['model_ema'])
        optimizer.load_state_dict(full_state_dict['opt_state'])
        global_step = full_state_dict['global_step']
    else:
        global_step = 0

    if accelerator.is_local_main_process:
        ema_model = copy.deepcopy(model).to(accelerator.device)
        diffuser = DiffusionGenerator(ema_model, vae,  accelerator.device, torch.float32)

    accelerator.print("model prep")
    model, train_loader, optimizer = accelerator.prepare(
        model, train_loader, optimizer
    )

    accelerator.init_trackers(
    project_name="at1256",
    config=asdict(config)
    )

    accelerator.print(count_parameters(model))
    accelerator.print(count_parameters_per_layer(model))


    ### Train:
    for i in range(1, 1000+1):
        accelerator.print(f'epoch: {i}')            
        for batch in train_loader:
            x = batch["image_latents.pyd"].squeeze(0)#[0:128,]#.to(accelerator.device)
            y=  batch["text_encodings.pyd"].squeeze(0)#[0:128,]#.to(accelerator.device)
            x = x/config.scaling_factor

            noise_level = torch.tensor(np.random.beta(config.beta_a, config.beta_b, len(x)),
                                        device=accelerator.device)
            signal_level = 1 - noise_level
            noise = torch.randn_like(x)

            x_noisy = noise_level.view(-1,1,1,1)*noise + signal_level.view(-1,1,1,1)*x

            x_noisy = x_noisy.float()
            noise_level = noise_level.float()
            label = y

            prob = 0.15
            mask = torch.rand(y.size(0), device=accelerator.device) < prob
            label[mask] = 0 # OR replacement_vector

            if global_step % 5000 == 0:
                # accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    ##eval and saving:
                    save_dir = f"/home/ray/ray_results/sm_checkpoint_{i}/"
                    os.makedirs(save_dir, exist_ok=True)
                    print('labelsize', label.size())
                    out = eval_gen(diffuser=diffuser, labels=label.squeeze())
                    out.save('img.jpg')
                    accelerator.log({f"step: {global_step}": wandb.Image("img.jpg")})
                    
                    opt_unwrapped = accelerator.unwrap_model(optimizer)
                    full_state_dict = {'model_ema':ema_model.state_dict(),
                                    'opt_state':opt_unwrapped.state_dict(),
                                    'global_step':global_step
                                    }
                    filepath =  f"{save_dir}/{config.model_name}_ckpt_{i}.bin"

                    accelerator.save(full_state_dict, filepath)
                    wandb.save(filepath)

            model.train()

            with accelerator.accumulate(model):
                ###train loop:
                optimizer.zero_grad()

                pred = model(x_noisy, noise_level.view(-1,1), label)
                loss = loss_fn(pred.half(), x)
                accelerator.log({"train_loss":loss.item(), "step":"global_step"})
                ray.train.report({"train_loss":loss.item(), "step":"global_step"})
                accelerator.backward(loss)
                optimizer.step()

                if accelerator.is_main_process:
                    update_ema(ema_model, model, alpha=config.alpha)

            global_step += 1
    accelerator.end_training()


    # ### Train:
    # for i in range(1, config.n_epoch+1):
    #     accelerator.print(f'epoch: {i}')            

    #     for batch in train_loader:

    #         # x = torch.from_numpy(batch["image_latents.npy"]).to(accelerator.device)
    #         # y=  torch.from_numpy(batch["text_encodings.npy"]).to(accelerator.device)
    #         x = batch["image_latents.pyg"].squeeze()#.to(accelerator.device)
    #         label=  batch["text_encodings.pyd"].squeeze()#.to(accelerator.device)
    #         accelerator.print( label.size(), x.size() )
    #         accelerator.print( "x", x.dtype, "label",label.dtype )

    #         # if x.size()[0] == label.size()[0]:
    #         x = x/config.scaling_factor

    #         noise_level = torch.tensor(np.random.beta(config.beta_a, config.beta_b, len(x)),
    #                                     device=accelerator.device)
    #         signal_level = 1 - noise_level
    #         noise = torch.randn_like(x, device=accelerator.device)
            
    #         accelerator.print(label.size(), x.size(), noise_level.size(), signal_level.size())

    #         x_noisy = noise_level.view(-1,1,1,1)*noise + signal_level.view(-1,1,1,1)*x

    #         x_noisy = x_noisy.half()
    #         noise_level = noise_level.half()
    #         # label = y.half()

    #         prob = 0.15
    #         mask = torch.rand(label.size(0), device=accelerator.device) < prob
    #         label[mask] = 0 # OR replacement_vector

    #         accelerator.print(x_noisy.size(), noise_level.size(), label.size() )
    #         accelerator.print("xnoisy", x_noisy.dtype, "noiselevel", noise_level.dtype, "label",label.dtype )

    #         if global_step % 10 == 0:
    #             accelerator.wait_for_everyone()
    #             if accelerator.is_main_process:
    #                 os.umask(0)  # file permission: 666; dir permission: 777
    #                 save_dir = f"/home/ray/ray_results/checkpoint_{i}/"
    #                 os.makedirs(save_dir, exist_ok=True)
    #                 # with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
    #                     ##eval and saving:
    #                 out = eval_gen(diffuser=diffuser, labels=label)
    #                 out.save('img.jpg')
    #                 accelerator.log({f"step: {global_step}": wandb.Image("img.jpg")})
    #                 opt_unwrapped = accelerator.unwrap_model(optimizer)
    #                 full_state_dict = {'model_ema':ema_model.state_dict(),
    #                                 'opt_state':opt_unwrapped.state_dict(),
    #                                 'global_step':global_step
    #                                 }
    #                 filepath =  f"{save_dir}/{config.model_name}_ckpt_{i}.bin"
    #                 accelerator.save(full_state_dict, filepath )
    #                 wandb.save(filepath)
    #                     # save_checkpoint(temp_checkpoint_dir,
    #                     #                 epoch=i,
    #                     #                 step= global_step,
    #                     #                 model=accelerator.unwrap_model(model),
    #                     #                 model_ema=accelerator.unwrap_model(model_ema),
    #                     #                 # model= model, #accelerator.unwrap_model(model),
    #                     #                 # model_ema= model_ema, #accelerator.unwrap_model(model_ema),
    #                     #                 optimizer=optimizer,
    #                     #                 lr_scheduler=lr_scheduler
    #                     #                 )
    #                     # ray.train.report({"epoch": int(i), "step": int(global_step)}, checkpoint=Checkpoint.from_directory(temp_checkpoint_dir) )
    #                     # os.system(f"b2 cp {filepath} b2://akash-thumper-v1-checkpoints")
    #             # else:
    #             #     ray.train.report({"epoch": int(i), "step": int(global_step)}) # checkpoint=C\

    #             # if accelerator.is_main_process:
    #             #     ##eval and saving:
    #             #     # out = eval_gen(diffuser=diffuser, labels=emb_val)
    #             #     # out.save('img.jpg')
    #             #     # accelerator.log({f"step: {global_step}": wandb.Image("img.jpg")})
    #             #     opt_unwrapped = accelerator.unwrap_model(optimizer)
    #             #     full_state_dict = {'model_ema':ema_model.state_dict(),
    #             #                     'opt_state':opt_unwrapped.state_dict(),
    #             #                     'global_step':global_step
    #             #                     }
    #             #     accelerator.save(full_state_dict, config.model_name)
    #             #     wandb.save(config.model_name)

    #         model.train()

    #         with accelerator.accumulate(model):
    #             ###train loop:
    #             optimizer.zero_grad()
    #             # accelerator.log({"xnoisy": x_noisy.dtype, "noiselevel":noise_level.dtype, "label":label.dtype} )

    #             pred = model(x_noisy, noise_level.view(-1,1), label)
    #             # loss = loss_fn(pred, x.float())
    #             loss = loss_fn(pred.half(), x)
    #             accelerator.print("pred", pred.dtype, x.dtype, "train_loss", loss.item())

    #             accelerator.log({"train_loss":loss.item()}, step=global_step)
    #             accelerator.backward(loss)
    #             optimizer.step()

    #             if accelerator.is_main_process:
    #                 update_ema(ema_model, model, alpha=config.alpha)

    #         global_step += 1
    #     if accelerator.is_main_process:
    #         accelerator
    #         os.umask(0)  # file permission: 666; dir permission: 777
    #         save_dir = f"/home/ray/ray_results/checkpoint_{i}/"
    #         os.makedirs(save_dir, exist_ok=True)

    #         # with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
    #             ##eval and saving:
    #             # out = eval_gen(diffuser=diffuser, labels=label)
    #             # out.save('img.jpg')
    #             # accelerator.log({f"step: {global_step}": wandb.Image("img.jpg")})
    #         opt_unwrapped = accelerator.unwrap_model(optimizer)
    #         full_state_dict = {'model_ema':ema_model.state_dict(),
    #                         'opt_state':opt_unwrapped.state_dict(),
    #                         'global_step':global_step
    #                         }
    #         filepath =  f"{save_dir}/{config.model_name}_ckpt_{i}.bin"
    #         accelerator.save(full_state_dict, filepath )
    #         wandb.save(filepath)
    #     # else:
    #     #     accelerator.print(x_noisy.size(), noise_level.size(), label.size(), x.size() )
    #     #     accelerator.print("xnoisy", x_noisy.dtype, "noiselevel", noise_level.dtype, "label",label.dtype )
    # accelerator.end_training()



import s3fs
# def main(config: ModelConfig, dataconfig: DataConfig):
if __name__ == '__main__':

    # print(os.environ)
    # os.environ["WANDB_API_KEY"]= 'b710005fcef0b6bb3ae5a33ce530af5b93a93844'
    # os.environ["WANDB_ENTITY"]= 'lcerkovnik'
    # os.environ["WANDB_PROJECT"]=  'rakt-test'

# def main(config: ModelConfig):
    config = ModelConfig( run_id="zl707jdn",
                            model_name ="dev123_ckpt_353.bin_ckpt_681.bin" ,
                                        #  dev123_ckpt_353.bin_ckpt_681.bin  
                            # model_name ="dev123_ckpt_353-bin_ckpt_681.bin" ,

                            # model_name ="dev123_ckpt_353/bin_ckpt_545.bin" ,
                            # model_name ="bin_ckpt_545.bin" ,

                            from_scratch=False, 
                            n_epoch= 4000,
                            data_dir="/home/ray/train_data/local2", 
                            model_dir="/home/ray/models"
                            )
    # """main train loop to be used with accelerate"""

    # config = ModelConfig(n_epoch=300)
    # config["args"]= args
    ray.init("auto")
    os.umask(0) 

    accelerator = Accelerator(mixed_precision="fp16", log_with="wandb")

    accelerator.print("Loading Data:")
    # latent_train_data = torch.tensor(np.load(dataconfig.latent_path), dtype=torch.float32)
    # train_label_embeddings = torch.tensor(np.load(dataconfig.text_emb_path), dtype=torch.float32)
    # emb_val = torch.tensor(np.load(dataconfig.val_path), dtype=torch.float32)
    # dataset = TensorDataset(latent_train_data, train_label_embeddings)
    # train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    s3 = s3fs.S3FileSystem(#anon=True, 
            #                        use_listings_cache=False,
                                key='****************',
            #                        key='*****************',
                                secret='****************',
            #                        secret='*****************',
                                endpoint_url='ENTER_URL_HERE', version_aware=True)
    
    import pyarrow.fs
    custom_fs = pyarrow.fs.PyFileSystem(pyarrow.fs.FSSpecHandler(s3))
    run_config = RunConfig(storage_path="akash-thumper-v1-checkpoints", storage_filesystem=custom_fs, 
                                #    sync_config=ray.train.SyncConfig(sync_artifacts=True),
                           )
    if not os.path.exists(config.data_dir):
        os.makedirs(config.data_dir, exist_ok=True)

    # for a in range(10, 30):
    #     if not os.path.exists(f"/home/ray/train_data/wds_at1--{a}.tar"):
    #         s3.get(f"s3://akash-thumper-v1-wds-256/wds_at1--{a}.tar", "/home/ray/train_data/local")    

    for a in range(0, 2210):
        if not os.path.exists(f"{config.data_dir}/wds_at1--{a}.tar"):
            s3.get(f"s3://akash-thumper-v3-wds-256/wds_at1--{a}.tar", config.data_dir)    
    
    import torch, pickle
   
    def parse_filename(row):
        # print(row.keys())
        # row["vae_features.pyd"] = pickle.loads(row["vae_features.pyd"]).squeeze(0)
        # row["attention_mask.pyd"] =  pickle.loads(row["attention_mask.pyd"]).squeeze(0)
        row2={}
        # row2["image_latents.npy"] =  row["image_latents.npy"].squeeze()
        # row2[ 'text_encodings.npy'] =  row[ 'text_encodings.npy'].squeeze()
        row2["image_latents.pyd"] =   pickle.loads(row["image_latents.pyd"]).squeeze()
        row2['text_encodings.pyd'] =   pickle.loads(row['text_encodings.pyd']).squeeze()
         
                                    
        del row["__key__"]

        return row2
    
    train_dataset = (
        ray.data.read_webdataset( config.data_dir,
                                #  "s3://akash-thumper-v1-wds-256/wds_at1--0.tar?scheme=http&endpoint_override=ENTER_URL_HERE" ,
                                 verbose_open= False, #parallelism = 64, # suffixes=[".tar"]
            # "s3://akash-thumper-v1-wds4?scheme=http&endpoint_override=ENTER_URL_HERE" ,
        # ray.data.read_webdataset("s3://akash-thumper-v1-wds-test?scheme=http&endpoint_override=ENTER_URL_HERE" ,

                                )
        .map(parse_filename) 
    )

    train_dataset.random_shuffle()

    # config["dataset_len"]= 18  #400000 #int(dataset_len)
    torch.cuda.empty_cache()
    import pyarrow.fs
    custom_fs = pyarrow.fs.PyFileSystem(pyarrow.fs.FSSpecHandler(s3))
    run_config = RunConfig(storage_path="akash-thumper-v1-checkpoints", storage_filesystem=custom_fs, 
                                #    sync_config=ray.train.SyncConfig(sync_artifacts=True),
                           )
    trainer = TorchTrainer(
        train_func,
        datasets={"train": train_dataset},
        train_loop_config=config,
        run_config= run_config,
    
        # datasets=ray_datasets,
        # dataset_config=DataConfig(datasets_to_split=["train", "validation"]),
        scaling_config=ScalingConfig(num_workers=15, use_gpu=True, resources_per_worker={"GPU": 1, "CPU":6 }
                                     )
    )

    result = trainer.fit()

# main(ModelConfig() )