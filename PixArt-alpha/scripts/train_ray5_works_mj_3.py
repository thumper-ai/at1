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
from accelerate import DistributedDataParallelKwargs



def rescale_zero_terminal_snr(betas):
    """
    Rescales betas to have zero terminal SNR Based on https://arxiv.org/pdf/2305.08891.pdf (Algorithm 1)


    Args:
        betas (`torch.FloatTensor`):
            the betas that the scheduler is being initialized with.

    Returns:
        `torch.FloatTensor`: rescaled betas with zero terminal SNR
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas


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
                 seeds=None, 
                 eval= False):
        """Generate images via reverse diffusion."""
        print('labelsize', labels.size())
        noise_levels = (1 - torch.pow(torch.arange(0, 1, 1 / n_iter), exponent)).tolist()
        new_img = self.initialize_image(seeds, num_imgs, img_size, seed)

        labels = torch.cat([labels, torch.zeros_like(labels)])
        if eval:
            self.model.eval()

        for i in range(len(noise_levels) - 1):
            curr_noise, next_noise = noise_levels[i], noise_levels[i + 1]
            noises = torch.full((2*num_imgs, 1), curr_noise)

            x0_pred = self.model(torch.cat([new_img, new_img]),
                                 noises.to(self.device, self.model_dtype),
                                 labels.to(self.device, self.model_dtype))

            x0_pred = self.apply_classifier_free_guidance(x0_pred, num_imgs, class_guidance)

            new_img = ((curr_noise - next_noise) * x0_pred + next_noise * new_img) / curr_noise

        x0_pred[:, 3, :, :] += sharp_f
        x0_pred[:, 0, :, :] += bright_f
        if eval:
            x0_pred_img = self.vae.decode((x0_pred*scale_factor).half())[0].cpu()
        else:
            x0_pred_img = self.vae.decode((x0_pred*scale_factor).half())[0]

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

    def forward(self, x, y, add_features=False):
        if add_features:
            features=[]
            n1 = self.norm1(self.self_attention(x) + x)
            features.append(n1.reshape(n1.size()[0], -1))
            n2 = self.norm2(self.cross_attention(n1, y) + n1)
            # features.append(n2.reshape(n2.size()[0], -1))
            n3 = self.norm3(self.mlp(n2) + n2)
            # features.append(n3.reshape(n3.size()[0], -1))
            features= torch.concat(features, 1 )
            return n3, features
        else:
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


    def forward(self, x, cond, add_features= False):
        features=[]
        x = self.patchify_and_embed(x)
        # if add_features:
        #     features.append(x.reshape(x.size()[0], -1 ))
        pos_enc = self.precomputed_pos_enc[:x.size(1)].expand(x.size(0), -1)
        x = x+self.pos_embed(pos_enc)
        if add_features:
            features.append(x.reshape(x.size()[0], -1 ))
        for block in self.decoder_blocks:
            if add_features:
                x, feature= block(x, cond, add_features)
                # features.append(feature.reshape(feature.size()[0], -1 ))
            else:
                x = block(x, cond, add_features)
        if add_features:
            features = torch.cat(features, 1)
            return self.out_proj(x), features

        else:
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

    def forward(self, x, noise_level, label, add_features=False):

        noise_level = self.fourier_feats(noise_level).unsqueeze(1)

        label = self.label_proj(label).unsqueeze(1)

        noise_label_emb = torch.cat([noise_level, label], dim=1) #bs, 2, d
        noise_label_emb = self.norm(noise_label_emb)
        if add_features:
            x, features = self.denoiser_trans_block(x,noise_label_emb, add_features)
            return x , features

        else:
            x = self.denoiser_trans_block(x, noise_label_emb, add_features)
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
                                        eval=True
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
    model_weights_ref:str=""

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
    os.environ["TORCH_DISTRIBUTED_DEBUG"]=  'DETAIL'
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    os.environ["WANDB_SERVICE_WAIT"] = "300"

    # if not os.path.exists("/home/ray/thumperai/train_data/wds_at1--0.tar"):
    #     s3.get("s3://akash-thumper-v1-wds-256/wds_at1--0.tar", "/home/ray/thumperai/train_data/")

    if not os.path.exists(config.data_dir):
        os.makedirs(config.data_dir, exist_ok=True)

    # for a in range(0, 2210):
    #     if not os.path.exists(f"{config.data_dir}/wds_at1--{a}.tar"):
    #         s3.get(f"s3://akash-thumper-v3-wds-256/wds_at1--{a}.tar", config.data_dir)    

    accelerator = Accelerator(mixed_precision="fp16", log_with="wandb", kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])

    accelerator.print("Loading Data:")


    # if not os.path.exists(f"/home/ray/train_data/mj_latents.npy"):
    # #     s3.get(f"akash-thumper-v1-mj-256/mj_latents.npy", "/home/ray/train_data/") 
    # #     s3.get(f"akash-thumper-v1-mj-256/mj_text_emb.npy",  "/home/ray/train_data/") 
    #     from huggingface_hub import hf_hub_download
    #     hf_hub_download(repo_id="apapiu/small_ldt", filename="mj_latents.npy", local_dir="/home/ray/train_data/")
    #     hf_hub_download(repo_id="apapiu/small_ldt", filename="mj_text_emb.npy", local_dir ="/home/ray/train_data/")

    # latent_train_data = torch.tensor(np.load("/home/ray/train_data/small_ldt/image_latents256.npy"), dtype=torch.float32)
    # train_label_embeddings = torch.tensor(np.load("/home/ray/train_data/small_ldt/orig_text_encodings256.npy"), dtype=torch.float32)
    from huggingface_hub import hf_hub_download

    if not os.path.exists(f"/home/ray/train_data/mj_text_emb.npy"):
        hf_hub_download(repo_id="apapiu/small_ldt", filename="mj_text_emb.npy", local_dir="/home/ray/train_data/")
    if not os.path.exists(f"/home/ray/train_data/mj_latents.npy"):
        hf_hub_download(repo_id="apapiu/small_ldt", filename="mj_latents.npy", local_dir="/home/ray/train_data/")
    if not os.path.exists(f"/home/ray/train_data/orig_text_encodings256.npy"):
        hf_hub_download(repo_id="apapiu/small_ldt", filename="orig_text_encodings256.npy", local_dir="/home/ray/train_data/")
    if not os.path.exists(f"/home/ray/train_data/image_latents256.npy"):
        hf_hub_download(repo_id="apapiu/small_ldt", filename="image_latents256.npy", local_dir="/home/ray/train_data/")

    # # latent_train_data = torch.tensor(np.load("/home/ray/train_data/small_ldt/image_latents256.npy"), dtype=torch.float32)
    # # train_label_embeddings = torch.tensor(np.load("/home/ray/train_data/small_ldt/orig_text_encodings256.npy"), dtype=torch.float32)
    latent_train_data = torch.tensor(np.load("/home/ray/train_data/mj_latents.npy"), dtype=torch.float32)
    train_label_embeddings = torch.tensor(np.load("/home/ray/train_data/mj_text_emb.npy"), dtype=torch.float32)
    latent_train_data = torch.concat([latent_train_data, torch.tensor(np.load("/home/ray/train_data/image_latents256.npy"), dtype=torch.float32)], axis =0)
    train_label_embeddings = torch.concat([train_label_embeddings, torch.tensor(np.load("/home/ray/train_data/orig_text_encodings256.npy"), dtype=torch.float32),], axis=0)

    # emb_val = torch.tensor(np.load(dataconfig.val_path), dtype=torch.float32)
    dataset = TensorDataset(latent_train_data, train_label_embeddings)

    # train_data_shard = ray.train.get_dataset_shard("train")

    # train_loader = train_data_shard.iter_torch_batches(
    #     prefetch_batches =  1, 
    #     device = accelerator.device, drop_last=True,
    #     # collate_fn= collate,
    #     batch_size=128# dtypes={"image_latents.pyd": torch.float16, "text_encodings.pyd": torch.float16} #dtypes={"vae": torch.float32, "txt_features": torch.float32, "attention_mask": torch.int32}
    # )
    # latent_train_data = torch.tensor(np.load("/home/ray/thumperai/small_ldt/image_latents256.npy"), dtype=torch.float32)
    # train_label_embeddings = torch.tensor(np.load("/home/ray/thumperai/small_ldt/orig_text_encodings256.npy"), dtype=torch.float32)
    # latent_train_data= []
    # train_label_embeddings= []
    # for a in range(15):
    #     image_latent_path = f"/home/ray/thumperai/train_data/image_latents_{a}.npy"
    #     text_emb_path = f"/home/ray/thumperai/train_data/text_encodings_{a}.npy"
    #     if not os.path.exists(image_latent_path):
    #         s3.get(f"akash-thumper-v1-face-256/image_latents_{a}.npy", "/home/ray/thumperai/train_data")
    #     if not os.path.exists(text_emb_path):
    #         s3.get(f"akash-thumper-v1-face-256/text_encodings_{a}.npy", "/home/ray/thumperai/train_data")
    #     try:
    #         lt_img = np.load(image_latent_path).squeeze()
    #         text_emb = np.load(text_emb_path).squeeze()
    #         print(a, "size ", lt_img.shape, text_emb.shape )
            
    #         latent_train_data.append(torch.tensor(lt_img, dtype=torch.float32))
    #         train_label_embeddings.append(torch.tensor(text_emb, dtype=torch.float32))
    #     except Exception as e:
    #         print(e)
            
    # latent_train_data= torch.cat(latent_train_data).squeeze()
    # train_label_embeddings= torch.cat(train_label_embeddings).squeeze()

    # image_latent_path = f"/home/ray/thumperai/train_data/image_latentsf_0.npy"
    # text_emb_path = f"/home/ray/thumperai/train_data/text_encodingsf_0.npy"
    # if not os.path.exists(image_latent_path):
    #     s3.get(f"akash-thumper-v1-face-256/image_latentsf_0.npy", "/home/ray/thumperai/train_data")
    # if not os.path.exists(text_emb_path):
    #     s3.get(f"akash-thumper-v1-face-256/text_encodingsf_0.npy", "/home/ray/thumperai/train_data")

    
    # latent_train_data = torch.tensor(np.load("/home/ray/thumperai/train_data/image_latentsf_0.npy"), dtype=torch.float32)
    # train_label_embeddings = torch.tensor(np.load("/home/ray/thumperai/train_data/text_encodingsf_0.npy"), dtype=torch.float32)
    # assert latent_train_data.size()[0] ==train_label_embeddings.size()[0], "error bad data shape"

    # latent_train_data = torch.tensor(np.load("/home/ray/thumperai/small_ldt/mj_latents.npy"), dtype=torch.float32)
    # train_label_embeddings = torch.tensor(np.load("/home/ray/thumperai/small_ldt/mj_text_emb.npy"), dtype=torch.float32)

    # emb_val = torch.tensor(np.load(dataconfig.val_path), dtype=torch.float32)
    # dataset = TensorDataset(latent_train_data, train_label_embeddings)
    train_loader = DataLoader(dataset, batch_size=80, shuffle=True)

    # def collate(input):
    #     X = torch.cat([ainput[0] for ainput in input["item"]])
    #     y = torch.cat([ainput[1] for ainput in input["item"]])
    #     return X, y # {"X":X, "Y":y}
    # # return {"X":X, "Y":y}
    # train_loader = train_data_shard.iter_torch_batches(
    #         prefetch_batches =  1, 
    #         # device = accelerator.device, drop_last=True,
    #         collate_fn= collate,
    #         batch_size=64# dtypes={"image_latents.pyd": torch.float16, "text_encodings.pyd": torch.float16} #dtypes={"vae": torch.float32, "txt_features": torch.float32, "attention_mask": torch.int32}
    #     )
    train_dataset_len = int(round(len(train_loader)// ray.train.get_context().get_world_size()))
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

        # if not os.path.exists(f"/home/ray/models/faceiter0_ckpt_22.bin"):

        # s3 = s3fs.S3FileSystem(#anon=True, 
        #         #                        use_listings_cache=False,
        #                             key='****************',
        #         #                        key='*****************',
        #                             secret='****************',
        #         #                        secret='*****************',
        #                             endpoint_url='ENTER_URL_HERE', version_aware=True)
        # # s3.get(f"s3://akash-thumper-v1-checkpoints/faceiter0_ckpt_22.bin", "/home/ray/models/")    
        # s3.get(f"s3://akash-thumper-v1-checkpoints/faceiter1_ckpt_1.bin", "/home/ray/models/")    

        # wandb.restore("faceiter0_ckpt_22.bin",  run_path=  f"lcerkovnik/at1256/runs/{config.run_id}", 
        #             #   run_path="/home/ray/models",
        #                 root="/home/ray/models",
        #             #   f"lcerkovnik/at1256/runs/{config.run_id}",
        #               replace=False)
        # wandb.restore(config.model_name, run_path=f"lcerkovnik/at1256/runs/{config.run_id}",
        #               replace=True)
        # full_state_dict = torch.load(config.model_name, map_location=accelerator.device)
        # full_state_dict = torch.load("/home/ray/models/faceiter0_ckpt_22.bin")
        # full_state_dict = torch.load("/home/ray/models/faceiter1_ckpt_1.bin")
        full_state_dict = ray.get(config.model_weights_ref)
        model.load_state_dict(full_state_dict['model_ema'])
        optimizer.load_state_dict(full_state_dict['opt_state'])
        global_step = full_state_dict['global_step']
    else:
        global_step = 0

    # if accelerator.is_local_main_process:
    ema_model = copy.deepcopy(model).to(accelerator.device)
    perceptual_loss_model = copy.deepcopy(model).to(accelerator.device)

    diffuser = DiffusionGenerator(ema_model, vae,  accelerator.device, torch.float32)
    perceptual_diffuser = DiffusionGenerator(perceptual_loss_model, vae,  accelerator.device, torch.float32)

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
        for x,y in train_loader:
            # x = batch["image_latents.pyd"].squeeze(0)[0:128,]#.to(accelerator.device)
            # y=  batch["text_encodings.pyd"].squeeze(0)[0:128,]#.to(accelerator.device)
            # x = batch["image_latents.pyd"].squeeze(0)[0:128,]#.to(accelerator.device)
            # y=  batch["text_encodings.pyd"].squeeze(0)[0:128,]#.to(accelerator.device)
            x = x/config.scaling_factor

            noise_level = torch.tensor(np.random.beta(config.beta_a, config.beta_b, len(x)),
                                        device=accelerator.device)
           
            noise_level = rescale_zero_terminal_snr(noise_level)
            signal_level = 1 - noise_level # alpha = 1- beta


            noise = torch.randn_like(x)

            x_noisy = noise_level.view(-1,1,1,1)*noise + signal_level.view(-1,1,1,1)*x  # beta*noise + alpha*x

            #other one was vt =  alpha^.5* noise - beta^.5*x = 
            # alpha_prod_t = signal_level
            # beta_prod_t = noise
            # pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            # pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample

            x_noisy = x_noisy.float()
            noise_level = noise_level.float()
            label = y

            prob = 0.15
            mask = torch.rand(y.size(0), device=accelerator.device) < prob
            label[mask] = 0 # OR replacement_vector

            if global_step % 4000 == 0:
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
                    full_state_dict = {'model_ema': ema_model.state_dict(),
                                    'opt_state':opt_unwrapped.state_dict(),
                                    'global_step':global_step
                                    }
                    filepath =  f"{save_dir}/{'faceiter3c'}_ckpt_{i}.bin"

                    accelerator.save(full_state_dict, filepath)
                    os.system(f"b2 cp {filepath} b2://akash-thumper-v1-checkpoints")
                    wandb.save(filepath)
                # dist.barrier()

            model.train()

            with accelerator.accumulate():
                ###train loop:
                optimizer.zero_grad()
                if global_step % 2000 ==0:

                    pred, pred_features = model(x_noisy, noise_level.view(-1,1), label,add_features = True)
                    class_guidance=4.5
                    seed=10
                    print("labels", label.size())
                    x0_pred_img, x0_pred = diffuser.generate(labels= label.squeeze(), #torch.repeat_interleave(labels, 8, dim=0),
                                                        num_imgs=label.size()[0],
                                                        class_guidance=class_guidance,
                                                        seed=seed,
                                                        n_iter=40,
                                                        exponent=1,
                                                        sharp_f=0.1,
                                                        )

                    pred, perceptual_features = perceptual_loss_model(x0_pred, noise_level.view(-1,1), label, add_features = True)

                    loss1 = loss_fn(pred_features, perceptual_features.float())
                    loss2 = loss_fn(pred, x.float())
                    loss= (0.75*loss1+ 0.25*loss2)

                    accelerator.log({ "train_loss":loss.item() , "perceptual_loss":loss1.item(), "mse": loss2.item() ,  "step": global_step})
                    ray.train.report({"train_loss":loss.item() ,"perceptual_loss":loss1.item(), "mse": loss2.item(),  "step": global_step})
                    accelerator.backward(loss1)
                else:
                    pred = model(x_noisy, noise_level.view(-1,1), label)
                    loss = loss_fn(pred, x.float())
                    accelerator.log({"mse":loss.item(), "step": global_step})
                    ray.train.report({"mse":loss.item(), "train_loss":loss.item() , "step": global_step})
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
    #                 save_dir = f"/home/ray/thumperai/ray_results/checkpoint_{i}/"
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
    #         save_dir = f"/home/ray/thumperai/ray_results/checkpoint_{i}/"
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

    print(os.environ)
    os.environ["WANDB_API_KEY"]= 'b710005fcef0b6bb3ae5a33ce530af5b93a93844'
    os.environ["WANDB_ENTITY"]= 'lcerkovnik'
    os.environ["WANDB_PROJECT"]=  'rakt-test'
    os.environ["TORCH_DISTRIBUTED_DEBUG"]=  'DETAIL'
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    os.environ["WANDB_SERVICE_WAIT"] = "300"
    ray.init("auto")

    # run_id = "w8sp8iff",#"v1riw11r" #"hq4aoiah" # "acdfeumf"# "bw6xd79g"
    # model_name = "faceiter3c_ckpt_83.bin", #"faceiter1_ckpt_290.bin" # "faceiter1_ckpt_60.bin" #"faceiter3c_ckpt_481.bin" #"faceiter3c_ckpt_11.bin"
    run_id = "v1riw11r" #"hq4aoiah" # "acdfeumf"# "bw6xd79g"
    model_name = "faceiter1_ckpt_509.bin" # "faceiter1_ckpt_60.bin" #"faceiter3c_ckpt_481.bin" #"faceiter3c_ckpt_11.bin"

    # run_id = "v1riw11r",#"v1riw11r" #"hq4aoiah" # "acdfeumf"# "bw6xd79g"
    # model_name = "faceiter3c_ckpt_83.bin", #"fa
# w8sp8iff/files/face/iter3c_ckpt_83.bin
# https://wandb.ai/lcerkovnik/at1256/runs/hq4aoiah/files/faceiter1_ckpt_60.bin
    wandb.restore(model_name, run_path=f"lcerkovnik/at1256/runs/{run_id}",
                replace=True)
    full_state_dict = torch.load(model_name)
    weights_ref = ray.put(full_state_dict)
# def main(config: ModelConfig):
    config = ModelConfig(  #run_id="dq695cr3", model_name="faceiter0_ckpt_42.bin",
                          run_id=run_id, model_name= model_name,
                        #  https://wandb.ai/lcerkovnik/at1256/runs/gargjnrx/files/faceiter0_ckpt_12.bin
    # dq695cr3/files/faceiter0_ckpt_20.bin
                        #  "dev123_ckpt_996.bin_ckpt_172.bin_ckpt_761.bin_ckpt_568.bin_ckpt_2.bin_ckpt_57.bin_ckpt_79.bin_ckpt_109.bin_ckpt_23.bin_ckpt_18.bin_ckpt_4.bin_ckpt_28.bin_ckpt_14.bin_ckpt_19.bin_ckpt_33.bin",
    # "dev123_ckpt_996.bin_ckpt_172.bin_ckpt_761.bin_ckpt_568.bin_ckpt_2.bin_ckpt_57.bin_ckpt_79.bin_ckpt_109.bin_ckpt_23.bin_ckpt_18.bin_ckpt_4.bin_ckpt_28.bin_ckpt_14.bin_ckpt_19.bin",
    # "dev123_ckpt_996.bin_ckpt_172.bin_ckpt_761.bin_ckpt_568.bin_ckpt_2.bin_ckpt_57.bin_ckpt_79.bin_ckpt_109.bin_ckpt_23.bin_ckpt_18.bin_ckpt_4.bin_ckpt_28.bin_ckpt_14.bin",
    # "dev123_ckpt_996.bin_ckpt_172.bin_ckpt_761.bin_ckpt_568.bin_ckpt_2.bin_ckpt_57.bin_ckpt_79.bin_ckpt_109.bin_ckpt_23.bin_ckpt_18.bin_ckpt_4.bin",
    # dev123_ckpt_996.bin_ckpt_172.bin_ckpt_761.bin_ckpt_568.bin_ckpt_2.bin_ckpt_57.bin_ckpt_79.bin_ckpt_109.bin_ckpt_23.bin_ckpt_18.bin",
    #  "dev123_ckpt_996.bin_ckpt_172.bin_ckpt_761.bin_ckpt_568.bin_ckpt_2.bin",#"dev123_ckpt_996.bin",
                        #   model_name ="None_ckpt_201.bin_ckpt_401.bin" ,
                        # https://wandb.ai/lcerkovnik/at1256/runs/7n3jtj7y/files/dev123_ckpt_996.bin_ckpt_172.bin_ckpt_761.bin_ckpt_568.bin_ckpt_2.bin_ckpt_57.bin_ckpt_79.bin_ckpt_109.bin_ckpt_23.bin_ckpt_18.bin_ckpt_4.bin
                            from_scratch=False, 
                            n_epoch= 2000,
                            data_dir="/home/ray/train_data/local2", 
                            model_dir="/home/ray/models",
                            model_weights_ref=weights_ref
                            )
    # """main train loop to be used with accelerate"""

    # config = ModelConfig(n_epoch=300)
    # config["args"]= args
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

    # if not os.path.exists(f"/home/ray/models/faceiter0_ckpt_22.bin"):
    #     s3.get(f"s3://akash-thumper-v1-checkpoints/faceiter0_ckpt_22.bin", "/home/ray/models/")    
    
    # if not os.path.exists(f"/home/ray/models/faceiter0_ckpt_20.bin"):
    #     s3.get(f"s3://akash-thumper-v1-checkpoints/faceiter0_ckpt_20.bin", "/home/ray/models/")    
    # for a in range(10, 30):
    #     if not os.path.exists(f"/home/ray/thumperai/train_data/wds_at1--{a}.tar"):
    #         s3.get(f"s3://akash-thumper-v1-wds-256/wds_at1--{a}.tar", "/home/ray/thumperai/train_data/local")    

    # for a in range(0, 2210):
    #     if not os.path.exists(f"{config.data_dir}/wds_at1--{a}.tar"):
    #         s3.get(f"s3://akash-thumper-v3-wds-256/wds_at1--{a}.tar", config.data_dir)    
    
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
    
    def collate(input):
        X = torch.cat([ainput[0] for ainput in input["item"]])
        y = torch.cat([ainput[1] for ainput in input["item"]])
        return X, y # {"X":X, "Y":y}
    
    # if not os.path.exists(f"/home/ray/train_data/small_ldt/mj_latents.npy"):
    #     os.system("cd /home/ray/train_data && git clone https://huggingface.co/apapiu/small_ldt ")
    #     # s3.get(f"/home/ray/thumperai/small_ldt/image_latents256.npy", config.data_dir)    
    # os.system("cd /home/ray/train_data && git pull https://huggingface.co/apapiu/small_ldt ")

#     if not os.path.exists(f"/home/ray/train_data/orig_text_encodings256.npy"):
#     #     s3.get(f"akash-thumper-v1-mj-256/mj_latents.npy", "/home/ray/train_data/") 
# #     s3.get(f"akash-thumper-v1-mj-256/mj_text_emb.npy",  "/home/ray/train_data/") 
#         from huggingface_hub import hf_hub_download
#         hf_hub_download(repo_id="apapiu/small_ldt", filename="mj_latents.npy", local_dir="/home/ray/train_data/")
#         hf_hub_download(repo_id="apapiu/small_ldt", filename="mj_text_emb.npy", local_dir ="/home/ray/train_data/")
#         hf_hub_download(repo_id="apapiu/small_ldt", filename="image_latents256.npy", local_dir="/home/ray/train_data/")
#         hf_hub_download(repo_id="apapiu/small_ldt", filename="orig_text_encodings256.npy", local_dir ="/home/ray/train_data/")

#     # # latent_train_data = torch.tensor(np.load("/home/ray/train_data/small_ldt/image_latents256.npy"), dtype=torch.float32)
#     # # train_label_embeddings = torch.tensor(np.load("/home/ray/train_data/small_ldt/orig_text_encodings256.npy"), dtype=torch.float32)
#     latent_train_data = torch.tensor(np.load("/home/ray/train_data/mj_latents.npy"), dtype=torch.float32)
#     train_label_embeddings = torch.tensor(np.load("/home/ray/train_data/mj_text_emb.npy"), dtype=torch.float32)
#     latent_train_data = torch.concat([latent_train_data, torch.tensor(np.load("/home/ray/train_data/image_latents256.npy"), dtype=torch.float32)], axis =0)
#     train_label_embeddings = torch.concat([train_label_embeddings, torch.tensor(np.load("/home/ray/train_data/orig_text_encodings256.npy"), dtype=torch.float32),], axis=0)

    # # emb_val = torch.tensor(np.load(dataconfig.val_path), dtype=torch.float32)
    # dataset = TensorDataset(latent_train_data, train_label_embeddings)
    # # train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    # train_dataset =  ray.data.from_torch(dataset)
    # train_dataset = (
    #     ray.data.read_webdataset( config.data_dir,
    #                             #  "s3://akash-thumper-v1-wds-256/wds_at1--0.tar?scheme=http&endpoint_override=ENTER_URL_HERE" ,
    #                              verbose_open= False, #parallelism = 64, # suffixes=[".tar"]
    #         # "s3://akash-thumper-v1-wds4?scheme=http&endpoint_override=ENTER_URL_HERE" ,
    #     # ray.data.read_webdataset("s3://akash-thumper-v1-wds-test?scheme=http&endpoint_override=ENTER_URL_HERE" ,
    #                             )
    #     .map(parse_filename) 
    # )

    # train_dataset.random_shuffle()

    # config["dataset_len"]= 18  #400000 #int(dataset_len)


    torch.cuda.empty_cache()
    import pyarrow.fs
    # # custom_fs = pyarrow.fs.PyFileSystem(pyarrow.fs.FSSpecHandler(s3))
    # run_config = RunConfig(storage_path="akash-thumper-v1-checkpoints", storage_filesystem=custom_fs, 
    #                             #    sync_config=ray.train.SyncConfig(sync_artifacts=True),
    #                        )
    trainer = TorchTrainer(
        train_func,
        # datasets={"train": train_dataset},
        train_loop_config=config,
        # run_config= run_config,
    
        # datasets=ray_datasets,
        # dataset_config=DataConfig(datasets_to_split=["train", "validation"]),
        scaling_config=ScalingConfig(num_workers=15, use_gpu=True, resources_per_worker={"GPU": 1, "CPU":6 }
                                     )
    )

    result = trainer.fit()

# main(ModelConfig() )

# https://huggingface.co/apapiu/small_ldt/tree/main