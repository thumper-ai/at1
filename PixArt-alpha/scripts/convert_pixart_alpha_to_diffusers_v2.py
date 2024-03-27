import argparse
import os

import torch
from transformers import T5EncoderModel, T5Tokenizer

from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, PixArtAlphaPipeline, Transformer2DModel
from collections import OrderedDict


ckpt_id = "PixArt-alpha/PixArt-alpha"
# https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/scripts/inference.py#L125
interpolation_scale = {512: 1, 1024: 2, 256:1}


def main(args):
    checkpoint_path = args.orig_ckpt_path
    if "akash-thumper-v1-checkpoints" in args.orig_ckpt_path:
        checkpoint_path= f"/mnt/sabrent/{os.path.basename(args.orig_ckpt_path)}"
        import s3fs
        s3 = s3fs.S3FileSystem(#anon=True, 
            #                        use_listings_cache=False,
                                key='*************',
            #                        key='*************',
                                secret='*************',
            #                        secret='*************',
                                endpoint_url='ENTER_URL', version_aware=True)
        if not os.path.exists(checkpoint_path):
            # s3.get('akash-thumper-v1-checkpoints/TorchTrainer_2023-12-27_13-10-55/TorchTrainer_6cbd8_00000_0_2023-12-27_13-10-55/checkpoint_000004/epoch_10_step_40.pth',
            s3.get(args.orig_ckpt_path, checkpoint_path)

    all_state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu')
                                )

    state_dict = all_state_dict.pop("state_dict")
    converted_state_dict = {}
    print(state_dict.keys())

    # Patch embeddings.
    converted_state_dict["pos_embed.proj.weight"] = state_dict.pop("x_embedder.proj.weight")
    converted_state_dict["pos_embed.proj.bias"] = state_dict.pop("x_embedder.proj.bias")

    # Caption projection.
    converted_state_dict["caption_projection.y_embedding"] = state_dict.pop("y_embedder.y_embedding")
    converted_state_dict["caption_projection.linear_1.weight"] = state_dict.pop("y_embedder.y_proj.fc1.weight")
    converted_state_dict["caption_projection.linear_1.bias"] = state_dict.pop("y_embedder.y_proj.fc1.bias")
    converted_state_dict["caption_projection.linear_2.weight"] = state_dict.pop("y_embedder.y_proj.fc2.weight")
    converted_state_dict["caption_projection.linear_2.bias"] = state_dict.pop("y_embedder.y_proj.fc2.bias")

    # AdaLN-single LN
    converted_state_dict["adaln_single.emb.timestep_embedder.linear_1.weight"] = state_dict.pop(
        "t_embedder.mlp.0.weight"
    )
    converted_state_dict["adaln_single.emb.timestep_embedder.linear_1.bias"] = state_dict.pop("t_embedder.mlp.0.bias")
    converted_state_dict["adaln_single.emb.timestep_embedder.linear_2.weight"] = state_dict.pop(
        "t_embedder.mlp.2.weight"
    )
    converted_state_dict["adaln_single.emb.timestep_embedder.linear_2.bias"] = state_dict.pop("t_embedder.mlp.2.bias")

    if args.image_size == 1024 and args.multi_scale_train:
        # Resolution.
        converted_state_dict["adaln_single.emb.resolution_embedder.linear_1.weight"] = state_dict.pop(
            "csize_embedder.mlp.0.weight"
        )
        converted_state_dict["adaln_single.emb.resolution_embedder.linear_1.bias"] = state_dict.pop(
            "csize_embedder.mlp.0.bias"
        )
        converted_state_dict["adaln_single.emb.resolution_embedder.linear_2.weight"] = state_dict.pop(
            "csize_embedder.mlp.2.weight"
        )
        converted_state_dict["adaln_single.emb.resolution_embedder.linear_2.bias"] = state_dict.pop(
            "csize_embedder.mlp.2.bias"
        )
        # Aspect ratio.
        converted_state_dict["adaln_single.emb.aspect_ratio_embedder.linear_1.weight"] = state_dict.pop(
            "ar_embedder.mlp.0.weight"
        )
        converted_state_dict["adaln_single.emb.aspect_ratio_embedder.linear_1.bias"] = state_dict.pop(
            "ar_embedder.mlp.0.bias"
        )
        converted_state_dict["adaln_single.emb.aspect_ratio_embedder.linear_2.weight"] = state_dict.pop(
            "ar_embedder.mlp.2.weight"
        )
        converted_state_dict["adaln_single.emb.aspect_ratio_embedder.linear_2.bias"] = state_dict.pop(
            "ar_embedder.mlp.2.bias"
        )
    # Shared norm.
    converted_state_dict["adaln_single.linear.weight"] = state_dict.pop("t_block.1.weight")
    converted_state_dict["adaln_single.linear.bias"] = state_dict.pop("t_block.1.bias")

    for depth in range(28):
        # Transformer blocks.
        converted_state_dict[f"transformer_blocks.{depth}.scale_shift_table"] = state_dict.pop(
            f"blocks.{depth}.scale_shift_table"
        )

        # Attention is all you need 🤘

        # Self attention.
        q, k, v = torch.chunk(state_dict.pop(f"blocks.{depth}.attn.qkv.weight"), 3, dim=0)
        q_bias, k_bias, v_bias = torch.chunk(state_dict.pop(f"blocks.{depth}.attn.qkv.bias"), 3, dim=0)
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_q.weight"] = q
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_q.bias"] = q_bias
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_k.weight"] = k
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_k.bias"] = k_bias
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_v.weight"] = v
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_v.bias"] = v_bias
        # Projection.
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_out.0.weight"] = state_dict.pop(
            f"blocks.{depth}.attn.proj.weight"
        )
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_out.0.bias"] = state_dict.pop(
            f"blocks.{depth}.attn.proj.bias"
        )

        # Feed-forward.
        converted_state_dict[f"transformer_blocks.{depth}.ff.net.0.proj.weight"] = state_dict.pop(
            f"blocks.{depth}.mlp.fc1.weight"
        )
        converted_state_dict[f"transformer_blocks.{depth}.ff.net.0.proj.bias"] = state_dict.pop(
            f"blocks.{depth}.mlp.fc1.bias"
        )
        converted_state_dict[f"transformer_blocks.{depth}.ff.net.2.weight"] = state_dict.pop(
            f"blocks.{depth}.mlp.fc2.weight"
        )
        converted_state_dict[f"transformer_blocks.{depth}.ff.net.2.bias"] = state_dict.pop(
            f"blocks.{depth}.mlp.fc2.bias"
        )

        # Cross-attention.
        q = state_dict.pop(f"blocks.{depth}.cross_attn.q_linear.weight")
        q_bias = state_dict.pop(f"blocks.{depth}.cross_attn.q_linear.bias")
        k, v = torch.chunk(state_dict.pop(f"blocks.{depth}.cross_attn.kv_linear.weight"), 2, dim=0)
        k_bias, v_bias = torch.chunk(state_dict.pop(f"blocks.{depth}.cross_attn.kv_linear.bias"), 2, dim=0)

        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_q.weight"] = q
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_q.bias"] = q_bias
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_k.weight"] = k
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_k.bias"] = k_bias
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_v.weight"] = v
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_v.bias"] = v_bias

        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_out.0.weight"] = state_dict.pop(
            f"blocks.{depth}.cross_attn.proj.weight"
        )
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_out.0.bias"] = state_dict.pop(
            f"blocks.{depth}.cross_attn.proj.bias"
        )

    # Final block.
    converted_state_dict["proj_out.weight"] = state_dict.pop("final_layer.linear.weight")
    converted_state_dict["proj_out.bias"] = state_dict.pop("final_layer.linear.bias")
    converted_state_dict["scale_shift_table"] = state_dict.pop("final_layer.scale_shift_table")

    # DiT XL/2
    transformer = Transformer2DModel(
        sample_size=args.image_size // 8,
        num_layers=28,
        attention_head_dim=72,
        in_channels=4,
        out_channels=8,
        patch_size=2,
        attention_bias=True,
        num_attention_heads=16,
        cross_attention_dim=1152,
        activation_fn="gelu-approximate",
        num_embeds_ada_norm=1000,
        norm_type="ada_norm_single",
        norm_elementwise_affine=False,
        norm_eps=1e-6,
        caption_channels=4096,
    )
    transformer.load_state_dict(converted_state_dict, strict=True)

    assert transformer.pos_embed.pos_embed is not None
    state_dict.pop("pos_embed")
    assert len(state_dict) == 0, f"State dict is not empty, {state_dict.keys()}"

    num_model_params = sum(p.numel() for p in transformer.parameters())
    print(f"Total number of transformer parameters: {num_model_params}")

    # if args.only_transformer:
    #     transformer.save_pretrained(os.path.join(args.dump_path, "transformer"))
    # else:
    scheduler = DPMSolverMultistepScheduler()

    vae = AutoencoderKL.from_pretrained(ckpt_id, subfolder="sd-vae-ft-ema")

    tokenizer = T5Tokenizer.from_pretrained(ckpt_id, subfolder="t5-v1_1-xxl")
    text_encoder = T5EncoderModel.from_pretrained(ckpt_id, subfolder="t5-v1_1-xxl")

    pipeline = PixArtAlphaPipeline(
        tokenizer=tokenizer, text_encoder=text_encoder, transformer=transformer, vae=vae, scheduler=scheduler
    )

    pipeline.save_pretrained(args.dump_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # set multi_scale_train=True if using PixArtMS structure during training else set it to False
    parser.add_argument("--multi_scale_train", default=True, type=str, required=True, help="If use Multi-Scale PixArtMS structure during training.")
    parser.add_argument("--orig_ckpt_path", default=None, type=str, required=False, help="Path to the checkpoint to convert.")
    parser.add_argument(
        "--image_size",
        default=256,
        type=int,
        choices=[256, 512, 1024],
        required=False,
        help="Image size of pretrained model, either 512 or 1024.",
    )
    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output pipeline.")
    parser.add_argument("--only_transformer", default=False, type=bool, required=True)

    args = parser.parse_args()
    main(args)

    # python convert_pixart_alpha_to_diffusers.py \
    # --orig_ckpt_path "epoch_120_step_0.pth" \
    # --dump_path "/home/logan/ray_results/checkpoint_000029" \
    # --only_transformer=True  --image_size 512   --multi_scale_train=False

                                                        

# odict_keys(['module.pos_embed', 'module.x_embedder.proj.weight', 'module.x_embedder.proj.bias', 
#             'module.t_embedder.mlp.0.weight', 'module.t_embedder.mlp.0.bias', 'module.t_embedder.mlp.2.weight', 'module.t_embedder.mlp.2.bias', 'module.t_block.1.weight', 'module.t_block.1.bias', 'module.y_embedder.y_embedding', 'module.y_embedder.y_proj.fc1.weight', 'module.y_embedder.y_proj.fc1.bias', 'module.y_embedder.y_proj.fc2.weight', 'module.y_embedder.y_proj.fc2.bias', 'module.blocks.0.scale_shift_table', 'module.blocks.0.attn.qkv.weight', 'module.blocks.0.attn.qkv.bias', 'module.blocks.0.attn.proj.weight', 'module.blocks.0.attn.proj.bias', 'module.blocks.0.cross_attn.q_linear.weight', 'module.blocks.0.cross_attn.q_linear.bias', 'module.blocks.0.cross_attn.kv_linear.weight', 'module.blocks.0.cross_attn.kv_linear.bias', 'module.blocks.0.cross_attn.proj.weight', 'module.blocks.0.cross_attn.proj.bias', 'module.blocks.0.mlp.fc1.weight', 'module.blocks.0.mlp.fc1.bias', 'module.blocks.0.mlp.fc2.weight', 'module.blocks.0.mlp.fc2.bias', 'module.blocks.1.scale_shift_table', 'module.blocks.1.attn.qkv.weight', 'module.blocks.1.attn.qkv.bias', 'module.blocks.1.attn.proj.weight', 'module.blocks.1.attn.proj.bias', 'module.blocks.1.cross_attn.q_linear.weight', 'module.blocks.1.cross_attn.q_linear.bias', 'module.blocks.1.cross_attn.kv_linear.weight', 'module.blocks.1.cross_attn.kv_linear.bias', 'module.blocks.1.cross_attn.proj.weight', 'module.blocks.1.cross_attn.proj.bias', 'module.blocks.1.mlp.fc1.weight', 'module.blocks.1.mlp.fc1.bias', 'module.blocks.1.mlp.fc2.weight', 'module.blocks.1.mlp.fc2.bias', 'module.blocks.2.scale_shift_table', 'module.blocks.2.attn.qkv.weight', 'module.blocks.2.attn.qkv.bias', 'module.blocks.2.attn.proj.weight', 'module.blocks.2.attn.proj.bias', 'module.blocks.2.cross_attn.q_linear.weight', 'module.blocks.2.cross_attn.q_linear.bias', 'module.blocks.2.cross_attn.kv_linear.weight', 'module.blocks.2.cross_attn.kv_linear.bias', 'module.blocks.2.cross_attn.proj.weight', 'module.blocks.2.cross_attn.proj.bias', 'module.blocks.2.mlp.fc1.weight', 'module.blocks.2.mlp.fc1.bias', 'module.blocks.2.mlp.fc2.weight', 'module.blocks.2.mlp.fc2.bias', 'module.blocks.3.scale_shift_table', 'module.blocks.3.attn.qkv.weight', 'module.blocks.3.attn.qkv.bias', 'module.blocks.3.attn.proj.weight', 'module.blocks.3.attn.proj.bias', 'module.blocks.3.cross_attn.q_linear.weight', 'module.blocks.3.cross_attn.q_linear.bias', 'module.blocks.3.cross_attn.kv_linear.weight', 'module.blocks.3.cross_attn.kv_linear.bias', 'module.blocks.3.cross_attn.proj.weight', 'module.blocks.3.cross_attn.proj.bias', 'module.blocks.3.mlp.fc1.weight', 'module.blocks.3.mlp.fc1.bias', 'module.blocks.3.mlp.fc2.weight', 'module.blocks.3.mlp.fc2.bias', 'module.blocks.4.scale_shift_table', 'module.blocks.4.attn.qkv.weight',
#              'module.blocks.4.attn.qkv.bias', 'module.blocks.4.attn.proj.weight', 'module.blocks.4.attn.proj.bias', 'module.blocks.4.cross_attn.q_linear.weight', 'module.blocks.4.cross_attn.q_linear.bias', 'module.blocks.4.cross_attn.kv_linear.weight', 'module.blocks.4.cross_attn.kv_linear.bias', 'module.blocks.4.cross_attn.proj.weight', 'module.blocks.4.cross_attn.proj.bias', 'module.blocks.4.mlp.fc1.weight', 'module.blocks.4.mlp.fc1.bias', 'module.blocks.4.mlp.fc2.weight', 'module.blocks.4.mlp.fc2.bias', 'module.blocks.5.scale_shift_table', 'module.blocks.5.attn.qkv.weight', 'module.blocks.5.attn.qkv.bias', 'module.blocks.5.attn.proj.weight', 'module.blocks.5.attn.proj.bias', 'module.blocks.5.cross_attn.q_linear.weight', 'module.blocks.5.cross_attn.q_linear.bias', 'module.blocks.5.cross_attn.kv_linear.weight', 'module.blocks.5.cross_attn.kv_linear.bias', 'module.blocks.5.cross_attn.proj.weight', 'module.blocks.5.cross_attn.proj.bias', 'module.blocks.5.mlp.fc1.weight', 'module.blocks.5.mlp.fc1.bias', 'module.blocks.5.mlp.fc2.weight', 'module.blocks.5.mlp.fc2.bias', 'module.blocks.6.scale_shift_table', 'module.blocks.6.attn.qkv.weight', 'module.blocks.6.attn.qkv.bias', 'module.blocks.6.attn.proj.weight', 'module.blocks.6.attn.proj.bias', 'module.blocks.6.cross_attn.q_linear.weight', 'module.blocks.6.cross_attn.q_linear.bias', 'module.blocks.6.cross_attn.kv_linear.weight', 'module.blocks.6.cross_attn.kv_linear.bias', 'module.blocks.6.cross_attn.proj.weight', 'module.blocks.6.cross_attn.proj.bias', 'module.blocks.6.mlp.fc1.weight', 'module.blocks.6.mlp.fc1.bias', 'module.blocks.6.mlp.fc2.weight', 'module.blocks.6.mlp.fc2.bias', 'module.blocks.7.scale_shift_table', 'module.blocks.7.attn.qkv.weight', 'module.blocks.7.attn.qkv.bias', 'module.blocks.7.attn.proj.weight', 'module.blocks.7.attn.proj.bias', 'module.blocks.7.cross_attn.q_linear.weight', 'module.blocks.7.cross_attn.q_linear.bias', 'module.blocks.7.cross_attn.kv_linear.weight', 'module.blocks.7.cross_attn.kv_linear.bias', 'module.blocks.7.cross_attn.proj.weight', 'module.blocks.7.cross_attn.proj.bias', 'module.blocks.7.mlp.fc1.weight', 'module.blocks.7.mlp.fc1.bias', 'module.blocks.7.mlp.fc2.weight', 'module.blocks.7.mlp.fc2.bias', 'module.blocks.8.scale_shift_table', 'module.blocks.8.attn.qkv.weight', 'module.blocks.8.attn.qkv.bias', 'module.blocks.8.attn.proj.weight', 'module.blocks.8.attn.proj.bias', 'module.blocks.8.cross_attn.q_linear.weight', 'module.blocks.8.cross_attn.q_linear.bias', 'module.blocks.8.cross_attn.kv_linear.weight', 'module.blocks.8.cross_attn.kv_linear.bias', 'module.blocks.8.cross_attn.proj.weight', 'module.blocks.8.cross_attn.proj.bias', 'module.blocks.8.mlp.fc1.weight', 'module.blocks.8.mlp.fc1.bias', 'module.blocks.8.mlp.fc2.weight', 'module.blocks.8.mlp.fc2.bias', 'module.blocks.9.scale_shift_table', 'module.blocks.9.attn.qkv.weight', 'module.blocks.9.attn.qkv.bias', 'module.blocks.9.attn.proj.weight', 'module.blocks.9.attn.proj.bias', 'module.blocks.9.cross_attn.q_linear.weight', 'module.blocks.9.cross_attn.q_linear.bias', 'module.blocks.9.cross_attn.kv_linear.weight', 'module.blocks.9.cross_attn.kv_linear.bias', 'module.blocks.9.cross_attn.proj.weight', 'module.blocks.9.cross_attn.proj.bias', 'module.blocks.9.mlp.fc1.weight', 'module.blocks.9.mlp.fc1.bias', 'module.blocks.9.mlp.fc2.weight', 'module.blocks.9.mlp.fc2.bias', 'module.blocks.10.scale_shift_table', 'module.blocks.10.attn.qkv.weight', 'module.blocks.10.attn.qkv.bias', 'module.blocks.10.attn.proj.weight', 'module.blocks.10.attn.proj.bias', 'module.blocks.10.cross_attn.q_linear.weight', 'module.blocks.10.cross_attn.q_linear.bias', 'module.blocks.10.cross_attn.kv_linear.weight', 'module.blocks.10.cross_attn.kv_linear.bias', 'module.blocks.10.cross_attn.proj.weight', 'module.blocks.10.cross_attn.proj.bias', 'module.blocks.10.mlp.fc1.weight', 'module.blocks.10.mlp.fc1.bias', 'module.blocks.10.mlp.fc2.weight', 'module.blocks.10.mlp.fc2.bias', 'module.blocks.11.scale_shift_table', 'module.blocks.11.attn.qkv.weight', 'module.blocks.11.attn.qkv.bias', 'module.blocks.11.attn.proj.weight', 'module.blocks.11.attn.proj.bias', 'module.blocks.11.cross_attn.q_linear.weight', 'module.blocks.11.cross_attn.q_linear.bias', 'module.blocks.11.cross_attn.kv_linear.weight', 'module.blocks.11.cross_attn.kv_linear.bias', 'module.blocks.11.cross_attn.proj.weight', 'module.blocks.11.cross_attn.proj.bias', 'module.blocks.11.mlp.fc1.weight', 'module.blocks.11.mlp.fc1.bias', 'module.blocks.11.mlp.fc2.weight', 'module.blocks.11.mlp.fc2.bias', 'module.blocks.12.scale_shift_table', 'module.blocks.12.attn.qkv.weight', 'module.blocks.12.attn.qkv.bias', 'module.blocks.12.attn.proj.weight', 'module.blocks.12.attn.proj.bias', 'module.blocks.12.cross_attn.q_linear.weight', 'module.blocks.12.cross_attn.q_linear.bias', 'module.blocks.12.cross_attn.kv_linear.weight', 'module.blocks.12.cross_attn.kv_linear.bias', 'module.blocks.12.cross_attn.proj.weight', 'module.blocks.12.cross_attn.proj.bias', 'module.blocks.12.mlp.fc1.weight', 'module.blocks.12.mlp.fc1.bias', 'module.blocks.12.mlp.fc2.weight', 'module.blocks.12.mlp.fc2.bias', 'module.blocks.13.scale_shift_table', 'module.blocks.13.attn.qkv.weight', 'module.blocks.13.attn.qkv.bias', 'module.blocks.13.attn.proj.weight', 'module.blocks.13.attn.proj.bias', 'module.blocks.13.cross_attn.q_linear.weight', 'module.blocks.13.cross_attn.q_linear.bias', 'module.blocks.13.cross_attn.kv_linear.weight', 'module.blocks.13.cross_attn.kv_linear.bias', 'module.blocks.13.cross_attn.proj.weight', 'module.blocks.13.cross_attn.proj.bias', 'module.blocks.13.mlp.fc1.weight', 'module.blocks.13.mlp.fc1.bias', 'module.blocks.13.mlp.fc2.weight', 'module.blocks.13.mlp.fc2.bias', 'module.blocks.14.scale_shift_table', 'module.blocks.14.attn.qkv.weight', 'module.blocks.14.attn.qkv.bias', 'module.blocks.14.attn.proj.weight', 'module.blocks.14.attn.proj.bias', 'module.blocks.14.cross_attn.q_linear.weight', 'module.blocks.14.cross_attn.q_linear.bias', 'module.blocks.14.cross_attn.kv_linear.weight', 'module.blocks.14.cross_attn.kv_linear.bias', 'module.blocks.14.cross_attn.proj.weight', 'module.blocks.14.cross_attn.proj.bias', 'module.blocks.14.mlp.fc1.weight', 'module.blocks.14.mlp.fc1.bias', 'module.blocks.14.mlp.fc2.weight', 'module.blocks.14.mlp.fc2.bias', 'module.blocks.15.scale_shift_table', 'module.blocks.15.attn.qkv.weight', 'module.blocks.15.attn.qkv.bias', 'module.blocks.15.attn.proj.weight', 'module.blocks.15.attn.proj.bias', 'module.blocks.15.cross_attn.q_linear.weight', 'module.blocks.15.cross_attn.q_linear.bias', 'module.blocks.15.cross_attn.kv_linear.weight', 'module.blocks.15.cross_attn.kv_linear.bias', 'module.blocks.15.cross_attn.proj.weight', 'module.blocks.15.cross_attn.proj.bias', 'module.blocks.15.mlp.fc1.weight', 'module.blocks.15.mlp.fc1.bias', 'module.blocks.15.mlp.fc2.weight', 'module.blocks.15.mlp.fc2.bias', 'module.blocks.16.scale_shift_table', 'module.blocks.16.attn.qkv.weight', 'module.blocks.16.attn.qkv.bias', 'module.blocks.16.attn.proj.weight', 'module.blocks.16.attn.proj.bias', 'module.blocks.16.cross_attn.q_linear.weight', 'module.blocks.16.cross_attn.q_linear.bias', 'module.blocks.16.cross_attn.kv_linear.weight', 'module.blocks.16.cross_attn.kv_linear.bias', 'module.blocks.16.cross_attn.proj.weight', 'module.blocks.16.cross_attn.proj.bias', 'module.blocks.16.mlp.fc1.weight', 'module.blocks.16.mlp.fc1.bias', 'module.blocks.16.mlp.fc2.weight', 'module.blocks.16.mlp.fc2.bias', 'module.blocks.17.scale_shift_table', 'module.blocks.17.attn.qkv.weight', 'module.blocks.17.attn.qkv.bias', 'module.blocks.17.attn.proj.weight', 'module.blocks.17.attn.proj.bias', 'module.blocks.17.cross_attn.q_linear.weight', 'module.blocks.17.cross_attn.q_linear.bias', 'module.blocks.17.cross_attn.kv_linear.weight', 'module.blocks.17.cross_attn.kv_linear.bias', 'module.blocks.17.cross_attn.proj.weight', 'module.blocks.17.cross_attn.proj.bias', 'module.blocks.17.mlp.fc1.weight', 'module.blocks.17.mlp.fc1.bias', 'module.blocks.17.mlp.fc2.weight', 'module.blocks.17.mlp.fc2.bias', 'module.blocks.18.scale_shift_table', 'module.blocks.18.attn.qkv.weight', 'module.blocks.18.attn.qkv.bias', 'module.blocks.18.attn.proj.weight', 'module.blocks.18.attn.proj.bias', 'module.blocks.18.cross_attn.q_linear.weight', 'module.blocks.18.cross_attn.q_linear.bias', 'module.blocks.18.cross_attn.kv_linear.weight', 'module.blocks.18.cross_attn.kv_linear.bias', 'module.blocks.18.cross_attn.proj.weight', 'module.blocks.18.cross_attn.proj.bias', 'module.blocks.18.mlp.fc1.weight', 'module.blocks.18.mlp.fc1.bias', 'module.blocks.18.mlp.fc2.weight', 'module.blocks.18.mlp.fc2.bias', 'module.blocks.19.scale_shift_table', 'module.blocks.19.attn.qkv.weight', 'module.blocks.19.attn.qkv.bias', 'module.blocks.19.attn.proj.weight', 'module.blocks.19.attn.proj.bias', 'module.blocks.19.cross_attn.q_linear.weight', 'module.blocks.19.cross_attn.q_linear.bias', 'module.blocks.19.cross_attn.kv_linear.weight', 'module.blocks.19.cross_attn.kv_linear.bias', 'module.blocks.19.cross_attn.proj.weight', 'module.blocks.19.cross_attn.proj.bias', 'module.blocks.19.mlp.fc1.weight', 'module.blocks.19.mlp.fc1.bias', 'module.blocks.19.mlp.fc2.weight', 'module.blocks.19.mlp.fc2.bias', 'module.blocks.20.scale_shift_table', 'module.blocks.20.attn.qkv.weight', 'module.blocks.20.attn.qkv.bias', 'module.blocks.20.attn.proj.weight', 'module.blocks.20.attn.proj.bias', 'module.blocks.20.cross_attn.q_linear.weight', 'module.blocks.20.cross_attn.q_linear.bias', 'module.blocks.20.cross_attn.kv_linear.weight', 'module.blocks.20.cross_attn.kv_linear.bias', 'module.blocks.20.cross_attn.proj.weight', 'module.blocks.20.cross_attn.proj.bias', 'module.blocks.20.mlp.fc1.weight', 'module.blocks.20.mlp.fc1.bias', 'module.blocks.20.mlp.fc2.weight', 'module.blocks.20.mlp.fc2.bias', 'module.blocks.21.scale_shift_table', 'module.blocks.21.attn.qkv.weight', 'module.blocks.21.attn.qkv.bias', 'module.blocks.21.attn.proj.weight', 'module.blocks.21.attn.proj.bias', 'module.blocks.21.cross_attn.q_linear.weight', 'module.blocks.21.cross_attn.q_linear.bias', 'module.blocks.21.cross_attn.kv_linear.weight', 'module.blocks.21.cross_attn.kv_linear.bias', 'module.blocks.21.cross_attn.proj.weight', 'module.blocks.21.cross_attn.proj.bias', 'module.blocks.21.mlp.fc1.weight', 'module.blocks.21.mlp.fc1.bias', 'module.blocks.21.mlp.fc2.weight', 'module.blocks.21.mlp.fc2.bias', 'module.blocks.22.scale_shift_table', 'module.blocks.22.attn.qkv.weight', 'module.blocks.22.attn.qkv.bias', 'module.blocks.22.attn.proj.weight', 'module.blocks.22.attn.proj.bias', 'module.blocks.22.cross_attn.q_linear.weight', 'module.blocks.22.cross_attn.q_linear.bias', 'module.blocks.22.cross_attn.kv_linear.weight', 'module.blocks.22.cross_attn.kv_linear.bias', 'module.blocks.22.cross_attn.proj.weight', 'module.blocks.22.cross_attn.proj.bias', 'module.blocks.22.mlp.fc1.weight', 'module.blocks.22.mlp.fc1.bias', 'module.blocks.22.mlp.fc2.weight', 'module.blocks.22.mlp.fc2.bias', 'module.blocks.23.scale_shift_table', 'module.blocks.23.attn.qkv.weight', 'module.blocks.23.attn.qkv.bias', 'module.blocks.23.attn.proj.weight', 'module.blocks.23.attn.proj.bias', 'module.blocks.23.cross_attn.q_linear.weight', 'module.blocks.23.cross_attn.q_linear.bias', 'module.blocks.23.cross_attn.kv_linear.weight', 'module.blocks.23.cross_attn.kv_linear.bias', 'module.blocks.23.cross_attn.proj.weight', 'module.blocks.23.cross_attn.proj.bias', 'module.blocks.23.mlp.fc1.weight', 'module.blocks.23.mlp.fc1.bias', 'module.blocks.23.mlp.fc2.weight', 'module.blocks.23.mlp.fc2.bias', 'module.blocks.24.scale_shift_table', 'module.blocks.24.attn.qkv.weight', 'module.blocks.24.attn.qkv.bias', 'module.blocks.24.attn.proj.weight', 'module.blocks.24.attn.proj.bias', 'module.blocks.24.cross_attn.q_linear.weight', 'module.blocks.24.cross_attn.q_linear.bias', 'module.blocks.24.cross_attn.kv_linear.weight', 'module.blocks.24.cross_attn.kv_linear.bias', 'module.blocks.24.cross_attn.proj.weight', 'module.blocks.24.cross_attn.proj.bias', 'module.blocks.24.mlp.fc1.weight', 'module.blocks.24.mlp.fc1.bias', 'module.blocks.24.mlp.fc2.weight', 'module.blocks.24.mlp.fc2.bias', 'module.blocks.25.scale_shift_table', 'module.blocks.25.attn.qkv.weight', 'module.blocks.25.attn.qkv.bias', 'module.blocks.25.attn.proj.weight', 'module.blocks.25.attn.proj.bias', 'module.blocks.25.cross_attn.q_linear.weight', 'module.blocks.25.cross_attn.q_linear.bias', 'module.blocks.25.cross_attn.kv_linear.weight', 'module.blocks.25.cross_attn.kv_linear.bias', 'module.blocks.25.cross_attn.proj.weight', 'module.blocks.25.cross_attn.proj.bias', 'module.blocks.25.mlp.fc1.weight', 'module.blocks.25.mlp.fc1.bias', 'module.blocks.25.mlp.fc2.weight', 'module.blocks.25.mlp.fc2.bias', 'module.blocks.26.scale_shift_table', 'module.blocks.26.attn.qkv.weight', 'module.blocks.26.attn.qkv.bias', 'module.blocks.26.attn.proj.weight', 'module.blocks.26.attn.proj.bias', 'module.blocks.26.cross_attn.q_linear.weight', 'module.blocks.26.cross_attn.q_linear.bias', 'module.blocks.26.cross_attn.kv_linear.weight', 'module.blocks.26.cross_attn.kv_linear.bias', 'module.blocks.26.cross_attn.proj.weight', 'module.blocks.26.cross_attn.proj.bias', 'module.blocks.26.mlp.fc1.weight', 'module.blocks.26.mlp.fc1.bias', 'module.blocks.26.mlp.fc2.weight', 'module.blocks.26.mlp.fc2.bias', 'module.blocks.27.scale_shift_table', 'module.blocks.27.attn.qkv.weight', 'module.blocks.27.attn.qkv.bias', 'module.blocks.27.attn.proj.weight', 'module.blocks.27.attn.proj.bias', 'module.blocks.27.cross_attn.q_linear.weight', 'module.blocks.27.cross_attn.q_linear.bias', 'module.blocks.27.cross_attn.kv_linear.weight', 'module.blocks.27.cross_attn.kv_linear.bias', 'module.blocks.27.cross_attn.proj.weight', 'module.blocks.27.cross_attn.proj.bias', 'module.blocks.27.mlp.fc1.weight', 'module.blocks.27.mlp.fc1.bias', 'module.blocks.27.mlp.fc2.weight', 'module.blocks.27.mlp.fc2.bias', 'module.final_layer.scale_shift_table', 'module.final_layer.linear.weight', 'module.final_layer.linear.bias'])
# Traceback (most recent call last):
#   File "/home/logan/ray_results/checkpoint_000029/convert_pixart_alpha_to_diffusers.py", line 198, in <module>
#     main(args)
#   File "/home/logan/ray_results/checkpoint_000029/convert_pixart_alpha_to_diffusers.py", line 21, in main
#     converted_state_dict["pos_embed.proj.weight"] = state_dict.pop("x_embedder.proj.weight")
# KeyError: 'x_embedder.proj.weight'
    
    #  python3 scripts/convert_pixart_alpha_to_diffusers_v2.py --multi_scale_train False --orig_ckpt_path akash-thumper-v1-checkpoints/TorchTrainer_2023-12-27_15-34-10/TorchTrainer_70101_00000_0_2023-12-27_15-34-11/checkpoint_000016/epoch_27_step_5253.pth --dump_path /mnt/sabrent/at3s_test/ --only_transformer False

    #  python3 launch_foundry.py --env pixart --cmd 'python scripts/train_ray5_works_moml_train_iter22tst1wds.py configs/pixart_config/at1_xl2_img256_sam_latent.py --n_gpu_workers 14' --jobname mdstrain97270