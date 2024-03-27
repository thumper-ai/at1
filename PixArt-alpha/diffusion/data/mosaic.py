"""Streaming LAION dataset."""

from io import BytesIO
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image
from streaming import Stream, StreamingDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPTokenizer

# from diffusion.datasets.laion.transforms import LargestCenterSquare

# Disable PIL max image size limit
Image.MAX_IMAGE_PIXELS = None


class StreamingLAIONDataset(StreamingDataset):
    """Implementation of the LAION dataset as a streaming dataset.

    Args:
        streams (Sequence[Stream], optional): One or more Streams to stream/cache samples from. StreamingLAIONDataset
            uses either ``streams`` or ``remote``/``local``. Default:``None``.
        remote (str, optional): Remote directory (S3 or local filesystem) where dataset is stored. Default: ``None``.
        local (str, optional): Local filesystem directory where dataset is cached during operation. Default: ``None``.
        split (str, optional): The dataset split to use. Currently, only ``None`` is supported. Default: ``None``.
        shuffle (bool): Whether to shuffle the samples in this dataset. Default: ``False``.
        shuffle_algo (str): What shuffle algorithm to use. Default: ``'py1s'``.
        shuffle_block_size (int): Unit of shuffling. Default: ``1 << 18``.
        tokenizer_name_or_path (str): The name or path of the tokenizer to use. Default: ``'stabilityai/stable-diffusion-2-base'``.
        transform (Optional[Callable]): The transforms to apply to the image. Default: ``None``.
        predownload (Optional[int]): The number of samples to prefetch. If ``None``, its value is set to ``8 * batch_size``. Default: ``None``.
        download_retry (Optional[int]): The number of times to retry a download. Default: ``2``.
        download_timeout (Optional[float]): The timeout for a download. Default: ``120``.
        batch_size (Optional[int]): Hint batch_size that will be used on each device's DataLoader. Default: ``None``.
        image_size (Optional[int]): The size to resize the image to. Default: ``None``.
        num_canonical_nodes (int, optional): The number of canonical nodes for shuffle. Default: ``None``.
    """

    def __init__(
        self,
        streams: Optional[Sequence[Stream]] = None,
        remote: Optional[str] = None,
        local: Optional[str] = None,
        split: Optional[str] = None,
        shuffle: bool = False,
        shuffle_algo: str = 'py1s',
        shuffle_block_size: int = 1 << 18,
        # tokenizer_name_or_path: str = 'stabilityai/stable-diffusion-2-base',
        # caption_drop_prob: float = 0.0,
        # transform: Optional[Callable] = None,
        predownload: Optional[int] = None,
        download_retry: int = 2,
        download_timeout: float = 120,
        batch_size: Optional[int] = None,
        image_size: Optional[int] = None,
        num_canonical_nodes: Optional[int] = None,
    ) -> None:

        super().__init__(
            streams=streams,
            remote=remote,
            local=local,
            split=split,
            shuffle=shuffle,
            shuffle_algo=shuffle_algo,
            shuffle_block_size=shuffle_block_size,
            predownload=predownload,
            keep_zip=False,
            download_retry=download_retry,
            download_timeout=download_timeout,
            validate_hash=None,
            batch_size=batch_size,
            num_canonical_nodes=num_canonical_nodes,
        )

        # self.transform = transform
        # self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name_or_path, subfolder='tokenizer')
        # self.caption_drop_prob = caption_drop_prob
        self.image_size = image_size

    def __getitem__(self, index):
        #  for i in range(20):
        #     try:
        sample = super().__getitem__(index)
        out={}

        if 'vae_features' in sample:
            out["vae"] =  torch.from_numpy(np.frombuffer(sample['vae_features'], dtype=np.float32).copy()).reshape( 8, 32, 32).squeeze()
            # (1, 8, 32, 32)
        
        if 'txt_features' in sample:
            out['txt_features'] = torch.from_numpy( np.frombuffer(sample['txt_features'], dtype=np.float32).copy()).reshape(( 120, 4096))
        # (1, 120, 4096)
        if 'attention_mask' in sample:
            out['attention_mask'] = torch.from_numpy( np.frombuffer(sample['attention_mask'], dtype=np.int64).copy()).reshape(120)
        # (1, 120)
        return out

            # except Exception as e:
            #     print(index,' info is not correct', e)
            #     index = np.random.randint(len(self))
            #     index = np.random.randint(20)
            # raise RuntimeError('Too many failed data attempts')
         
    
        # sample = super().__getitem__(index)
        # out={}

        # if 'vae_features' in sample:
        #     out["vae"] =  torch.from_numpy(np.frombuffer(sample['vae_features'], dtype=np.float32).copy()).reshape( 8, 32, 32).squeeze()
        #     # (1, 8, 32, 32)
        
        # if 'txt_features' in sample:
        #     out['txt_features'] = torch.from_numpy( np.frombuffer(sample['txt_features'], dtype=np.float32).copy()).reshape(( 120, 4096))
        # # (1, 120, 4096)
        # if 'attention_mask' in sample:
        #     out['attention_mask'] = torch.from_numpy( np.frombuffer(sample['attention_mask'], dtype=np.int64).copy()).reshape(120)
        # # (1, 120)
        # return out
       



def build_streaming_laion_dataloader(
    remote: Union[str, List],
    local: Union[str, List],
    batch_size: int,
    # tokenizer_name_or_path: str = 'stabilityai/stable-diffusion-2-base',
    # caption_drop_prob: float = 0.0,
    resize_size: int = 256,
    num_samples: Optional[int] = None,
    predownload: int = 100_000,
    download_retry: int = 2,
    download_timeout: float = 120,
    drop_last: bool = True,
    shuffle: bool = True,
    num_canonical_nodes: Optional[int] = None,
    **dataloader_kwargs,
):
    """Builds a streaming LAION dataloader.

    Args:
        remote (str, Sequence[str]): One or more remote directories (S3 or local filesystem) where dataset is stored.
        local (str, Sequence[str]): One or more local filesystem directories where dataset is cached during operation.
        batch_size (int): The batch size to use.
        tokenizer_name_or_path (str): The name or path of the tokenizer to use. Default: ``'stabilityai/stable-diffusion-2-base'``.
        caption_drop_prob (float): The probability of dropping a caption. Default: ``0.0``.
        resize_size (int): The size to resize the image to. Default: ``256``.
        num_samples (Optional[int]): The number of samples to use. Default: ``None`` uses all available samples.
        predownload (Optional[int]): The number of samples to prefetch. Default: ``100_000``.
        download_retry (Optional[int]): The number of times to retry a download. Default: ``2``.
        download_timeout (Optional[float]): The timeout for a download. Default: ``120``.
        drop_last (bool): Whether to drop the last batch if it is incomplete. Default: ``True``.
        shuffle (bool): Whether to shuffle the samples in this dataset. Default: ``True``.
        num_canonical_nodes (int, optional): The number of canonical nodes for shuffle. Default: ``None``.
        **dataloader_kwargs: Additional arguments to pass to the dataloader.
    """
    if isinstance(remote, str) and isinstance(local, str):
        # Hacky... make remote and local lists to simplify downstream code
        remote, local = [remote], [local]
    elif isinstance(remote, Sequence) and isinstance(local, Sequence):
        if len(remote) != len(local):
            ValueError(
                f'remote and local Sequences must be the same length, got lengths {len(remote)} and {len(local)}')
    else:
        ValueError(f'remote and local must be both Strings or Sequences, got types {type(remote)} and {type(local)}.')

    # Create a Stream for each (remote, local) pair
    streams = []
    for r, l in zip(remote, local):
        streams.append(Stream(remote=r, local=l, download_retry=download_retry, download_timeout=download_timeout))

    # center_square_crop = LargestCenterSquare(resize_size)
    # Normalize from 0 to 1 to -1 to 1
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # transform = transforms.Compose([center_square_crop, transforms.ToTensor(), normalize])
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    dataset = StreamingLAIONDataset(
        streams=streams,
        split=None,
        shuffle=shuffle,
        # tokenizer_name_or_path=tokenizer_name_or_path,
        # caption_drop_prob=caption_drop_prob,
        # transform=transform,
        predownload=predownload,
        download_retry=download_retry,
        download_timeout=download_timeout,
        batch_size=batch_size,
        image_size=resize_size,
        num_canonical_nodes=num_canonical_nodes,
    )
    # Create a subset of the dataset
    if num_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(num_samples))  # type: ignore

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=None,
        drop_last=drop_last,
        **dataloader_kwargs,
    )

    return dataloader