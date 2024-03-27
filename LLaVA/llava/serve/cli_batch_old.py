import argparse
import os.path

import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import pandas as pd
import glob
from tqdm import tqdm

def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    paths = glob.glob(os.path.join(args.image_folder,"*.png"))
    paths += glob.glob(os.path.join(args.image_folder,"*.jpg"))
    paths += glob.glob(os.path.join(args.image_folder,"*.jpeg"))
    paths +=  glob.glob(os.path.join(args.image_folder,"*.webp"))
    # prompt=args.system_prompt+" USER: <image>\n"+args.user_prompt+"\nASSISTANT:"
    prompt="USER: <image>\n"+args.user_prompt+"\nASSISTANT:"
    file_names=[]
    texts=[]
    image_tensors=[]
    for path in tqdm(paths):
        image = Image.open(path)

        # Check if the image has an alpha (transparency) channel
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            # Create a new white background image
            background = Image.new('RGB', image.size, (255, 255, 255))

            # Paste the image onto the background using the alpha channel as a mask
            background.paste(image, (0, 0), image)

        # Similar operation in model_worker.py
        image_tensor = process_images([image], image_processor, args)
        image_tensors.append(image_tensor)

    print(image_tensor.size())
    image_tensor = torch.cat(image_tensors, 0)
    print(image_tensor.size())
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    # prompt = "このイラストを日本語でできる限り詳細に説明してください。表情や髪の色、目の色、耳の種類、服装、服の色など注意して説明してください。説明は反復を避けてください。"

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    print("input_ids",input_ids,  input_ids.size())
    # input_ids=input_ids.repeat(15,0)
    # input_ids=torch.cat([input_ids,input_ids],0)
    input_ids= torch.repeat_interleave(input_ids, repeats=image_tensor.size()[0], dim=0)
    print("input_ids",input_ids,  input_ids.size())


    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
    print("output_ids",output_ids.size())
    for a in range(output_ids.size()[0]):
        outputs = tokenizer.decode(output_ids[a, input_ids.shape[1]:]).strip()
        print(outputs)
        texts.append(outputs)
        file_names.append(paths[a].split("/")[-1])

    if args.debug:
        print("\n", {"outputs": outputs}, "\n")

    pd.DataFrame({"file_name":file_names,"text":texts}).to_csv(args.output_csv,index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    # parser.add_argument("--system-prompt", type=str, required=True)
    parser.add_argument("--user-prompt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    args = parser.parse_args()
    main(args)

# python -m llava.serve.cli_batch --model-path liuhaotian/llava-v1.5-13b \ 
# --load-8bit \
# --system-prompt あなたは日本語を喋る人工知能です。誠実に画像をもとに日本語で応答を返してください。 \
# --user-prompt このイラストを日本語でできる限り詳細に説明してください。表情や髪の色、目の色、耳の種類、服装、服の色 など注意して説明してください。説明は反復を避けてください。\
#  --image-folder '/mnt/NVM/test'  \
# --output-csv '/mnt/NVM/test/metadata.csv'


# python -m llava.serve.cli_batch --model-path '/mnt/sabrent/llava-v1.5-7b' \ 
# --load-4bit \
# --max-new-tokens 128 \
# --user-prompt 'describe this image and its style in a highly detailed manner' \
# --image-folder '/home/logan/ldb/r2'  \
# --output-csv '/home/logan/thumperai/test.csv'

# 15images