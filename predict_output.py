import PIL

import skimage.io as io
import torch
import clip
from transformers import GPT2Tokenizer

from train import ClipCaptionModel, ClipCaptionPrefix
from predict import generate_beam, generate2

model_path =  "./coco_train/coco_prefix-{epoch}.pt"
CPU = torch.device('cpu')


def get_device(device_id: int):
    if not torch.cuda.is_available():
        return CPU
    device_id = min(torch.cuda.device_count() - 1, device_id)
    return torch.device(f'cuda:{device_id}')


def predict_caption(model, clip_model, tokenizer, image_url, use_beam_search=False):
    image = io.imread(image_url)
    pil_image = PIL.Image.fromarray(image)

    image = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        # if type(model) is ClipCaptionE2E:
        #     prefix_embed = model.forward_image(image)
        # else:
        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
        prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
    if use_beam_search:
        generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
    else:
        generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)
    return generated_text_prefix


if __name__ == "__main__":
    prefix_length = 40
    prefix_length_clip = 40
    prefix_dim = 512
    num_layers = 8
    model_load_epoch = "009"
    image_url = ''

    model = ClipCaptionPrefix(prefix_length, clip_length=prefix_length_clip, prefix_size=prefix_dim,
                              num_layers=num_layers, mapping_type='transformer')

    model.load_state_dict(torch.load(model_path.format(epoch=model_load_epoch), map_location=CPU))
    model = model.eval()
    device = get_device(0)
    clipcap_model = model.to(device)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    print(predict_caption(clipcap_model, clip_model, tokenizer, image_url))


