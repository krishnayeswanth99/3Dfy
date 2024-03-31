import torch
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from cog_sdxl.dataset_and_utils import TokenEmbeddingsHandler
from diffusers.models import AutoencoderKL
from PIL import Image
from tqdm import tqdm
from image_util import get_360
from utils import flush
import numpy as np
import cv2


def generate_images_to_video(prompts=[], fps=5, file_path='video.mp4'):

    pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
    ).to("cuda")

    pipe.load_lora_weights("jbilcke-hf/sdxl-panorama", weight_name="lora.safetensors")

    text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]

    embedding_path = hf_hub_download(repo_id="jbilcke-hf/sdxl-panorama", filename="embeddings.pti", repo_type="model")
    embhandler = TokenEmbeddingsHandler(text_encoders, tokenizers)
    embhandler.load_embeddings(embedding_path)

    video_seconds = []

    for i in tqdm(prompts):
        video_seconds.append([])
        for j in range(fps):
            prompt=f"{i} <s0><s1>"
            image = pipe(
                prompt,
                cross_attention_kwargs={"scale": 0.8}
            ).images[0]
            video_seconds[-1].append(image)


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(file_path, fourcc, fps, (2048, 1024))

    # Write the images to the video
    for img_lis in video_seconds:
        for img in img_lis:
            video.write(cv2.cvtColor(get_360(cv2.resize(np.array(img),(1024,1024))), cv2.COLOR_RGB2BGR))

    video.release()
    
    del pipe
    flush()
    
    return video_seconds

def __generate_3d_from_image_text(img, prompt='', n=1):

    pipe_xl = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe_img2img = StableDiffusionXLImg2ImgPipeline(
        vae=pipe_xl.vae,
        text_encoder=pipe_xl.text_encoder,
        text_encoder_2=pipe_xl.text_encoder_2,
        tokenizer=pipe_xl.tokenizer,
        tokenizer_2=pipe_xl.tokenizer_2,
        unet=pipe_xl.unet,
        scheduler=pipe_xl.scheduler,
        # safety_checker=None,
        feature_extractor=pipe_xl.feature_extractor,
        # requires_safety_checker=False,
        image_encoder = pipe_xl.image_encoder,
        add_watermarker = False
    ).to("cuda")

    pipe_img2img.load_lora_weights("jbilcke-hf/sdxl-panorama",
                                weight_name="lora.safetensors")

    text_encoders = [pipe_img2img.text_encoder, pipe_img2img.text_encoder_2]
    tokenizers = [pipe_img2img.tokenizer, pipe_img2img.tokenizer_2]

    embedding_path = hf_hub_download(repo_id="jbilcke-hf/sdxl-panorama", filename="embeddings.pti", repo_type="model")
    embhandler = TokenEmbeddingsHandler(text_encoders, tokenizers)
    embhandler.load_embeddings(embedding_path)

    images = []

    for _ in tqdm(range(n)):
        images.append(pipe_img2img(prompt=prompt+" <s0><s1>", image=img, strength=0.75,
                        guidance_scale=7.5,
                        cross_attention_kwargs={"scale": 0.8},
                        added_cond_kwargs={}).images[0])
    
    del pipe_xl, pipe_img2img, text_encoders, tokenizers, embhandler

    flush()
    
    return images

def generate_similar_image_to_video(images=[], prompts=[], fps=5, file_path='video.mp4'):

    video_seconds = []

    for img, prompt in zip(images, prompts):        
        video_seconds.append(__generate_3d_from_image_text(img, prompt))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(file_path, fourcc, fps, (2048, 1024))

    for img_lis in video_seconds:
        for img in img_lis:
            video.write(cv2.cvtColor(get_360(cv2.resize(np.array(img),(1024,1024))), cv2.COLOR_RGB2BGR))

    video.release()

    flush()

    return video_seconds

# prompts = [
#     "Create a mesmerizing video that portrays the passage of time through various landscapes, seasons, and natural phenomena. Utilize the stable diffusion model to seamlessly blend transitions between different time periods, showcasing the beauty of evolution and change.",
#     "Generate an ethereal video exploring surreal dreamscapes and imaginative landscapes. Use the stable diffusion model to evoke a sense of wonder and curiosity as the viewer is transported through vibrant and otherworldly realms filled with fantastical creatures and mesmerizing visuals.",
#     "Craft a visually stunning video capturing the pulsating energy and rhythm of urban life. Employ the stable diffusion model to seamlessly blend scenes of bustling city streets, architectural marvels, and vibrant cultural events, creating a symphony of movement and color that reflects the dynamic essence of metropolitan living.",
#     "Dive into the cosmic depths of the universe with a breathtaking video journey through celestial wonders and cosmic phenomena. Harness the power of the stable diffusion model to depict awe-inspiring scenes of galaxies, nebulae, and celestial bodies in motion, accompanied by a mesmerizing soundtrack that transports viewers to the far reaches of space.",
#     "Create an enchanting video exploring mystical realms and magical encounters. Use the stable diffusion model to weave together scenes of enchanting forests, ancient ruins, and mystical creatures, imbuing the video with an air of mystery and wonder that invites viewers on a spellbinding adventure through the realms of fantasy"
# ]

# generate_images_to_video(prompts)
