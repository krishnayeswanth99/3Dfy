import torch
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline
from cog_sdxl.dataset_and_utils import TokenEmbeddingsHandler
from diffusers.models import AutoencoderKL
from PIL import Image
from tqdm import tqdm
from image_util import get_360
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
    video = cv2.VideoWriter(file_path, fourcc, 5, (1024, 1024))

    # Write the images to the video
    for img_lis in video_seconds:
        for img in img_lis:
            video.write(cv2.cvtColor(get_360(np.array(img)), cv2.COLOR_RGB2BGR))

    video.release()

prompts = [
    "Create a mesmerizing video that portrays the passage of time through various landscapes, seasons, and natural phenomena. Utilize the stable diffusion model to seamlessly blend transitions between different time periods, showcasing the beauty of evolution and change.",
    "Generate an ethereal video exploring surreal dreamscapes and imaginative landscapes. Use the stable diffusion model to evoke a sense of wonder and curiosity as the viewer is transported through vibrant and otherworldly realms filled with fantastical creatures and mesmerizing visuals.",
    "Craft a visually stunning video capturing the pulsating energy and rhythm of urban life. Employ the stable diffusion model to seamlessly blend scenes of bustling city streets, architectural marvels, and vibrant cultural events, creating a symphony of movement and color that reflects the dynamic essence of metropolitan living.",
    "Dive into the cosmic depths of the universe with a breathtaking video journey through celestial wonders and cosmic phenomena. Harness the power of the stable diffusion model to depict awe-inspiring scenes of galaxies, nebulae, and celestial bodies in motion, accompanied by a mesmerizing soundtrack that transports viewers to the far reaches of space.",
    "Create an enchanting video exploring mystical realms and magical encounters. Use the stable diffusion model to weave together scenes of enchanting forests, ancient ruins, and mystical creatures, imbuing the video with an air of mystery and wonder that invites viewers on a spellbinding adventure through the realms of fantasy"
]

generate_images_to_video(prompts)