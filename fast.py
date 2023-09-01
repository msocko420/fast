from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware  
from PIL import Image
from diffusion.gaussian_diffusion import diffusion_from_config
from util.notebooks import decode_latent_mesh
import torch
import os
from shap_e.models.download import load_model, load_config
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.util.notebooks import decode_latent_mesh
from PIL import Image
from io import BytesIO
import logging
from fastapi.logger import logger as fastapi_logger
logging.basicConfig(level=logging.DEBUG)
fastapi_logger.addHandler(logging.StreamHandler())

app = FastAPI()

origins = [
    "http://192.168.1.118:3000", "http://localhost:5173", "http://localhost:8000", "http://192.168.1.118", "http://192.168.1.118:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

guidance_scale = 18.0

@app.post("/render/obj")
async def render_obj(text_prompt: str = Form(None), uploaded_image_file: UploadFile = File(None)):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        xm = load_model('transmitter', device=device)
        xm = torch.nn.DataParallel(xm) if torch.cuda.device_count() > 1 else xm
        xm.to(device)
        diffusion = diffusion_from_config(load_config('diffusion'))

        if text_prompt:
            model = load_model('text300M', device=device)
            model = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
            model.to(device)
            batch_size = 1
            prompt = text_prompt
            latents = sample_latents(
                batch_size=batch_size,
                model=model.module if isinstance(model, torch.nn.DataParallel) else model,
                diffusion=diffusion,
                guidance_scale=guidance_scale,
                model_kwargs=dict(texts=[prompt] * batch_size),
                progress=True,
                clip_denoised=True,
                use_fp16=True,
                use_karras=True,
                karras_steps=64,
                sigma_min=1e-3,
                sigma_max=160,
                s_churn=0,
            )
        elif uploaded_image_file:
            image_data = await uploaded_image_file.read()
            image = Image.open(BytesIO(image_data))
            model = load_model('image300M', device=device)
            model = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
            model.to(device)
            batch_size = 1
            latents = sample_latents(
                batch_size=batch_size,
                model=model.module if isinstance(model, torch.nn.DataParallel) else model,
                diffusion=diffusion,
                guidance_scale=guidance_scale,
                model_kwargs=dict(images=[image] * batch_size),
                progress=True,
                clip_denoised=True,
                use_fp16=True,
                use_karras=True,
                karras_steps=64,
                sigma_min=1e-3,
                sigma_max=160,
                s_churn=0,
            )

        obj_files = []
        for i, latent in enumerate(latents):
            original_xm = xm.module if isinstance(xm, torch.nn.DataParallel) else xm
            t = decode_latent_mesh(original_xm, latent).tri_mesh()
            obj_filename = f'WOW_LOOK_WHAT_YOU_DID{i}.obj'
            with open(obj_filename, 'w') as f:
                t.write_obj(f)
            obj_files.append(obj_filename)

        headers = {
            "Content-Disposition": f"attachment; filename=WOW_LOOK_WHAT_YOU_DID0.obj"
        }
        return FileResponse(obj_files[0], headers=headers)

    except Exception as e:
        # Log the exception to the server logs for debugging
        fastapi_logger.error(f"Exception occurred: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the request.")
