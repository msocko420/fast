
import streamlit as st
import torch
import os
from shap_e.models.download import load_model, load_config
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.util.notebooks import decode_latent_mesh
import pyvista as pv
from stpyvista import stpyvista
from PIL import Image
import base64
import subprocess
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from io import BytesIO
from PIL import ImageSequence

# Function to convert video to base64
def video_to_base64(video_path):
    with open(video_path, 'rb') as video_file:
        video_base64 = base64.b64encode(video_file.read()).decode()
    return video_base64

# Embed a looping and autoplaying video without controls
video_file_path = 'C:/Users/mike/shap-e/shap_e/logo.MP4' # Update with the path to your video
video_base64 = video_to_base64(video_file_path)
video_html = f"""
<video width="690" height="740" autoplay loop muted>
    <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
    Your browser does not support the video tag.
</video>
"""
st.markdown(video_html, unsafe_allow_html=True)

# Custom CSS for fontsize, styling, theme color, text alignment
st.markdown("""
    <style>
        body {
            background-color: #F0E6FA !important; /* Lighter purple color */
        }

        body, h1, h2, h3, p, input, button {
            font-family: 'Gill Sans', sans-serif; /* Add this line */
            font-size: 110%;
        }

        .stBtn>button {
            width:100%;
            padding: 20px 0; 
            color:#EFEFEF;
            background-color:#800080;
            border-radius:20px;
            border:0;
        }

        .stTextInput>div>div>input {
            border-radius:28px;
            border: 1px solid #800080;
        }

        .stSidebar>.sidebar-content {
            background-color: #EFEFEF;
        }
        .css-17eq0hr {
            background-color: #EFEFEF;
            border-radius:17px;
            color:#800080;
        }
        .block-container {
            background-color: #EFEFEF; /* Light purple color */
        }
    </style>
    """, unsafe_allow_html=True)


# Set sidebar
st.sidebar.header("Settings")
guidance_scale = st.sidebar.slider("Guidance Scale", min_value=1.0, max_value=30.0, value=17.0, step=0.5)
mesh_color = st.sidebar.color_picker("Mesh Color", "#800080")
background_color = st.sidebar.color_picker("Background Color", "#ADD8E6")
line_width = st.sidebar.slider("Line Width", min_value=1, max_value=10, value=1)

# Title and description
st.title("Generative 3D For All!")
st.subheader("This Prototype is a starting point for a new Generative AI platform. With proper resources and time, this platform aims to provide the largest array of cutting Edge Generative AI technology.")
st.subheader("Enter a Prompt Below or upload an image. Prompts will generate OBJ files or GIF renders, with more to come. After AI generates your model, download your .obj file and click our Cubee link below to have one of the first ever 3D Printed models rendered from AI!")
st.markdown("<a href='https://cubee3d.com/store/KashMunkey%20Creative%20LLC' target='_blank' style='color:#9370DB'>Cubee Network of printshops World Wide!</a>", unsafe_allow_html=True)

# Section for OBJ rendering

st.header("3D OBJ Rendering")
# Text input for user prompt and file uploader for image prompt
text_prompt = st.text_input('OBJ Text Input:', key='obj_text_prompt')
uploaded_image_file = st.file_uploader('OBJ Image Input: Or try your luck with images!', type=['png', 'jpg', 'jpeg'], key='obj_image_prompt')

if st.button('Enter the rabbit hole..'):
    with st.spinner('Generating 3D Object...'):
        progress_bar = st.progress(0)

        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load models and configurations
        xm = load_model('transmitter', device=device)
        xm = torch.nn.DataParallel(xm) if torch.cuda.device_count() > 1 else xm
        xm.to(device)
        diffusion = diffusion_from_config(load_config('diffusion'))
        progress_bar.progress(0.25)

        if text_prompt:
            model = load_model('text300M', device=device)
            model = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
            model.to(device)
            batch_size = 3
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
            model = load_model('image300M', device=device)
            model = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
            model.to(device)
            batch_size = 3
            image = Image.open(uploaded_image_file)

            latents = sample_latents(
                batch_size=batch_size,
                model=model.module if isinstance(model, torch.nn.DataParallel) else model,
                diffusion=diffusion,
                guidance_scale= guidance_scale,
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

        progress_bar.progress(0.50)
        for i, latent in enumerate(latents):
            original_xm = xm.module if isinstance(xm, torch.nn.DataParallel) else xm # Access the original model
            t = decode_latent_mesh(original_xm, latent).tri_mesh() # Using original_xm here

            # Save OBJ file
            obj_filename = f'WOW_LOOK_WHAT_YOU_DID{i}.obj'
            with open(obj_filename, 'w') as f:
                t.write_obj(f)

            # Read the OBJ file using PyVista
            mesh = pv.read(obj_filename)

            # Initialize a plotter object
            plotter = pv.Plotter(window_size=[800, 800])

            # Add mesh to the plotter
            plotter.add_mesh(mesh, color=mesh_color, line_width=line_width)

            # Final touches
            plotter.view_isometric()
            plotter.background_color = background_color

            # Display the plotter
            stpyvista(plotter, key=f"pv_mesh_{i}")
            # Provide download button
            st.download_button(f'Download {obj_filename}', data=open(obj_filename, 'r').read(), file_name=obj_filename)
            os.remove(obj_filename)            
            torch.cuda.empty_cache()
        progress_bar.progress(1.0)
        st.success('3D Object generated! See above.')


st.header("FULL COLOR GIF RENDERING")
# Text input for user prompt and file uploader for image prompt
text_prompt1 = st.text_input('Hope for the best, expect the worse..', '', key='other_text_prompt')
uploaded_image_file1 = st.file_uploader('Or try your luck with images!', type=['png', 'jpg', 'jpeg'], key='other_image_prompt')

if st.button('Something more fun perhaps?'):
    with st.spinner('WOOOOAAAHH ITS MAGIC!!'):
        progress_bar = st.progress(0)

        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load models and configurations
        original_xm = load_model('transmitter', device=device)
        xm = torch.nn.DataParallel(original_xm) if torch.cuda.device_count() > 1 else original_xm
        xm.to(device)

        diffusion = diffusion_from_config(load_config('diffusion'))
        progress_bar.progress(0.25)

        if text_prompt1: # Change this to text_prompt1
            model = load_model('text300M', device=device)
            model = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
            model.to(device)
            batch_size = 1
            prompt = text_prompt1 # Change this to text_prompt1



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
                karras_steps=124,
                sigma_min=1e-3,
                sigma_max=160,
                s_churn=0,
            )
        elif uploaded_image_file1: # Change this to uploaded_image_file1
            model = load_model('image300M', device=device)
            model = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
            model.to(device)
            batch_size = 1
            image = Image.open(uploaded_image_file1) # Change this to uploaded_image_file1

            latents = sample_latents(
                batch_size=batch_size,
                model=model.module if isinstance(model, torch.nn.DataParallel) else model,
                diffusion=diffusion,
                guidance_scale= guidance_scale,
                model_kwargs=dict(images=[image] * batch_size),
                progress=True,
                clip_denoised=True,
                use_fp16=True,
                use_karras=True,
                karras_steps=124,
                sigma_min=1e-3,
                sigma_max=160,
                s_churn=0,
            )

        progress_bar.progress(0.25)
        render_mode = 'stf' # you can change this to 'stf'
        size = 180 # this is the size of the renders; higher values take longer to render.

                    # Assuming latents is defined earlier in your code
        cameras = create_pan_cameras(size, device)
            
        for i, latent in enumerate(latents):
            images = decode_latent_images(original_xm, latent, cameras, rendering_mode=render_mode)
            progress_bar.progress(0.50)
            # Convert the images into a GIF
            img, *imgs = images
            buf = BytesIO()
            img.save(buf, format='GIF', append_images=imgs, save_all=True, duration=200, loop=2)
            # Display the GIF in Streamlit
            st.image(buf.getvalue(), caption='Generated GIF', use_column_width=True)
        progress_bar.progress(1.0)
        st.success("I GIVE YOU COLOR!")

st.header("Stable Dreamfusion- THIS RENDER OPTION IS STILL BEING BUILT- STAY TUNED!")
st.subheader("Enter Parameters for Stable Dreamfusion")

def dreamfusion_training(Prompt_text, Workspace):
    # Build the command-line arguments
    command_args = [
        "python", "main.py",
        "--text", Prompt_text,
        "--workspace", Workspace,
        "-O", # This corresponds to the "-O" argument in main.py
    ]

    # Execute the command
    result = subprocess.run(command_args, capture_output=True, text=True)

    # You can process the result.stdout and result.stderr to handle the output and errors
    if result.returncode == 0:
        st.success('Training Complete!')
        # You may display additional information from result.stdout
    else:
        st.error('Training failed!')
        st.error(result.stderr) # Displaying the error for troubleshooting

# Widgets for user input
Prompt_text = st.text_input('Prompt Text', 'a DSLR photo of a delicious hamburger')
Workspace = st.text_input('Workspace', 'trial')

if st.button("Drift into the Void.."):
    with st.spinner("Go get a snack and a blanket..it's going to be a while"):
        dreamfusion_training(Prompt_text, Workspace)

st.header("We plan to have full functionality for Stable-dreamfusion, as well as many other features showcased in the GitHub Page linked below")
st.markdown("<a href='https://github.com/threestudio-project/threestudio#instructnerf2nerf-' target='_blank'>Three Studios on Github</a>", unsafe_allow_html=True)

# Logo image
logo1_path = 'C:/Users/mike/shap-e/shap_e/monkey.png' # Add the correct path to your logo image
logo1_image = Image.open(logo1_path)
st.image(logo1_image, width=670, caption="Kash Munkey Creative") # You can adjust the width as needed
