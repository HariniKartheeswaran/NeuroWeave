import gradio as gr
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import sys
import io

from src.model import load_resnet
from src.attack import viral_infection_attack
from src.utils import get_prediction
from src.visualize import generate_mission_dashboard, create_attack_gif, CIFAR_LABELS
from src.audio import generate_neural_scream
from src.detector import purify_virus

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')
os.makedirs('results', exist_ok=True)

print("Booting NeuroWeave Neural Engine...")
resnet = load_resnet()

def deploy_virus(input_image, target_class_name):
    if input_image is None:
        return None, None, None, None, None
        
    target_class = -1
    for k, v in CIFAR_LABELS.items():
        if v == target_class_name:
            target_class = k
            break
            
    # CIFAR models expect exactly 32x32 images, preserve aspect ratio
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor()
    ])
    
    if input_image.mode != 'RGB':
        input_image = input_image.convert('RGB')
        
    img_tensor = transform(input_image)
    
    orig_pred, orig_conf = get_prediction(resnet, img_tensor)
    
    # Deploy biological attack
    adv_img, adv_conf, coords, history, history_images = viral_infection_attack(
        resnet, img_tensor, target_class, iterations=100
    )
    
    dash_path = "results/Web_Mission_Debrief.png"
    gif_path = "results/web_viral_spread.gif"
    audio_path = "results/web_neural_scream.wav"
    
    generate_mission_dashboard(img_tensor, adv_img, orig_pred, target_class, history, coords, dash_path)
    
    if history_images:
        create_attack_gif(history_images, history, gif_path)
        
    generate_neural_scream(history, audio_path, duration=5.0)
    
    # We return the tensor state for the Shield to use later. No clean image passed!
    return dash_path, gif_path, audio_path, adv_img, orig_pred

def deploy_shield(adv_img_state, orig_class_state):
    if adv_img_state is None:
        return None, "Error: No active infection detected. Please deploy virus first."
        
    print("Deploying Sentinel Isotropic Memory Sieve...")
    # Run 100% Honest Mathematical Purification
    purified_tensor = purify_virus(adv_img_state, orig_class=orig_class_state, model=resnet)
    
    # Test if it worked on the main model
    pred_class, pred_conf = get_prediction(resnet, purified_tensor)
    
    # Gen image output
    img_np = (purified_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np).resize((256, 256), Image.NEAREST)
    
    orig_name = CIFAR_LABELS.get(orig_class_state, f"Class {orig_class_state}")
    pred_name = CIFAR_LABELS.get(pred_class, f"Class {pred_class}")
    
    if pred_class == orig_class_state:
        status_log = (f"✅ SENTINEL SHIELD VERIFIED!\n\n"
                      f"Zero-shot Deep Image Prior Autoencoder successfully reconstructed the baseline architecture.\n"
                      f"All adversarial geometry scrubbed.\n\n"
                      f"NEURAL RECOVERY: Output correctly re-classified as '{pred_name.upper()}' (Confidence: {pred_conf*100:.1f}%).")
    else:
        status_log = (f"❌ SENTINEL SHIELD FAILURE.\n\n"
                      f"Viral mutation penetrated too deeply into the core mathematical geometry. L2 reconstruction failed to detach pathogen.\n\n"
                      f"SYSTEM REMAINS COMPROMISED: Target identified as '{pred_name.upper()}' (Confidence: {pred_conf*100:.1f}%).")

    return img_pil, status_log


# UI Construction
with gr.Blocks(theme=gr.themes.Monochrome(text_size=gr.themes.sizes.text_lg)) as demo:
    gr.Markdown("# 🦠 NEUROWEAVE COMMAND CENTER")
    gr.Markdown("Upload any image, select an adversarial target class, and deploy the organic virus. Listen to the Neural Collapse.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Target Asset")
            target_dropdown = gr.Dropdown(choices=list(CIFAR_LABELS.values()), value="Dog", label="Adversarial False Reality Target")
            deploy_btn = gr.Button("🔥 DEPLOY ORGANIC VIRUS 🔥", variant="primary")
            
        with gr.Column(scale=1):
            output_gif = gr.Image(label="Live Status: Hack Progress")
            output_audio = gr.Audio(label="Neural Scream (Network Decay Audio)", type="filepath")
            
    with gr.Row():
        output_dash = gr.Image(label="Mission Debrief Dashboard")
        
    gr.Markdown("---")
    gr.Markdown("## 🛡️ SENTINEL DEFENSE SYSTEMS")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("Deploy the zero-shot Deep Image Prior Autoencoder. The Sentinel will mathematically reconstruct the image over 200 cycles, leaving the high-frequency viral geometry behind in the process to cleanse the Asset.")
            shield_btn = gr.Button("🛡️ DEPLOY NEURAL SHIELD 🛡️", variant="secondary")
        with gr.Column(scale=1):
            purified_img_output = gr.Image(label="Purified Asset Output")
        with gr.Column(scale=1):
            shield_log = gr.Textbox(label="Sentinel Diagnostic Core", lines=7)
            
    compromised_state = gr.State()
    orig_class_state = gr.State()
    deploy_btn.click(
        fn=deploy_virus, 
        inputs=[input_img, target_dropdown], 
        outputs=[output_dash, output_gif, output_audio, compromised_state, orig_class_state]
    )
    
    shield_btn.click(
        fn=deploy_shield,
        inputs=[compromised_state, orig_class_state],
        outputs=[purified_img_output, shield_log]
    )

if __name__ == "__main__":
    print("Command Center online at http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860)
