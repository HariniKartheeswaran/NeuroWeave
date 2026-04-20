import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from src.model import load_resnet, load_cnn
from src.attack import viral_infection_attack, fgsm_attack
from src.utils import get_prediction, normalize
from src.visualize import generate_mission_dashboard, create_attack_gif
from src.detector import detect_anomaly, advanced_denoise_detector
from src.audio import generate_neural_scream
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')
os.makedirs('results', exist_ok=True)

print("--- Initiating NeuroWeave Synesthesia Module ---\n")

print("Loading target model (CIFAR10 ResNet-20)...")
resnet = load_resnet()
print("Loading surrogate model (Simple CNN)...")
cnn = load_cnn()

transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

runs = 20
target_class = 5

transfer_success = 0

for i,(img,label) in enumerate(loader):
    if i >= runs:
        break

    img = img.squeeze(0)

    # Original Prediction
    orig_pred, orig_conf = get_prediction(resnet, img)

    # Biological Viral Infection Attack
    adv_img, adv_conf, coords, history, history_images = viral_infection_attack(resnet, img, target_class, iterations=100)
    new_pred, new_conf = get_prediction(resnet, adv_img)
    
    # Baseline comparison: FGSM Attack
    fgsm_img = fgsm_attack(resnet, img, target_class)
    fgsm_pred, fgsm_conf = get_prediction(resnet, fgsm_img)

    # Transfer test on CNN
    cnn_pred, _ = get_prediction(cnn, adv_img)
    if cnn_pred == target_class:
        transfer_success += 1

    # Detector mechanisms
    status1 = detect_anomaly(orig_conf, new_conf)
    caught, status2 = advanced_denoise_detector(resnet, adv_img, orig_pred)

    print(f"Sample {i+1} | Orig Pred: {orig_pred} ({orig_conf:.2f}) -> Adv Pred: {new_pred} ({new_conf:.2f}) [FGSM: {fgsm_pred}]")
    print(f"  +- Transfer to CNN: {cnn_pred} | Shield: {status2}")

    # Generate Synesthesia Arts for the first target
    if i == 0:
        print("\n--- Generating Mission Dashboard for Sample 1...")
        
        # Generates massive informative poster
        generate_mission_dashboard(img, adv_img, orig_pred, target_class, history, coords, "results/Mission_Debrief.png")
        
        if history_images:
            create_attack_gif(history_images, history, "results/viral_spread.gif")
            
        generate_neural_scream(history, "results/neural_scream.wav", duration=5.0)
        print("--- Unified Dashboard generated in results/Mission_Debrief.png!")
        print("--- Audio track generated in results/neural_scream.wav!\n")

print(f"\nSimulation Complete. System compromised {transfer_success / runs * 100:.2f}% of the time.")