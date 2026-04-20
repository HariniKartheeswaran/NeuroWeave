import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

CIFAR_LABELS = {
    0: 'Airplane', 1: 'Automobile', 2: 'Bird', 3: 'Cat', 4: 'Deer',
    5: 'Dog', 6: 'Frog', 7: 'Horse', 8: 'Ship', 9: 'Truck'
}

def generate_mission_dashboard(orig_img, adv_img, orig_class, target_class, history, infected_coords, path):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 13))
    fig.patch.set_facecolor('#0d0d12') 
    
    orig_name = CIFAR_LABELS.get(orig_class, f"Class {orig_class}")
    target_name = CIFAR_LABELS.get(target_class, f"Class {target_class}")
    
    # 3 rows now: Images, Timeline, Diagnostic Report
    gs = gridspec.GridSpec(3, 3, height_ratios=[1.2, 1, 0.4], figure=fig, hspace=0.35)
    
    # ------------------ TOP ROW ------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(orig_img.permute(1, 2, 0).numpy())
    ax1.set_title(f"Target Acquired\n(AI thinks: {orig_name})", color='#00ff9d', pad=15, fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    ax2.set_facecolor('#0d0d12')
    image_shape = orig_img.shape
    heatmap = np.zeros((image_shape[1], image_shape[2]))
    for i, (x, y) in enumerate(infected_coords):
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[2]:
            heatmap[x, y] = len(infected_coords) - i
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    X, Y = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[2]))
    ax2.plot_surface(X, Y, heatmap.T, cmap='magma', linewidth=0, antialiased=True)
    ax2.set_title("Viral Insertion Vectors\n(3D Pixel Map)", color='#ff3366', pad=15, fontsize=16, fontweight='bold')
    ax2.view_init(elev=30, azim=45) 
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(adv_img.permute(1, 2, 0).numpy())
    ax3.set_title(f"System Compromised\n(AI fooled into: {target_name})", color='#ff3366', pad=15, fontsize=16, fontweight='bold')
    ax3.axis('off')
    
    # ------------------ MIDDLE ROW ------------------
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_facecolor('#111116')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['bottom'].set_color('#333344')
    ax4.spines['left'].set_color('#333344')
    
    ax4.set_ylim(-0.05, 1.05)
    ax4.plot(history, color='#ff3366', linewidth=3, marker='o', markersize=4)
    ax4.set_title("Mission Timeline: Neural Collapse", color='white', pad=15, fontsize=18, fontweight='bold')
    ax4.set_xlabel("Iterations (Virus Spreading across image)", color='#00ff9d', fontsize=14, labelpad=10)
    ax4.set_ylabel(f"AI Confidence in False Reality ({target_name})", color='#00ff9d', fontsize=14, labelpad=10)
    ax4.grid(True, color='#333344', linestyle='--', alpha=0.5)
    
    ax4.annotate('Virus Injected', xy=(0, history[0]), xytext=(30, 40), textcoords='offset points',
                 arrowprops=dict(facecolor='#00ff9d', shrink=0.05, width=2, headwidth=8),
                 color='#00ff9d', fontsize=14, fontweight='bold')
                 
    mid_point = len(history)//2
    ax4.annotate('Mutation Growth Phase', xy=(mid_point, history[mid_point]), xytext=(0, 60), textcoords='offset points', ha='center',
                 arrowprops=dict(facecolor='#cc00ff', shrink=0.05, width=2, headwidth=8),
                 color='#cc00ff', fontsize=14, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.5", fc="#111116", ec="#cc00ff", lw=2))
                 
    ax4.fill_between(range(len(history)), history, color='#ff3366', alpha=0.15)
    
    # ------------------ BOTTOM ROW ------------------
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    final_conf = history[-1] * 100
    report_text = (
        f"> INCIDENT LOG: Gradient-guided organic viral attack deployed. Total mutation cycles: {len(history)}.\n"
        f"> INITIAL STATE: Neural network positively identified asset as a '{orig_name.upper()}'.\n"
        f"> END STATE: Biological-style structural mutation compromised {len(infected_coords)} vectors.\n"
        f"> VERDICT: Severe semantic hallucination achieved. Network logic bypassed.\n"
        f"  Target securely misclassified as '{target_name.upper()}' with {final_conf:.1f}% confidence."
    )
    
    ax5.text(0.5, 0.5, report_text, ha='center', va='center', color='#00ff9d', fontsize=14, linespacing=1.8,
             family='monospace', bbox=dict(facecolor='#0a0a0a', edgecolor='#00ff9d', boxstyle='round,pad=1.5', lw=1.5))
    
    fig.suptitle("NEUROWEAVE: ADVERSARIAL DEBRIEF", color='white', fontsize=28, fontweight='black', y=0.96)
                
    plt.savefig(path, facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=150)
    plt.close()

def create_attack_gif(history_images, history_conf, path):
    """
    Creates an animated GIF with a highly requested Security Camera / Hacker HUD overlay.
    """
    frames = []
    
    # Use default PIL font 
    font = ImageFont.load_default()
        
    for i, img_tensor in enumerate(history_images):
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        img_pil = img_pil.resize((350, 350), Image.NEAREST)
        
        draw = ImageDraw.Draw(img_pil)
        conf = history_conf[i] * 100
        
        # Top Left HUD Box
        draw.rectangle([5, 5, 120, 20], fill=(10, 10, 15, 220), outline=(0, 255, 157))
        draw.text((10, 7), f"CYCLE: {i:03d}/{len(history_images)}", font=font, fill=(0, 255, 157))
        
        # Bottom HUD Warning Box
        draw.rectangle([5, 325, 345, 340], fill=(10, 10, 15, 220), outline=(255, 51, 102))
        draw.text((10, 327), f"SYSTEM OVERRIDE -> FALSE REALITY CONF: {conf:.1f}%", font=font, fill=(255, 51, 102))
        
        # Draw Scanlines
        for strip_y in range(0, 350, 10):
            draw.line([(0, strip_y), (350, strip_y)], fill=(0, 255, 157, 40), width=1)
            
        frames.append(img_pil)
        
    if frames:
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=100,  
            loop=0
        )