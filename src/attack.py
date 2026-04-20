import numpy as np
import torch
import torch.nn.functional as F
from src.utils import normalize

def perturb_image(params, image):
    img = image.clone()
    coords = []
    for p in params:
        x, y, r, g, b = map(int, p)
        img[:, x, y] = torch.tensor([r / 255.0, g / 255.0, b / 255.0])
        coords.append((x, y))
    return img, coords

def targeted_attack(model, image, target_class, iterations=50):
    best_img = image.clone()
    best_conf = 0
    best_coords = []
    history = []
    history_images = []

    for i in range(iterations):
        xs = np.random.randint(0, image.shape[1], 5)
        ys = np.random.randint(0, image.shape[2], 5)
        rgb = np.random.randint(0, 256, (5, 3))

        params = [(xs[j], ys[j], rgb[j][0], rgb[j][1], rgb[j][2]) for j in range(5)]
        perturbed, coords = perturb_image(params, image)

        output = model(normalize(perturbed.unsqueeze(0)))
        prob = torch.softmax(output, dim=1)[0][target_class].item()

        history.append(prob)
        if prob > best_conf:
            best_conf = prob
            best_img = perturbed
            best_coords = coords
            
        history_images.append(best_img.clone())

    return best_img, best_conf, best_coords, history, history_images

def viral_infection_attack(model, image, target_class, iterations=100, infection_rate=5):
    """
    Biological Synesthesia Attack (Deep PGD Mask)
    Spreads like cellular automata, but continually optimizes ALL infected pixels 
    using PGD-like gradients to guarantee absolute network collapse.
    """
    current_img = image.clone()
    
    infected = set()
    seed_x = np.random.randint(0, image.shape[1])
    seed_y = np.random.randint(0, image.shape[2])
    infected.add((seed_x, seed_y))
    
    history = [] 
    history_images = []
    target_tensor = torch.tensor([target_class], dtype=torch.long)
    epsilon = 0.05  # Slow, highly controlled mutation simmering
    
    for i in range(iterations):
        # Calculate structural vulnerability gradient for the WHOLE image
        grad_img = current_img.clone().unsqueeze(0).requires_grad_(True)
        output = model(normalize(grad_img))
        
        # Minimize loss of target class to maximize target probability
        loss = F.nll_loss(F.log_softmax(output, dim=1), target_tensor)
        model.zero_grad()
        loss.backward()
        
        # Get gradient directional signs
        data_grad = grad_img.grad.data.squeeze(0).sign()
        
        # Find all vulnerable cells at the infection frontier
        frontier = []
        for (x,y) in list(infected):
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[2]:
                        if (nx, ny) not in infected:
                            frontier.append((nx, ny))
        
        # Organically expand the viral footprint
        if frontier:
            np.random.shuffle(frontier)
            viral_cells = frontier[:infection_rate]
            for nx, ny in viral_cells:
                infected.add((nx, ny))
                
        # Mutate ALL infected cells continuously! The infection deepens as it ages.
        mutated_img = current_img.clone()
        for x, y in list(infected):
            # Guided mutation: step towards target class in pixel space
            mutated_img[:, x, y] = torch.clamp(mutated_img[:, x, y] - epsilon * data_grad[:, x, y], 0.0, 1.0)
            
        current_img = mutated_img
        
        # Evaluate current strength of the virus
        with torch.no_grad():
            output_eval = model(normalize(current_img.unsqueeze(0)))
            prob = torch.softmax(output_eval, dim=1)[0][target_class].item()
            
        history.append(prob)
        history_images.append(current_img.clone())
        
    return current_img, prob, list(infected), history, history_images

def fgsm_attack(model, image, target_class, epsilon=0.03):
    img = image.clone().unsqueeze(0).requires_grad_(True)
    output = model(normalize(img))
    target = torch.tensor([target_class], dtype=torch.long)
    loss = F.nll_loss(F.log_softmax(output, dim=1), target)
    
    model.zero_grad()
    loss.backward()
    
    data_grad = img.grad.data
    sign_data_grad = data_grad.sign()
    perturbed_image = img - epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image.squeeze(0).detach()