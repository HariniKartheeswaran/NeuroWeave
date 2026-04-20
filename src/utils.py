import torch
from torchvision import transforms

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2470, 0.2435, 0.2616)
normalize = transforms.Normalize(cifar10_mean, cifar10_std)

def get_prediction(model, image):
    """
    Returns predicted class index and confidence score.
    """
    model.eval()

    with torch.no_grad():
        x = image.unsqueeze(0)
        x = normalize(x)
        output = model(x)  # Add batch dimension
        probs = torch.softmax(output, dim=1)

        conf, pred = torch.max(probs, dim=1)

    return pred.item(), conf.item()


def get_topk_predictions(model, image, k=3):
    """
    Returns top-k predictions and their confidence scores.
    Useful for deeper analysis.
    """
    model.eval()

    with torch.no_grad():
        output = model(image.unsqueeze(0))
        probs = torch.softmax(output, dim=1)

        topk_conf, topk_pred = torch.topk(probs, k)

    return topk_pred.squeeze().tolist(), topk_conf.squeeze().tolist()


def confidence_drop(original_conf, new_conf):
    """
    Calculates confidence drop after adversarial attack.
    """
    return original_conf - new_conf


def print_prediction_info(orig_pred, orig_conf, new_pred, new_conf):
    """
    Nicely prints prediction changes.
    """
    print("\n--- Prediction Analysis ---")
    print(f"Original Prediction: {orig_pred} | Confidence: {orig_conf:.4f}")
    print(f"Adversarial Prediction: {new_pred} | Confidence: {new_conf:.4f}")
    print(f"Confidence Drop: {confidence_drop(orig_conf, new_conf):.4f}")