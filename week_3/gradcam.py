import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import argparse
import yaml
import os

from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as T
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from models import ResNet50


def parse_args():
    """
    Parse command-line arguments for the Grad-CAM script.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments. The namespace contains:

        config_path : str
            Path to a YAML configuration file. Defaults to
            ``configs/config_gradcam.yaml``.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", type=str,
                   default=str(Path("configs") / "config_gradcam.yaml"))
    return p.parse_args()


def get_modules_by_name(model, layer_names):
    """
    Retrieve backbone submodules by their qualified (dotted) names.

    This is mainly used to convert layer names provided in a configuration file
    (e.g., YAML) into actual PyTorch module objects that can be passed to tools
    such as Grad-CAM.

    Parameters
    ----------
    model : torch.nn.Module
        Model instance that contains a ``backbone`` attribute. The backbone must
        implement ``named_modules()`` (standard for ``torch.nn.Module``).
    layer_names : Sequence[str]
        Iterable of qualified module names to retrieve (e.g., ``"layer4.2"``,
        ``"layer4.2.conv3"``).

    Returns
    -------
    list[torch.nn.Module]
        List of modules corresponding to ``layer_names``, in the same order.
    """
    name_to_module = dict(model.backbone.named_modules())
    missing = [n for n in layer_names if n not in name_to_module]
    if missing:
        examples = [k for k in name_to_module.keys() if k.startswith("layer4")][:15]
        raise KeyError(f"Unknown target layer(s): {missing}\nExamples: {examples}")
    return [name_to_module[n] for n in layer_names]


def get_gt_class_idx(image_path: str, class_to_idx: dict) -> int:
    """
    Infer the ground-truth class index from an image path.

    This function assumes the dataset follows an ImageFolder-style directory
    structure where the parent directory of an image corresponds to the class
    name, e.g.::

        .../test/highway/a866042.jpg  ->  class name "highway"

    Parameters
    ----------
    image_path : str
        Path to the image file. The class name is extracted from
        ``Path(image_path).parent.name``.
    class_to_idx : dict
        Mapping from class name (str) to class index (int), typically obtained
        from ``torchvision.datasets.ImageFolder(...).class_to_idx``.

    Returns
    -------
    int
        Ground-truth class index associated with the image.
    """
    class_name = Path(image_path).parent.name
    if class_name not in class_to_idx:
        raise KeyError(f"Class '{class_name}' not found in class_to_idx.")
    return class_to_idx[class_name]


def gradcam_three_views(model, image_path: str, output_path, true_label: int, target_layers, device=None):
    """
    Compute and save three Grad-CAM visualizations for a single image.

    This function generates Grad-CAM overlays for:

    1. The predicted class (top-1 logit).
    2. The ground-truth class provided via ``true_label``.
    3. The top competing class (2nd highest logit).

    A 3-panel figure (ground-truth / predicted / 2nd-highest) is saved to
    ``output_path``.

    Parameters
    ----------
    model : torch.nn.Module
        Model used for inference and Grad-CAM. Must implement
        ``model.extract_grad_cam(input_image, target_layer, targets)`` and accept
        an input tensor of shape ``(1, 3, 224, 224)`` after preprocessing.
    image_path : str
        Path to the input image file.
    output_path : str or pathlib.Path
        Destination filepath (including filename and extension) where the
        Grad-CAM figure will be saved. Parent directories are created if needed.
    true_label : int
        Ground-truth class index for the image.
    target_layers : list[torch.nn.Module]
        Target layer(s) used to compute Grad-CAM. For ResNet-50, common choices
        are ``[model.backbone.layer4[-1]]`` or ``[model.backbone.layer4[-1].conv3]``.
    device : torch.device or None, optional
        Device to run inference on. If ``None``, the device is inferred from
        the model parameters.
    """
    device = device or next(model.parameters()).device
    model.eval()

    # Deterministic model input preprocessing (recommended for Grad-CAM)
    input_tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    # Same spatial ops, but no normalization for overlay (must be [0,1])
    viz_tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])

    img_pil = Image.open(image_path).convert("RGB")
    x = input_tf(img_pil).unsqueeze(0).to(device)
    rgb = viz_tf(img_pil).permute(1, 2, 0).cpu().numpy()

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    top2 = torch.topk(logits[0], k=2).indices.tolist()
    pred_class, second_class = int(top2[0]), int(top2[1])

    cam_pred = model.extract_grad_cam(x, target_layers, [ClassifierOutputTarget(pred_class)])
    cam_true = model.extract_grad_cam(x, target_layers, [ClassifierOutputTarget(int(true_label))])
    cam_second = model.extract_grad_cam(x, target_layers, [ClassifierOutputTarget(second_class)])

    overlay_pred = show_cam_on_image(rgb, cam_pred, use_rgb=True)
    overlay_true = show_cam_on_image(rgb, cam_true, use_rgb=True)
    overlay_second = show_cam_on_image(rgb, cam_second, use_rgb=True)

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    true_name = idx_to_class[int(true_label)]
    pred_name = idx_to_class[int(pred_class)]
    second_name = idx_to_class[int(second_class)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    axes[0].imshow(overlay_true);   axes[0].set_title(f"Ground-truth\n{true_name} p={probs[true_label]:.3f}"); axes[0].axis("off")
    axes[1].imshow(overlay_pred);   axes[1].set_title(f"Predicted\n{pred_name} p={probs[pred_class]:.3f}"); axes[1].axis("off")
    axes[2].imshow(overlay_second); axes[2].set_title(f"2nd-highest\n{second_name} p={probs[second_class]:.3f}"); axes[2].axis("off")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    args = parse_args()
    config_path = Path(args.config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    gradcam_cfg = cfg["gradcam"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = paths["model_path"]
    output_path = paths['output_path']
    images_paths = paths["image_paths"]
    test_path = paths["test_path"]

    # Build mapping class_name -> index
    data_test = ImageFolder(test_path)  
    class_to_idx = data_test.class_to_idx

    # Load model
    model = ResNet50(num_classes=8, feature_extraction=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Read target layers from YAML
    layer_names = gradcam_cfg["target_layers"]   
    target_layers = get_modules_by_name(model, layer_names)

    for image_path in images_paths:
        gt_idx = get_gt_class_idx(image_path, class_to_idx)
        print(f"Image: {image_path} | GT: {Path(image_path).parent.name} ({gt_idx})")
        image_name = Path(image_path).stem
        model_name = Path(model_path).stem

        output_name = image_name + "_" + model_name + ".jpg"
        output_path = os.path.join(output_path,output_name)

        gradcam_three_views(model, image_path, output_path, true_label=gt_idx, target_layers=target_layers, device=device)
