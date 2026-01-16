"""
Main entry file for training and testing.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import wandb

from src.model import RepNet, switch_model_to_deploy
from src.train import train, evaluate
from src.utils import check_equivalence, make_data_loaders


cfg = {
    "stem_ch": 64,
    "stages": [
        (64,  2, False),
        (128, 2, True),
        (256, 2, True),
        (512, 1, True),
    ]
}

NUM_CLASSES = 8 
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = (64, 128)
USE_AUGMENTATION = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(dataset_path, num_classes, epochs=50, lr=1e-3, weight_decay=1e-4, batch_size=(64, 128), augmentation=True):
    run = wandb.init(
        project="Week4-Oriol",
        entity="xavipba-universitat-aut-noma-de-barcelona",
        config={
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "architecture": cfg,
        },
    )

    train_loader, test_loader = make_data_loaders(dataset_path, augmentation=augmentation, batch_size=batch_size)

    model = RepNet(num_classes=num_classes, cfg=cfg).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state, best_val_acc, best_val_loss = train(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        device=DEVICE,
        optimizer=optimizer,
        criterion=criterion,
        epochs=epochs,
        wandb_run=run,
    )

    model.load_state_dict(best_state)
    model.eval()
    run.log({"best_val_loss": best_val_loss, "best_val_acc": best_val_acc})

    model_save_path = Path.cwd() / "Week4" / "models"
    model_save_path.mkdir(exist_ok=True, parents=True)
    torch.save(best_state, model_save_path / "repnet_train.pth")
    

    check_equivalence(model,device=DEVICE, input_shape=(8, 3, 224, 224))


    # Deploy conversion + save
    model.eval()
    switch_model_to_deploy(model)

    deploy_loss, deploy_acc = evaluate(model, criterion, test_loader, DEVICE)
    run.log({"deploy_loss": deploy_loss, "deploy_acc": deploy_acc})
    torch.save(model.state_dict(), model_save_path / "repnet_deploy.pth")

    run.finish()


if __name__ == "__main__":
    main(
        dataset_path="~/mcv/datasets/C3/2425/MIT_small_train_1",
        num_classes=NUM_CLASSES,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        batch_size=BATCH_SIZE,
        augmentation=USE_AUGMENTATION,
    )