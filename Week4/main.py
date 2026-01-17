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
from src.utils import check_equivalence, make_data_loaders, build_optimizer, build_scheduler
from src.configs.architectures import ARCHS



def main(dataset_path):
    NUM_CLASSES = 8 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    run = wandb.init(
        project="Week4-Oriol",
        entity="xavipba-universitat-aut-noma-de-barcelona",
        config={
            "arch": "base",
            "epochs": 40,
            "train_pipeline": "basic",  # plain | basic | basic_erasing | randaug | randaug_erasing
            "label_smoothing": 0.0,
            "mix_policy": "none",  # none | mixup
            "mixup_alpha": 0.2,    # used only if mix_policy == "mixup"
            "optimizer": "adamw",
            "scheduler": "cosine",
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "momentum": 0.9,
            "batch_train": 64,
            "batch_test": 128,
    }
        )

    wandb.run.name = (
        f"{wandb.config.arch}"
        f"_{wandb.config.train_pipeline}"
        f"_mix-{wandb.config.mix_policy}{wandb.config.mixup_alpha if wandb.config.mix_policy=='mixup' else ''}"
        f"_ls-{wandb.config.label_smoothing}"
        f"_{wandb.config.optimizer}"
        f"_{wandb.config.scheduler}"
        f"_lr-{wandb.config.lr}"
        f"_wd-{wandb.config.weight_decay}"
    )

    cfg = ARCHS[wandb.config.arch]
    train_loader, test_loader = make_data_loaders(
        dataset_path,
        batch_size=(wandb.config.batch_train, wandb.config.batch_test),
        train_pipeline=wandb.config.train_pipeline,
    )

    model = RepNet(num_classes=NUM_CLASSES, cfg=cfg).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=wandb.config.label_smoothing)
    optimizer = build_optimizer(
        wandb.config.optimizer,
        model.parameters(),
        lr=wandb.config.lr,
        weight_decay=wandb.config.weight_decay,
        momentum=wandb.config.momentum,
    )
    
    scheduler = build_scheduler(wandb.config.scheduler, optimizer, epochs=wandb.config.epochs)

    best_state, best_val_acc, best_val_loss = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=DEVICE,
        optimizer=optimizer,
        criterion=criterion,
        epochs=wandb.config.epochs,
        wandb_run=run,
        scheduler=scheduler,
        mix_policy=wandb.config.mix_policy,
        mixup_alpha=wandb.config.mixup_alpha,
    )

    model.load_state_dict(best_state)
    model = model.to(DEVICE)
    model.eval()
    run.log({"best_val_loss": best_val_loss, "best_val_acc": best_val_acc})

    model_save_path = Path.cwd() / "Week4" / "models"
    model_save_path.mkdir(exist_ok=True, parents=True)
    torch.save(best_state, model_save_path / "repnet_train.pth")
    

    check_equivalence(model, device=DEVICE, input_shape=(8, 3, 224, 224))


    # Deploy conversion + save
    model.eval()
    switch_model_to_deploy(model)

    deploy_loss, deploy_acc = evaluate(model, criterion, test_loader, DEVICE)
    run.log({"deploy_loss": deploy_loss, "deploy_acc": deploy_acc})
    torch.save(model.state_dict(), model_save_path / "repnet_deploy.pth")

    run.finish()


if __name__ == "__main__":
    main(dataset_path="~/mcv/datasets/C3/2425/MIT_small_train_1")