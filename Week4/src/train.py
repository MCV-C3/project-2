"""
This file contains utils to train the model.
"""
import torch



def train_one_epoch(model, criterion, dataloader, optimizer, device):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0

    for batch_id, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()


        total_loss += loss.item() * inputs.size(0)
        total_correct += (logits.argmax(1) == targets).sum().item()
        total += inputs.size(0)

    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, criterion, dataloader, device):
    model.eval()

    total_loss, total_correct, total = 0.0, 0, 0

    for batch_id, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        logits = model(inputs)
        loss = criterion(logits, targets)

        total_loss += loss.item() * inputs.size(0)
        total_correct += (logits.argmax(1) == targets).sum().item()
        total += inputs.size(0)

    return total_loss / total, total_correct / total


def train(
    model,
    train_loader,
    val_loader,
    device,
    optimizer,
    criterion,
    epochs,
    wandb_run=None):

    model.to(device)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, criterion, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, criterion, val_loader, device)

        print(
            f"[{epoch:03d}] "
            f"train: {tr_loss:.4f}/{tr_acc:.4f} | "
            f"test: {val_loss:.4f}/{val_acc:.4f}"
        )

        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            })

        if val_acc > best_val_acc:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_acc = val_acc
            best_val_loss = val_loss

    return best_state, best_val_acc, best_val_loss