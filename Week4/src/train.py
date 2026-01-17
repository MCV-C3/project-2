"""
This file contains utils to train the model.
"""
import torch



def mixup_batch(x, y, alpha):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    perm = torch.randperm(x.size(0), device=x.device)
    x2 = x[perm]
    y2 = y[perm]
    x_mix = lam * x + (1 - lam) * x2
    return x_mix, y, y2, lam

def train_one_epoch(model, criterion, dataloader, optimizer, device, mix_policy="none", mixup_alpha=0.2):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        if mix_policy == "mixup":
            inputs, y_a, y_b, lam = mixup_batch(inputs, targets, mixup_alpha)
            logits = model(inputs)
            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            # train accuracy against y_a
            correct = (logits.argmax(1) == y_a).sum().item()
        else:
            logits = model(inputs)
            loss = criterion(logits, targets)
            correct = (logits.argmax(1) == targets).sum().item()

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_correct += correct
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
    test_loader,
    device,
    optimizer,
    criterion,
    epochs,
    wandb_run=None,
    scheduler=None,
    mix_policy="none", 
    mixup_alpha=0.2):

    model.to(device)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs+1):
        tr_loss, tr_acc = train_one_epoch(
            model, 
            criterion, 
            train_loader, 
            optimizer, 
            device,
            mix_policy,
            mixup_alpha)
            
        val_loss, val_acc = evaluate(model, criterion, test_loader, device)

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[{epoch:03d}] "
            f"train: {tr_loss:.4f}/{tr_acc:.4f} | "
            f"test: {val_loss:.4f}/{val_acc:.4f}"
        )

        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch,
                "lr": current_lr,
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