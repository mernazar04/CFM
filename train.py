
import torch
from torch.utils.data import DataLoader

from dataset import PDE_Dataset
from model import ConditionalFlowModel
from flow_matching import conditional_flow_matching_loss


# 1. Устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# 2. Dataset и DataLoader
train_dataset = PDE_Dataset("train")
val_dataset = PDE_Dataset("val")

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)


# 3. Модель
model = ConditionalFlowModel().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)


# 4. Обучение
epochs = 10

for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for x, c in train_loader:
        x = x.to(device)
        c = c.to(device)

        optimizer.zero_grad()

        loss = conditional_flow_matching_loss(model, x, c)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    # 5. Валидация
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x, c in val_loader:
            x = x.to(device)
            c = c.to(device)

            loss = conditional_flow_matching_loss(model, x, c)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Train Loss: {train_loss:.6f} "
        f"Val Loss: {val_loss:.6f}"
    )


# 6. Сохранение модели
torch.save(model.state_dict(), "cfm_model.pth")

print("Training completed!")
print("Model saved to cfm_model.pth")
