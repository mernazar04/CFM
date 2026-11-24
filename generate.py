
import torch
import numpy as np

from model import ConditionalFlowModel


# Устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# Загружаем обученную модель
model = ConditionalFlowModel().to(device)
model.load_state_dict(
    torch.load("cfm_model.pth", map_location=device)
)
model.eval()


@torch.no_grad()
def generate(class_label, num_samples=4, steps=100):
    """
    class_label:
        1 = ACE
        2 = Wave-Gauss
    """

    # Начальный шум
    x = torch.randn(
        num_samples, 16, 128, 128,
        device=device
    )

    # Интегрирование методом Эйлера
    dt = 1.0 / steps

    for i in range(steps):
        t = torch.full(
            (num_samples,),
            i * dt,
            device=device
        )

        c = torch.full(
            (num_samples,),
            class_label,
            dtype=torch.long,
            device=device
        )

        velocity = model(x, t, c)

        x = x + dt * velocity

    return x.cpu().numpy()


if __name__ == "__main__":

    # Генерация ACE
    ace_samples = generate(class_label=1)

    # Генерация Wave-Gauss
    wg_samples = generate(class_label=2)

    np.save("generated_ace.npy", ace_samples)
    np.save("generated_wg.npy", wg_samples)

    print("ACE shape:", ace_samples.shape)
    print("Wave-Gauss shape:", wg_samples.shape)

    print("ACE range:",
          ace_samples.min(), ace_samples.max())

    print("Wave-Gauss range:",
          wg_samples.min(), wg_samples.max())

    print("Generation completed!")
