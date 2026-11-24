
import torch
import torch.nn.functional as F


def conditional_flow_matching_loss(model, x1, c):
    """
    x1: реальные PDE-траектории (B, 16, 128, 128)
    c: метки классов (B,)
    """

    batch_size = x1.shape[0]
    device = x1.device

    # 1. Случайный шум
    x0 = torch.randn_like(x1)

    # 2. Случайное время для каждого объекта
    t = torch.rand(batch_size, device=device)

    # 3. Промежуточное состояние
    t_view = t.view(batch_size, 1, 1, 1)
    xt = (1 - t_view) * x0 + t_view * x1

    # 4. Целевое векторное поле
    target = x1 - x0

    # 5. Предсказание модели
    prediction = model(xt, t, c)

    # 6. MSE loss
    loss = F.mse_loss(prediction, target)

    return loss


if __name__ == "__main__":
    from model import ConditionalFlowModel

    model = ConditionalFlowModel()

    x1 = torch.randn(4, 16, 128, 128)
    c = torch.tensor([1, 2, 1, 2])

    loss = conditional_flow_matching_loss(model, x1, c)

    print("CFM Loss:", loss.item())
