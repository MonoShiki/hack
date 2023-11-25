from torch import nn
import torch


class ConstantBias(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.channels = channels
        self.linear = nn.Linear(channels * 210 * 280, 3)

    def forward(self, x, seasonal_component):
        # Преобразуем x
        x = x.view(*x.shape[:-3], self.channels * 210 * 280)
        seasonal_component = np.load("/home/jupyter/datasphere/project/seasonal_components_wrf_mean.npy")

        # Применяем линейное преобразование
        output = self.linear(x)

        # Добавляем сезонную компоненту
        output = output + seasonal_component

        # Переформатируем обратно
        output = output.view(*output.shape[:-1], self.channels, 210, 280)
        output = output.permute(0, 1, 2, 4, 3)

        return output
