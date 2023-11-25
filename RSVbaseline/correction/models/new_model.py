from torch import nn
import torch

# Необходимо использовать LSTM модель для коррекции данных 
# Вариант 1 мы прям Идиоты и показываем нихуя V
# Вариант 2 мы дурачки и добавили 1 слой 
# Вариант 3 мы дебилы и додумались использовать LSTM модель
# Вариант 4 мы выше червяков и используем сезонную компоненту
# Вариант 4 мы выше жуков и используем сезонную компоненту и скользящую среднюю
class LSTMModel(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.channels = channels
        self.linear = nn.Linear(channels*210*280, 3)

    def forward(self, x):
        '''
        input: Season*BatchSize*Crit*Height*Width
        :param input:
        :return:
        '''
        print(*x.shape[:-3])
        output = self.linear(x.view(*x.shape[:-3], self.channels*210*280)) # view - переводит наш тензор в нужный вид x.shape[:-3] -
                                                                          # 4 32, с размерностью 3 * 210 * 280
        o_input = torch.split(x, 3, dim=-3)
        print(o_input)
        output = o_input[0].permute(3, 4, 0, 1, 2) + output
        return output.permute(2, 3, 4, 0, 1)