
from torch import nn
from prettytable import PrettyTable


class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        self.__input_shape = None

    def forward(self):
        exit('The function `forward` needs to be implemented. This is an abstract class.')

    def get_model_name(self):
        exit('`get_model_name` needs to be implemented. This is an abstract class.')

    def count_parameters(self):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
        print(table)
        print(f"Total Trainable Named Parameters: {total_params}")
        for parameter in self.parameters():
            if not parameter.requires_grad:
                continue
            param = parameter.numel()
            total_params += param
        print(f"Total Trainable Parameterss: {total_params}")
        return total_params
