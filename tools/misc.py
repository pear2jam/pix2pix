import torch


def get_rotate_matrix(alpha):
    theta = torch.FloatTensor([alpha])
    return torch.tensor([[torch.cos(alpha), -torch.sin(alpha), 0],
                         [torch.sin(alpha), torch.cos(alpha), 0]])