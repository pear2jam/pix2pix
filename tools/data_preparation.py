"""
data preparing module
"""
import torch
import torch.nn.functional as tf
import torchvision.transforms as tt
import warnings
from tools.misc import get_rotate_matrix


def flip(image):
    return torch.flip(image, [2])


def rotate(image, angle):
    rot_mat = get_rotate_matrix(angle)
    rot_mat = rot_mat.view(1, 2, 3)
    warnings.filterwarnings("ignore")
    grid = tf.affine_grid(rot_mat, [1, 3, 256, 256])
    rot_x = tf.grid_sample(image.view(1, 3, 256, 256), grid, padding_mode='reflection')

    return rot_x[0]


def crop(image):
    return tt.RandomResizedCrop(size=[256, 256], scale=[0.89, 1])(image)  # 0.89 ~ 256/289

def random_transformation(image):
    rand_angle = torch.rand(1, dtype=torch.float) - 0.5  # from -0.5 to 0.5 radians
    if torch.rand(1, dtype=torch.float) > 0.5:
        image = flip(image)
    image = rotate(image, rand_angle)
    image = crop(image)
    return image


def split(x, part=1, transform=True, info=True):
    """
    splitting and adding variety to the dataset
    :param x: input data with shape (len(x), 3, 256, 512)
    :param part: part of x to use (float from 0 to 1)
    :param transform: make random transformations of images
    :param info: print information about preparing process
    :raises ValueError: if part < 0 or part > 1
    :return: images data with shape (n*len(x), 2, 3, 256, 256)
    """
    if part < 0 or part > 1:
        raise ValueError("'part' parameter should be float value from 0 to 1")

    x_data = torch.FloatTensor(int(len(x)*part), 2, 3, 256, 256)

    part_len = int(len(x)*part / 20)  # отображение по 5 процентов
    total = -1  # текущие 5 процентов
    if info:
        print("preparing data %: ", end="")
    for i in range(int(len(x)*part)):
        if info and i // part_len > total:  # обновление процентов
            print(i // part_len * 5, end=" ")
            total = i // part_len

        x_temp = x[i][0].unfold(2, 256, 256).permute(2, 0, 1, 3)
        if transform:
            x_data[i][0] = random_transformation(x_temp[0])
            x_data[i][1] = random_transformation(x_temp[1])
        else:
            x_data[i][0] = x_temp[0]
            x_data[i][1] = x_temp[1]

    if info:
        print("100\nfinished")
        print("preparated images total:", int(len(x)*part))
    return x_data


def move_to(data, device):
    """
    moving data to device
    :param data: data to move
    :param device: device
    :return: moved data
    """
    if isinstance(data, (list, tuple)):
        return [move_to(x, device) for x in data]
    return data.to(device, non_blocking=True)
