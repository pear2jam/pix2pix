"""
Подготовка данных к обучению
"""
import torch
import torch.nn.functional as tf
import warnings
from tools.misc import get_rotate_matrix

def split(x, turned_add=False, rotate_add=True, max_angle=0.5, part=1, info=True):
    """
    splitting and adding variety to the dataset
    :param x: input data with shape (len(x), 3, 256, 512)
    :param turned_add: add flipped images to the end
    :param rotate_add: add rotated images to the end
    :param max_angle: maximum absolute angle of rotating for "rotate_add"
    :param part: part of x to use (float from 0 to 1)
    :param info: print information about preparing process

    :raises ValueError: if part < 0 or part > 1

    :return: images data with shape (n*len(x), 2, 3, 256, 256)
    """
    if part < 0 or part > 1:
        raise ValueError("'part' parameter should be float value from 0 to 1")

    size = 1
    if turned_add:
        size += 1
    if rotate_add:
        size += 1

    x_data = torch.FloatTensor(int(size*len(x)*part), 2, 3, 256, 256)

    part_len = int(len(x)*part / 20)  # отображение по 5 процентов
    total = -1  # текущие 5 процентов
    if info:
        print("preparing data %: ", end="")
    for i in range(int(len(x)*part)):
        if info and i // part_len > total:  # обновление процентов
            print(i // part_len * 5, end=" ")
            total = i // part_len

        x_temp = x[i][0].unfold(2, 256, 256).permute(2, 0, 1, 3)
        x_data[i][0] = x_temp[0]
        x_data[i][1] = x_temp[1]

        shift = 1  # коэфициэнт свдига по длине входного массива для дополнительных обработок

        if turned_add:
            x_data[int(len(x)*part)*shift + i][0] = torch.flip(x_temp[0], [2])
            x_data[int(len(x)*part)*shift + i][1] = torch.flip(x_temp[1], [2])
            shift += 1

        if rotate_add:
            rot_mat = get_rotate_matrix(torch.rand(1, dtype=torch.float)*2*max_angle - max_angle)
            rot_mat = rot_mat.view(1, 2, 3)
            warnings.filterwarnings("ignore")
            grid = tf.affine_grid(rot_mat, [1, 3, 256, 256])

            rot_x = tf.grid_sample(x_temp[0].view(1, 3, 256, 256), grid, padding_mode='reflection')
            x_data[int(len(x) * part) * shift + i][0] = rot_x[0]
            rot_x = tf.grid_sample(x_temp[1].resize(1, 3, 256, 256), grid, padding_mode='reflection')
            x_data[int(len(x) * part) * shift + i][1] = rot_x[0]

    if info:
        print("100\nfinished")
        print("images total:", int(size*len(x)*part))
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
