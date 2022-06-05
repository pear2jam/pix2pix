import torch  # для работы с данными
from torchvision.datasets import ImageFolder  # для загрузки данных
from torch.utils.data import DataLoader  # для деления данных на батчи при обучении

from tools import tools  # составные архитектуры
import tools.data_preparation as dp  # для обработки данных
import tools.losses as loss  # для функций ошибки
import torchvision.transforms as tt  # для обработки данных
import time  # для оценки времени


# CONFIGURATION
# ----------------------
data_path = './dataset/val'  # path of dataset folder
transform = True  # random transformation of data

gen_type = 1  # 0 - big, 1 - light

load_models = False  # load learned models and continue learning
gen_path = './models/lgen.pth'  # generator path (for load_models = True)
dis_path = './models/backup/backup_d0.pth'  # discriminator path (for load_models = True)

part = 0.4  # part of the data on which the model is training [0; 1]
epochs = 20
batch_size = 16
max_steps = 2**32  # max steps of optimisation

lr_gen = 1.5e-4  # learning rates for generator and discriminator
lr_dis = 0.8e-4

backup = True  # make backups
backup_rate = 11  # how much epochs for another backup
backup_dis = False  # save discriminator in backup

show_est_time = True  # estimate learning time on start
# ----------------------

data = ImageFolder(data_path, transform=tt.Compose([
  tt.ToTensor()
]))


if not load_models:
    dis = tools.Discriminator()
    if gen_type == 0:
        gen = tools.Generator()
    else:
        gen = tools.Generator1()
else:
    gen = torch.load('models/lgen.pth', map_location='cpu')
    dis = torch.load('models/lgen.pth', map_location='cpu')

data = dp.split(data, transform=transform, info=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

gen = dp.move_to(gen, device)
dis = dp.move_to(dis, device)


part_learn = int(len(data) * part)

if batch_size > part_learn:
    raise(Exception("Batch size more than data"))

x_loader = DataLoader(data[:part_learn], batch_size=batch_size, drop_last=False, shuffle=True)

gen_optim = torch.optim.Adam(gen.parameters(), lr=lr_gen)
dis_optim = torch.optim.Adam(dis.parameters(), lr=lr_dis)

gen_loss, dis_loss = 0, 0

print(">>>>>>>>>>>>>")
print(f"images: {part_learn}")

steps = 0
start_time = time.time()
for epoch in range(epochs):
    if steps >= max_steps:
        break
    epoch_start_time = time.time()
    if not show_est_time:
        print(">> ", epoch, " | ", sep="", end="")
    for data in x_loader:
        if steps >= max_steps:
            print("Finished due to max steps limit")
            break

        data = dp.move_to(data, device)
        X = data[:, 0]
        y = data[:, 1]

        dis_optim.zero_grad()
        dis_loss = loss.discriminator_loss(X, y, gen, dis)
        dis_loss.backward()
        dis_optim.step()

        gen_optim.zero_grad()
        gen_loss = loss.generator_loss(X, y, gen, dis)
        gen_loss.backward()
        gen_optim.step()

        del data, X, y

        if show_est_time:
            show_est_time = False
            batch_time = time.time() - start_time
            batches = part_learn // batch_size
            epoch_time = batches * batch_time
            total_time = epoch_time * epochs

            print("Estimated time:")
            print(f"batch: {batch_time}s")
            print(f"epoch: {epoch_time}s")
            print(f"total: {total_time}s")
            print(">> ", epoch, " | ", sep="", end = "")

    if backup and (epoch + 1) % backup_rate == 0:
        torch.save(gen, "models/backup/backup_g" + str(epoch // backup_rate) + ".pth")
        if backup_dis:
            torch.save(dis, "models/backup/backup_d" + str(epoch // backup_rate) + ".pth")

    print(f'finished with Gen Loss: {float(gen_loss)} '
          f',Dis Loss: {float(dis_loss)} ({int(time.time()-epoch_start_time)}s)')

torch.save(gen, 'models/gen.pth')
torch.save(gen, 'models/dis.pth')
#torch
