import torch  # для работы с данными
from torchvision.datasets import ImageFolder  # для загрузки данных
from torch.utils.data import DataLoader  # для деления данных на батчи при обучении

from tools import tools  # составные архитектуры
import tools.data_preparation as dp  # для обработки данных
import tools.losses as loss  # для функций ошибки
import torchvision.transforms as tt  # для обработки данных
import time  # для оценки времени

from matplotlib import pyplot as plt


data = ImageFolder('./dataset/val', transform=tt.Compose([
  tt.ToTensor()
]))

gen = tools.Generator1()
dis = tools.Discriminator()

gen.train()

data = dp.split(data, turned_add=False, rotate_add=False, part=1, info=True)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

gen = dp.move_to(gen, device)
dis = dp.move_to(dis, device)

part = 1  # part of the data on which the model is training
epochs = 10
batch_size = 2

lr_gen = 2e-4  # learning rates for generator and discriminator
lr_dis = 2e-4


backup = True  # make backups
backup_rate = 1  # how much epochs for another backup

part_learn = int(len(data) * part)

if batch_size > part_learn:
    raise(Exception("Batch size more than data"))

x_loader = DataLoader(data[:part_learn], batch_size=batch_size, drop_last=True, shuffle=True)

gen_optim = torch.optim.Adam(gen.parameters(), lr=lr_gen)
dis_optim = torch.optim.Adam(dis.parameters(), lr=lr_dis)

pics = 0

gen_loss, dis_loss = 0, 0

print(">>>>>>>>>>>>>")
print(f"images: {part_learn}")

show_est_time = True
start_time = time.time()
for epoch in range(epochs):
    if not show_est_time:
        print(">> ", epoch, " | ", sep="", end="")
        print("from ", part_learn, "p : ", sep="", end="")
    for data in x_loader:
        data = dp.move_to(data, device)
        X = data[:, 0]
        y = data[:, 1]

        dis_optim.zero_grad()
        gen_optim.zero_grad()

        dis_loss = loss.discriminator_loss(X, y, gen, dis)
        dis_loss.backward()
        dis_optim.step()


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
            print(">> ", epoch, " | ", sep="", end="")
            print("from ", part_learn, "p : ", sep="", end="")

        pics += batch_size
        print(pics, end="p ")

    if backup and (epoch + 1) % backup_rate == 0:
        torch.save(gen, "models/backup/backup_" + str(epoch // backup_rate) + ".pth")

    print()
    print("finished with Gen Loss: ", float(gen_loss), " ,Dis Loss: ", float(dis_loss))

    pics = 0

torch.save(gen, 'models/lgen.pth')