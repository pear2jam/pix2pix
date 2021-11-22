import torch  # для работы с данными
from torchvision.datasets import ImageFolder  # для загрузки данных
from torch.utils.data import DataLoader  # для деления данных на батчи при обучении

from tools import tools  # составные архитектуры
import tools.data_preparation as dp  # для обработки данных
import tools.losses as loss  # для функций ошибки
import torchvision.transforms as tt  # для обработки данных
import pickle  # для сохранения / загрузки модели
import time  # для оценки времени


def save(to_save, name="model.pkl"):
    with open(name, "wb") as f:
        pickle.dump(to_save, f)


def load(name="model.pkl"):
    with open(name, "rb") as f:
        return pickle.load(f)


data = ImageFolder('./dataset/val', transform=tt.Compose([
  tt.ToTensor()
]))

gen = tools.Generator()
dis = tools.Discriminator()

gen.train()

data = dp.split(data, turned_add=True, rotate_add=True, part=0.1, info=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

gen = dp.move_to(gen, device)
dis = dp.move_to(dis, device)

part = 0.1  # part of the data on which the model is training
epochs = 10
batch_size = 4

lr_gen = 2e-4  # learning rates for generator and discriminator
lr_dis = 2e-4

gen_per_dis = 1  # generation learning iterations per discriminator learning iterations

backup = True  # make backups
backup_rate = 10  # how much epochs for another backup

part_learn = int(len(data) * part)

if batch_size > part_learn:
    raise(Exception("Batch size for than data"))

x_loader = DataLoader(data[:part_learn], batch_size=batch_size, drop_last=True, shuffle=True)

X = next(iter(x_loader))[:, 0]
y = next(iter(x_loader))[:, 1]
enter = torch.cat((X, y), 1)
res = dis(enter)[0].permute(1, 2, 0).detach()
res = res.view(30, 30)

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
        dis_loss = loss.discriminator_loss(X, y, gen, dis)
        dis_loss.backward()
        dis_optim.step()

        gen_loss = 0
        for i in range(gen_per_dis):
            gen_optim.zero_grad()
            gen_loss = loss.generator_loss(X, y, gen, dis)
            gen_loss.backward()
            gen_optim.step()

        del data, X, y
        torch.cuda.empty_cache()

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
        gen_cpu = dp.move_to(gen, torch.device("cpu"))
        save(gen_cpu, "models/backup/backup_" + str(epoch // backup_rate) + ".pkl")
        dp.move_to(gen, device)

    print()
    print("finished with Gen Loss: ", float(gen_loss), " ,Dis Loss: ", float(dis_loss))

    pics = 0

save(dp.move_to(gen, torch.device("cpu")), "models/lgenerator.pkl")
