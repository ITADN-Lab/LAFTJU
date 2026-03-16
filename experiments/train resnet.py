import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from readData import read_dataset
from load_ImageNet import *
from ResNet import ResNet18, ResNet101, ResNet50
from ATJU import ATJU
from torch.optim import lr_scheduler
import time
import os



def save_txt(rewards, file_name, txt_name):
    rewards = np.atleast_1d(rewards).flatten()
    # 若目录不存在，则创建
    dir_path = os.path.join('./txt_save', file_name)
    os.makedirs(dir_path, exist_ok=True)
    # 完整文件路径
    file_path = os.path.join('./txt_save/'+file_name, f'{txt_name}.txt')
    # 追加写入；如果文件不存在 open(..., 'a') 会自动创建
    with open(file_path, 'a', encoding='utf-8') as f:
        # 逐行写入，读取更方便
        for r in rewards:
            f.write(f'{r}\n')


def main():

    now_time = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())
    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 读数据
    batch_size = 256
    # train_loader, valid_loader, test_loader = read_dataset(batch_size=batch_size, pic_path='dataset')
    train_loader, valid_loader = load_ImageNet(ImageNet_PATH=r"C:\Users\Administrator\Desktop\ATJU\experiments\imagenet",
                                               batch_size=batch_size, workers=6, pin_memory=True)
    # 加载模型(使用预处理模型，修改最后一层，固定之前的权重)
    n_class = 1000
    model = ResNet50(num_classes=n_class)
    """
    ResNet18网络的7x7降采样卷积和池化操作容易丢失一部分信息,
    所以在实验中将7x7的降采样层和最大池化层去掉,替换为一个3x3的降采样卷积,
    同时减小该卷积层的步长和填充大小
    """
    # model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
    # model.fc = torch.nn.Linear(512, n_class)  # 将最后的全连接层改掉
    model = model.to(device)
    # 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss().to(device)

    n_epochs = 150
    valid_loss_min = np.inf  # track change in validation loss
    accuracy = []

    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9,0.999), weight_decay=1e-4, eps=1e-8)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
    # 开始训练



    for epoch in tqdm(range(1, n_epochs+1)):

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        total_sample = 0
        right_sample = 0

        ###################
        # 训练集的模型 #
        ###################
        model.train() #作用是启用batch normalization和drop out
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            #（清除梯度）
            optimizer.zero_grad()
            # (正向传递：通过向模型传递输入来计算预测输出)
            output = model(data).to(device)  #（等价于output = model.forward(data).to(device) ）
            #（计算损失值）
            loss = criterion(output, target)
            # （反向传递：计算损失相对于模型参数的梯度）
            loss.backward()
            # 执行单个优化步骤（参数更新）
            optimizer.step()
            #（更新损失）
            train_loss += loss.item()*data.size(0)

        ######################
        # 验证集的模型#
        ######################

        model.eval()  # 验证模型
        with torch.no_grad():
            for data, target in valid_loader:
                data = data.to(device)
                target = target.to(device)
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data).to(device)
                # calculate the batch loss
                loss = criterion(output, target)
                # update average validation loss
                valid_loss += loss.item()*data.size(0)
                # (将输出概率转换为预测类)
                _, pred = torch.max(output, 1)
                # (将预测与真实标签进行比较)
                correct_tensor = pred.eq(target.data.view_as(pred))
                # correct = np.squeeze(correct_tensor.to(device).numpy())
                # total_sample += batch_size

                total_sample += target.size(0)
                right_sample += correct_tensor.sum().item()

            print("Accuracy:",100*right_sample/total_sample,"%")
            accuracy.append(right_sample/total_sample)

            # 计算平均损失
            train_loss = train_loss/len(train_loader.sampler)
            valid_loss = valid_loss/len(valid_loader.sampler)

            # 显示训练集与验证集的损失函数
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch, train_loss, valid_loss))

            # 如果验证集损失函数减少，就保存模型。
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
                torch.save(model.state_dict(), 'model_dict/resnet50.pt')
                valid_loss_min = valid_loss

            save_txt(valid_loss, file_name=f'imagenet_valid_loss', txt_name=f'SGD_{now_time}')
            save_txt(train_loss, file_name=f'imagenet_train_loss', txt_name=f'SGD_{now_time}')
            save_txt(right_sample / total_sample, file_name=f'imagenet_valid_acc', txt_name=f'SGD_{now_time}')

            scheduler.step()

            print(f"Current LR: {optimizer.param_groups[0]['lr']}")  # 单阶段optim_lr展示


if __name__ == '__main__':
    main()