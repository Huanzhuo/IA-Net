import os
import torch
import argparse
import datetime
import numpy as np
import torch.optim as optim
from models.resnet import resnet50, Model, resnet50_1d
from losses.loss import ContrastiveLoss_
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from dataset.industry_dataset_identify import IndustryDatasetIdentifyDiff, IndustryDatasetIdentify
from sklearn.model_selection import train_test_split
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp import autocast


nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H')
result_dir = './results/pump_slider_fan_valve/all_ids/{}'.format(nowTime)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


def save_model(model, optimizer, step, path):
    if len(os.path.dirname(path)) > 0 and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save({
        'model': model,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        # 'amp': amp.state_dict(),
    }, path)


def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        checkpoint['optimizer_state_dict']['param_groups'][0]['lr'] = 0.001
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    return step


def valiadate(model, criterion, val_loader):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        for i, (mix, mix_, status) in enumerate(val_loader):
            mix = mix.cuda()
            mix_ = mix_.cuda()
            status = status.cuda()
            features, features_ = model(mix, mix_)
            loss = criterion(features, features_, status)
            total_loss += loss.item()
    return total_loss


def get_files(path):
    wav_files = []
    for _, _, files in os.walk(path):
        for f in files:
            if f.split('.')[-1] == 'wav':
                wav_files.append(f)
    return wav_files


def get_ids(total_path):
    train_normal_ids_files = []
    val_normal_ids_files = []
    for dir in os.listdir(total_path):
        if dir.split('_')[0] != "id":
            continue
        normal_path = total_path + dir + '/normal'
        normal_files = get_files(normal_path)
        train_normals, val_normal = train_test_split(normal_files, random_state=42, test_size=0.2)
        train_normal_ids_files.append(train_normals)
        val_normal_ids_files.append(val_normal)
    return train_normal_ids_files, val_normal_ids_files


def main(args):
    print(args)

    # dataset and dataloader
    s1_path = '../MIMII/pump/'
    s2_path = '../MIMII/slider/'
    s3_path = '../MIMII/fan/'
    s4_path = '../MIMII/valve/'
    sources_path = [s1_path, s2_path, s3_path, s4_path]
    train_s1_normal_ids_files, val_s1_normal_ids_files = get_ids(s1_path)
    train_s2_normal_ids_files, val_s2_normal_ids_files = get_ids(s2_path)
    train_s3_normal_ids_files, val_s3_normal_ids_files = get_ids(s3_path)
    train_s4_normal_ids_files, val_s4_normal_ids_files = get_ids(s4_path)

    train_normals = [train_s1_normal_ids_files, train_s2_normal_ids_files, train_s3_normal_ids_files, train_s4_normal_ids_files]
    val_normals = [val_s1_normal_ids_files, val_s2_normal_ids_files, val_s3_normal_ids_files, val_s4_normal_ids_files]

    train_dataset = IndustryDatasetIdentifyDiff(sources_path, train_normals)
    val_dataset = IndustryDatasetIdentifyDiff(sources_path, val_normals, n=320)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # generate the model
    if args.load:
        print("Continuing training full model from checkpoint " + str(args.load_model))
        model = torch.load('./result_identify/pump_fan_valve/all_ids/model_best.pt')['model']
        # model = torch.load('./result/2020-08-20-01/model_best.pt')['model']
    else:
        model = resnet50_1d(n_spk=4)
        model = model.cuda()
    # generate optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=5e-4)
    clr = lr_scheduler.CosineAnnealingLR(optimizer, 5, eta_min=1e-5)
    # use mix precision training, may reduce the accuracy but increase the training speed
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    criterion = ContrastiveLoss_()
    scaler = GradScaler()
    # criterion = CosLoss()

    # Set up training state dict that will also be saved into checkpoints
    state = {"worse_epochs": 0,
             "epochs": 0,
             "best_loss": np.Inf,
             'step': 0}

    print('Start training...')
    for i in range(args.epochs):
        print("Training one epoch from iteration " + str(state["epochs"]))
        model.train()
        train_loss = 0.0
        for i, (mix, mix_, status) in enumerate(train_loader):
            mix = mix.cuda()
            mix_ = mix_.cuda()
            status = status.cuda()
            cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
            optimizer.zero_grad()
            with autocast():
                features, features_ = model(mix, mix_)
                loss = criterion(features, features_, status)
            train_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            state['step'] += 1

            if i % 20 == 0:
                print("{:4d}/{:4d} --- Loss: {:.6f} with learnig rate {:.6f}".format(i, len(train_dataset) // args.batch_size, loss.item(), cur_lr))

        clr.step()
        val_loss = valiadate(model, criterion, val_loader)
        train_loss = train_loss
        val_loss = val_loss

        print("Validation loss" + str(val_loss))

        # EARLY STOPPING CHECK
        checkpoint_path = args.model_path + str(state['epochs']) + '.pth'
        print("Saving model...")
        if val_loss >= state["best_loss"]:
            state["worse_epochs"] += 1
        else:
            print("MODEL IMPROVED ON VALIDATION SET!")
            state["worse_epochs"] = 0
            state["best_loss"] = val_loss
            state["best_checkpoint"] = checkpoint_path
            best_checkpoint_path = args.model_path + 'best.pt'
            best_state_dict_path = args.model_path + 'best_state_dict.pt'
            save_model(model, optimizer, state, best_checkpoint_path)
            torch.save(model.state_dict(), best_state_dict_path)
        print(state)
        state["epochs"] += 1
        if state["worse_epochs"] > args.hold_step:
            break
    last_model = args.model_path + 'last_model.pt'
    save_model(model, optimizer, state, last_model)
    print("Training finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='{}/model_'.format(result_dir),
                        help='Path to save model')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_size', type=float, default=2.0,
                        help='Output duration')
    parser.add_argument('--sr', type=int, default=16000,
                        help='Sampling rate')
    parser.add_argument('--channels', type=int, default=1,
                        help="Input channel, mono or sterno, default mono")
    parser.add_argument('--load_model', type=str, default='',
                        help="Path of hdf5 file")
    parser.add_argument("--load", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=200,
                        help="Epochs of half lr")
    parser.add_argument("--hold_step", type=int, default=20,
                        help="Epochs of hold step")
    args = parser.parse_args()
    main(args)
