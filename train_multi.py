import os
import torch
import argparse
import datetime
import numpy as np
from configs import cfg
import torch.optim as optim
from models.resnet import resnet50, Model, resnet50_1d
from losses.loss import ContrastiveLoss_
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from dataset.industry_dataset import IndustryDataset, IndusDatasetBoth, IndusDatasetMulti
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp import autocast



nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H')
result_dir = './result/pump_fan_valve/margin_3_0.95_testset/{}'.format(nowTime)
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


def train_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=bool, default=True,
                        help='use gpu, default True')
    parser.add_argument('--model_path', type=str, default='{}/model_'.format(result_dir),
                        help='Path to save model')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_size', type=float, default=2.0,
                        help='Output duration')
    parser.add_argument('--sr', type=int, default=16000,
                        help='Sampling rate')
    parser.add_argument('--loss', type=str, default="L1",
                        help="L1 or L2")
    parser.add_argument('--channels', type=int, default=1,
                        help="Input channel, mono or sterno, default mono")
    parser.add_argument('--h5_dir', type=str, default='../WaveUNet/H5/',
                        help="Path of hdf5 file")
    parser.add_argument('--load_model', type=str, default='../y-net-result/2020-07-23-18/model_best.pth',
                        help="Path of hdf5 file")
    parser.add_argument("--load", type=bool, default=True)
    parser.add_argument("--epochs", type=int, default=200,
                        help="Epochs of half lr")
    parser.add_argument("--hold_step", type=int, default=30,
                        help="Epochs of hold step")
    parser.add_argument("--example_freq", type=int, default=200,
                        help="write an audio summary into Tensorboard logs")
    parser.add_argument("--reconst_loss", type=bool, default=False,
                        help='use reconstruction loss or not')
    return parser.parse_args()


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


def main():
    args = train_cfg()
    print(args)

    # generate the summarywriter, dataset and dataloader
    writer = SummaryWriter(result_dir)

    pump_normal_path = '../MIMII/pump/id_00/normal'
    pump_abnormal_path = '../MIMII/pump/id_00/abnormal'
    slider_normal_path = '../MIMII/slider/id_00/normal'
    slider_abnormal_path = '../MIMII/slider/id_00/abnormal'
    fan_normal_path = '../MIMII/fan/id_00/normal'
    fan_abnormal_path = '../MIMII/fan/id_00/abnormal'
    valve_normal_path = '../MIMII/valve/id_00/normal'
    valve_abnormal_path = '../MIMII/valve/id_00/abnormal'

    pump_normal_files = get_files(pump_normal_path)
    pump_abnormal_files = get_files(pump_abnormal_path)
    slider_normal_files = get_files(slider_normal_path)
    slider_abnormal_files = get_files(slider_abnormal_path)
    fan_normal_files = get_files(fan_normal_path)
    fan_abnormal_files = get_files(fan_abnormal_path)
    valve_normal_files = get_files(valve_normal_path)
    valve_abnormal_files = get_files(valve_abnormal_path)

    # train_pump_normal, val_pump_normal = train_test_split(pump_normal_files, random_state=42, test_size=0.2)
    train_slider_normal, val_slider_normal = train_test_split(slider_normal_files, random_state=42, test_size=0.2)
    train_fan_normal, val_fan_normal = train_test_split(fan_normal_files, random_state=42, test_size=0.2)
    train_valve_normal, val_valve_normal = train_test_split(valve_normal_files, random_state=42, test_size=0.2)

    # train_pump_abnormal, val_pump_abnormal = train_test_split(pump_abnormal_files, random_state=42, test_size=0.95)
    train_slider_abnormal, val_slider_abnormal = train_test_split(slider_abnormal_files, random_state=42, test_size=0.95)
    train_fan_abnormal, val_fan_abnormal = train_test_split(fan_abnormal_files, random_state=42, test_size=0.95)
    train_valve_abnormal, val_valve_abnormal = train_test_split(valve_abnormal_files, random_state=42, test_size=0.95)

    # pump_path = '../MIMII/pump/id_00'
    slider_path = '../MIMII/slider/id_00'
    fan_path = '../MIMII/fan/id_00'
    valve_path = '../MIMII/valve/id_00'

    sources_path = [slider_path, fan_path, valve_path]
    train_normal_ids = [train_slider_normal, train_fan_normal, train_valve_normal]
    train_abnormal_ids = [train_slider_abnormal, train_fan_abnormal, train_valve_abnormal]
    val_normal_ids = [val_slider_normal, val_fan_normal, val_valve_normal]
    val_abnormal_ids = [val_slider_abnormal, val_fan_abnormal, val_valve_abnormal]
    '''
    sources_path = [pump_path, slider_path, fan_path, valve_path]
    train_normal_ids = [train_pump_normal, train_slider_normal, train_fan_normal, train_valve_normal]
    train_abnormal_ids = [train_pump_abnormal, train_slider_abnormal, train_fan_abnormal, train_valve_abnormal]
    val_normal_ids = [val_pump_normal, val_slider_normal, val_fan_normal, val_valve_normal]
    val_abnormal_ids = [val_pump_abnormal, val_slider_abnormal, val_fan_abnormal, val_valve_abnormal]
    '''

    train_dataset = IndusDatasetMulti(sources_path, train_normal_ids, train_abnormal_ids)
    val_dataset = IndusDatasetMulti(sources_path, val_normal_ids, val_abnormal_ids, n=320)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # generate the model
    args.load = False
    if args.load:
        print("Continuing training full model from checkpoint " + str(args.load_model))
        model = torch.load('./result/valve_slider/margin_3_0.95_testset/model_best.pt')['model']
        # model = torch.load('./result/2020-08-20-01/model_best.pt')['model']
    else:
        model = resnet50_1d(3)
        model = model.cuda()
    # generate optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=5e-4)
    clr = lr_scheduler.CosineAnnealingLR(optimizer, 5, eta_min=1e-5)
    scaler = GradScaler()
    # use mix precision training, may reduce the accuracy but increase the training speed
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    criterion = ContrastiveLoss_()
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
            writer.add_scalar("learning_rate", cur_lr, state['step'])
            optimizer.zero_grad()
            with autocast():
                features, features_ = model(mix, mix_)
                loss = criterion(features, features_, status)
            train_loss += loss.item()
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # clr.step()
            state['step'] += 1

            if i % 20 == 0:
                print("{:4d}/{:4d} --- Loss: {:.6f} with learnig rate {:.6f}".format(i, len(train_dataset) // args.batch_size, loss.item(), cur_lr))

        clr.step()
        val_loss = valiadate(model, criterion, val_loader)
        train_loss = train_loss
        val_loss = val_loss

        print("Validation loss" + str(val_loss))
        writer.add_scalar("train_loss", train_loss, state['epochs'])
        writer.add_scalar("val_loss", val_loss, state['epochs'])

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
        if state["worse_epochs"] > args.hold_step and i > 80:
            break
    last_model = args.model_path + 'last_model.pt'
    save_model(model, optimizer, state, last_model)
    print("Training finished")
    writer.close()


if __name__ == '__main__':
    main()
