import os
import data
import json
import torch
import solver
import modeling
import numpy as np
from torch.nn.init import xavier_normal_, normal_
from utils.visualization import Visualization
from utils.miscellaneous import mkdir
from tensorboardX import SummaryWriter


def adjust_learning_rate(lr, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 20))
    print("learning rate %f in epoch %d" % (lr, epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def trainer(cfg):
    dataloader_train, dataset_size_train = data.make_dataloader(cfg, is_train=True)
    dataloader_test, dataset_size_test = data.make_dataloader(cfg, is_train=False)
    model = modeling.build(cfg)
    model.cuda()
    optimizer = solver.make_optimizer(cfg, model)
    vis_test = Visualization(cfg, dataset_size_test, is_train=False)
    score_writer = open("/home/xz/Projects/HiSA/experiments/logs/charades.txt", mode="w", encoding="utf-8")

    cfg.EPOCHS = 50
    for epoch in range(cfg.EPOCHS):
        model.train()
        for iteration, batch in enumerate(dataloader_train):
            index = batch[0]
            videoFeat = batch[1].cuda()
            videoFeat_lengths = batch[2].cuda()
            tokens = batch[3].cuda()
            tokens_lengths = batch[4].cuda()
            start = batch[5].cuda()
            end = batch[6].cuda()
            localiz = batch[7].cuda()
            frame_start = batch[13].cuda()
            frame_end = batch[14].cuda()

            loss, pred_start, pred_end = model(videoFeat, videoFeat_lengths, tokens, tokens_lengths, start, end,
                                               localiz, frame_start, frame_end)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        sumloss = 0
        sumsample = 0
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader_test):
                index = batch[0]
                videoFeat = batch[1].cuda()
                videoFeat_lengths = batch[2].cuda()
                tokens = batch[3].cuda()
                tokens_lengths = batch[4].cuda()
                start = batch[5].cuda()
                end = batch[6].cuda()
                localiz = batch[7].cuda()
                frame_start = batch[13].cuda()
                frame_end = batch[14].cuda()
                localiz_lengths = batch[8]
                time_starts = batch[9]
                time_ends = batch[10]
                factors = batch[11]
                fps = batch[12]

                duration = batch[15]
                vid_names = batch[16]
                loss, pred_start, pred_end = model(videoFeat, videoFeat_lengths, tokens, tokens_lengths, start,
                                                    end, localiz, frame_start, frame_end)
                sumloss += loss.item() * float(videoFeat.shape[0])
                sumsample += videoFeat.shape[0]
                vis_test.run(index, pred_start, pred_end, start, end, videoFeat_lengths, epoch, loss.detach(),
                             time_starts, time_ends, factors, fps, duration, vid_names)
        score_str = vis_test.plot(epoch)
        print("Test_Loss :{}".format(sumloss / sumsample))
        score_writer.write(score_str)
        score_writer.flush()
        model.train()
    score_writer.close()


def tester(cfg):
    print('testing')
    dataloader_test, dataset_size_test = data.make_dataloader(cfg, is_train=False)

    model = modeling.build(cfg)

    if cfg.TEST.MODEL.startswith('.'):
        load_path = cfg.TEST.MODEL.replace(".", os.path.realpath("."))
    else:
        load_path = cfg.TEST.MODEL

    model = torch.load(load_path)
    model.cuda()

    vis_test = Visualization(cfg, dataset_size_test, is_train=False)

    writer_path = os.path.join(cfg.VISUALIZATION_DIRECTORY, cfg.EXPERIMENT_NAME)
    writer = SummaryWriter(writer_path)

    total_iterations = 0
    total_iterations_val = 0

    model.eval()
    epoch = 1
    for iteration, batch in enumerate(dataloader_test):
        index = batch[0]
        videoFeat = batch[1].cuda()
        videoFeat_lengths = batch[2].cuda()
        tokens = batch[3].cuda()
        tokens_lengths = batch[4].cuda()
        start = batch[5].cuda()
        end = batch[6].cuda()
        localiz = batch[7].cuda()
        localiz_lengths = batch[8]
        time_starts = batch[9]
        time_ends = batch[10]
        factors = batch[11]
        fps = batch[12]
        frame_start = batch[13]
        frame_end =batch[14]

        loss, individual_loss, pred_start, pred_end, attention, atten_loss = model(videoFeat, videoFeat_lengths, tokens, tokens_lengths, start, end, localiz,frame_start,frame_end)
        aux = vis_test.run(index, pred_start, pred_end, start, end, videoFeat_lengths, epoch, loss.detach(), individual_loss, attention, atten_loss, time_starts, time_ends, factors, fps)
        total_iterations_val += 1
    a = vis_test.plot(epoch)


