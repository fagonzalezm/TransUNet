import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume

def inference(args, model,testloader, test_save_path=None):
    # logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):

        # h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size], case=case_name, need_zoom=1)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list_norm = metric_list / len(metric_list[0])
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list_norm[i-1][0], metric_list_norm[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return metric_list

def trainer_scian(args, model, snapshot_path):
    from datasets.dataset_scian import Scian_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/logTrain.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Scian_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    db_train2 = Scian_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train")                    
    db_val = Scian_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="validation")
    print("The length of train set is: {}".format(len(db_train))) #Aquí con RandomGenerator se aumentan los datos

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    trainloader2 = DataLoader(db_train2, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001) #Aquí se define el optimizador
    writer = SummaryWriter( '../tensorflow/' + "TransUnet" + args.vit_name+ '_skip' + str(args.n_skip) + '_vitPatchSize' + str(args.vit_patches_size)+ '_maxIt' +str(args.max_iterations)[0:2]+'k'+ '_maxEpo' +str(args.max_epochs) + '_batchSize' +str(args.batch_size)+ '_learningRate' + str(args.base_lr) + '_imageSize' +str(args.img_size)+ "/" + args.exp)
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9 # Comentar(?)
            # for param_group in optimizer.param_groups:               # Comentar(?)
            #     param_group['lr'] = lr_                              # Comentar(?)

            iter_num = iter_num + 1
            # writer.add_scalar('info/lr', lr_, iter_num)

            writer.add_scalar('train/total_loss', loss, iter_num)
            writer.add_scalar('train/loss_ce', loss_ce, iter_num)
            writer.add_scalar('train/loss_dice', loss_dice, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))
            logging.info('iteration %d : loss : %f' % (iter_num, loss.item(), ))


            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 1  # int(max_epoch/6)
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break
        metric_train = inference(args, model, trainloader2)
        metric_val = inference(args, model, valloader)
        writer.add_scalar('train/dice', metric_train[0], epoch_num)
        writer.add_scalar('train/hd95', metric_train[1], epoch_num)
        writer.add_scalar('train/jaccard', metric_train[2], epoch_num)
        writer.add_scalar('train/precision', metric_train[3], epoch_num)
        writer.add_scalar('train/recall', metric_train[4], epoch_num)
        writer.add_scalar('train/sensitivity', metric_train[5], epoch_num)
        writer.add_scalar('train/specificity', metric_train[6], epoch_num)
        writer.add_scalar('train/true_negative_rate', metric_train[7], epoch_num)
        writer.add_scalar('train/true_positive_rate', metric_train[8], epoch_num)
        writer.add_scalar('train/f1', metric_train[9], epoch_num)
        writer.add_scalar('train/accuracy', metric_train[10], epoch_num)

        writer.add_scalar('val/dice', metric_val[0], epoch_num)
        writer.add_scalar('val/hd95', metric_val[1], epoch_num)
        writer.add_scalar('val/jaccard', metric_val[2], epoch_num)
        writer.add_scalar('val/precision', metric_val[3], epoch_num)
        writer.add_scalar('val/recall', metric_val[4], epoch_num)
        writer.add_scalar('val/sensitivity', metric_val[5], epoch_num)
        writer.add_scalar('val/specificity', metric_val[6], epoch_num)
        writer.add_scalar('val/true_negative_rate', metric_val[7], epoch_num)
        writer.add_scalar('val/true_positive_rate', metric_val[8], epoch_num)
        writer.add_scalar('val/f1', metric_val[9], epoch_num)
        writer.add_scalar('val/accuracy', metric_val[10], epoch_num)



    writer.close()
    return "Training Finished!"