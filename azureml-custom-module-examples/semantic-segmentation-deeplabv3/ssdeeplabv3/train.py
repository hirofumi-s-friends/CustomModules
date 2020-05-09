import logging
import os
import tarfile
import time
import zipfile
from pathlib import Path
from tempfile import mkdtemp

import fire
import torch
from torchvision import transforms

from ssdeeplabv3.dataset import VOCSegmentation
from ssdeeplabv3.metric import SegmentationMetric
from ssdeeplabv3.model import SegmentationNet

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                    datefmt='%d-%M-%Y %H:%M:%S',
                    level=logging.INFO)

# This mapping is used for handling compressed files
COMPRESSED_EXT_TO_CLASS = {
    '.zip': zipfile.ZipFile,
    '.tar': tarfile.TarFile,
    '.gz': tarfile.TarFile,
    '.bz2': tarfile.TarFile,
}


def val(model, val_loader, use_cuda, metric):
    model.eval()
    metric.reset()
    total_iteration = len(val_loader)
    for i, (input, target) in enumerate(val_loader):
        start_time = time.time()
        if use_cuda and torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
        with torch.no_grad():
            output = model(input)
        metric.update(output['out'], target)
        pixAcc, mIoU = metric.get()
        logging.info(
            'Validation || current iteration / total iterations : [{} / {}], time : {:.2f}, '
            'epoch pixAcc: {:.3f}, epoch mIoU: {:.3f}'.format(
                i + 1, total_iteration,
                time.time() - start_time, pixAcc, mIoU))
    return pixAcc, mIoU


def train(model, criteria, train_loader, optimizer, use_cuda):
    model.train()
    total_iteration = len(train_loader)
    total_loss = 0
    for i, (input, target) in enumerate(train_loader):
        start_time = time.time()
        if use_cuda and torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
        output = model(input)
        loss = criteria(output['out'], target)
        total_loss += loss.data.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logging.info(
            'Train || current iteration / total iterations : [{} / {}], '
            'time : {:.2f}, loss : {:.4f}, '
            'epoch loss: {:.4f}'.format(i + 1, total_iteration,
                                        time.time() - start_time,
                                        loss.data.item(),
                                        total_loss / (i + 1)))
    return total_loss / total_iteration


def decompress_file(f):
    ext = Path(f).suffix
    if ext not in COMPRESSED_EXT_TO_CLASS:
        raise NotImplementedError(f"The extension {ext} is not supported now.")
    target_dir = mkdtemp()
    with COMPRESSED_EXT_TO_CLASS[ext](f) as zf:
        zf.extractall(target_dir)
    return target_dir


def entrance(model_type='deeplabv3_resnet101',
             model_path='ssdeeplabv3/saved_model',
             data_path='VOCtrainval_11-May-2012.tar',
             save_path='ssdeeplabv3/saved_model',
             pretrained=True,
             learning_rate=0.0001,
             epochs=50,
             batch_size=4,
             use_cuda=False):
    root_path = f'{decompress_file(data_path)}/VOCdevkit'
    print(f'decompressed file {root_path}')
    # image transform
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    train_dataset = VOCSegmentation(root=root_path,
                                    split='train',
                                    mode='train',
                                    transform=input_transform)
    val_dataset = VOCSegmentation(root=root_path,
                                  split='val',
                                  mode='val',
                                  transform=input_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=4,
                                               shuffle=True,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             num_workers=4,
                                             shuffle=False,
                                             pin_memory=True)
    criteria = torch.nn.CrossEntropyLoss(ignore_index=-1)
    metric = SegmentationMetric(train_dataset.NUM_CLASS)
    os.makedirs(save_path, exist_ok=True)
    model = SegmentationNet(model_type=model_type,
                            model_path=model_path,
                            pretrained=pretrained)
    if use_cuda and torch.cuda.is_available():
        model = model.cuda()
        criteria = criteria.cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate,
                                momentum=0.9,
                                weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=10,
                                                gamma=0.5)
    with open(os.path.join(save_path, 'log.csv'), 'w') as f:
        f.write('Train or Val,epoch,loss,pixel accuracy,mean IoU\n')
    mIoU_list = []
    for epoch in range(epochs):
        scheduler.step()
        logging.info('current epoch / total epochs : [{} / {}]'.format(
            epoch + 1, epochs))
        loss = train(model, criteria, train_loader, optimizer, use_cuda)
        pixAcc, mIoU = val(model, val_loader, use_cuda, metric)
        with open(os.path.join(save_path, 'log.csv'), 'a') as f:
            f.write('Train,{},{:.4f},,\n'.format(epoch, loss))
            f.write('Val,{},,{:.4f},{:.4f}\n'.format(epoch, pixAcc, mIoU))
        torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))
        mIoU_list.append(mIoU)
        if len(mIoU_list
               ) > 2 and mIoU_list[-1] < mIoU_list[-2] < mIoU_list[-3]:
            logging.info(
                'Validation loss did not descend any more. Stop here.')
            break

    logging.info('This experiment has been completed.')


if __name__ == '__main__':
    fire.Fire(entrance)
