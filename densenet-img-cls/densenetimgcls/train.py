from torchvision import datasets
from shutil import copyfile
from .densenet import DenseNet
from .utils import save_checkpoint, AverageMeter, get_transform, evaluate, logger
from .smt_fake import smt_fake_model
import os
import time
import fire
import torch


def train_epoch(model, loader, optimizer, epoch, epochs, print_freq=1):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()
    # Model on train mode
    model.train()
    end = time.time()
    batches = len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        # Create variables
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
        # compute output
        output = model(input)
        loss = torch.nn.functional.cross_entropy(output, target)
        # measure accuracy and record loss
        batch_size = target.size(0)
        _, pred = output.data.cpu().topk(1, dim=1)
        error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum().item() / batch_size, batch_size)
        losses.update(loss.item(), batch_size)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                f'Epoch: [{epoch + 1}/{epochs}]',
                f'Iter: [{batch_idx + 1}/{batches}]',
                f'Avg_Time_Batch/Avg_Time_Epoch: {batch_time.val:.3f}/{batch_time.avg:.3f}',
                f'Avg_Loss_Batch/Avg_Loss_Epoch: {losses.val:.4f}/{losses.avg:.4f}',
                f'Avg_Error_Batch/Avg_Error_Epoch: {error.val:.4f}/{error.avg:.4f}'
            ])
            logger.info(res)
    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg


def train(model, train_set, valid_set, save_path, epochs, batch_size,
          lr=0.001, wd=0.0001, momentum=0.9, random_seed=None, patience=10):
    # torch cuda random seed setting
    if random_seed is not None:
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                torch.cuda.manual_seed_all(random_seed)
            else:
                torch.cuda.manual_seed(random_seed)
        else:
            torch.manual_seed(random_seed)
    # DataLoader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               pin_memory=(torch.cuda.is_available()), num_workers=0)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                                               pin_memory=(torch.cuda.is_available()), num_workers=0)
    if torch.cuda.is_available():
        model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                nesterov=True, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * epochs, 0.75 * epochs],
                                                     gamma=0.1)
    with open(os.path.join(save_path, 'log.txt'), 'w') as f:
        f.write('Start training\n')
    best_error = 1
    counter = 0
    for epoch in range(epochs):
        scheduler.step()
        _, train_loss, train_error = train_epoch(model=model, loader=train_loader,
                                                 optimizer=optimizer, epoch=epoch, epochs=epochs)
        _, valid_loss, valid_error, _ = evaluate(model=model, loader=valid_loader)
        # Determine if model is the best
        if valid_error < best_error:
            is_best = True
            best_error = valid_error
        else:
            is_best = False
        state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict()
        # early stop
        if epoch == 0:
            last_epoch_valid_loss = valid_loss
        else:
            if valid_loss >= last_epoch_valid_loss:
                counter += 1
            else:
                counter = 0
            last_epoch_valid_loss = valid_loss
        logger.info(counter)
        early_stop = save_checkpoint({
            "epoch": epoch + 1,
            "state_dict": state_dict,
            "best_error": best_error,
            "best_accuracy": 1-best_error,
            "train_loss": train_loss,
            "train_error": train_error,
            "valid_loss": valid_loss,
            "valid_error": valid_error,
            "counter": counter
             }, is_best, save_path, patience)
        if early_stop:
            break


def entrance(model_path='', data_path='', save_path='saved_model',
             model_type='densenet201', pretrained=True, memory_efficient=False,
             num_classes=3, epochs=100, batch_size=16, learning_rate=0.001, random_seed=231, patience=2):
    train_transforms, test_transforms = get_transform()
    train_set = datasets.ImageFolder(data_path, transform=train_transforms)
    valid_set = datasets.ImageFolder(data_path, transform=test_transforms)
    indices = torch.randperm(len(train_set))
    valid_size = len(train_set) // 10
    train_indices = indices[:len(indices) - valid_size]
    valid_indices = indices[len(indices) - valid_size:]
    train_set = torch.utils.data.Subset(train_set, train_indices)
    valid_set = torch.utils.data.Subset(valid_set, valid_indices)
    model = DenseNet(model_type=model_type, model_path=model_path,
                     pretrained=pretrained, memory_efficient=memory_efficient, classes=num_classes)
    os.makedirs(save_path, exist_ok=True)
    train(model=model, train_set=train_set, valid_set=valid_set,
          save_path=save_path, epochs=epochs, batch_size=batch_size, lr=learning_rate,
          random_seed=random_seed, patience=patience)
    # workaround for smt
    smt_fake_model(save_path)
    # workaround for ds postprocess
    copyfile(os.path.join(data_path, 'index_to_label.json'), os.path.join(save_path, 'index_to_label.json'))
    logger.info('This experiment has been completed.')


if __name__ == '__main__':
    fire.Fire(entrance)
