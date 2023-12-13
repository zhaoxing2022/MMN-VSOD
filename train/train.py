import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import torch.multiprocessing as mp
import argparse
import pytorch_iou
import pytorch_ssim
from dataloader import *
from model import MMN_VSOD
import platform

parser = argparse.ArgumentParser(description='VSOD Training')

# Pretrained with ISOD tasks.
parser.add_argument('--image_epochs', default=16, type=int, help="pretrained on image dataset")
# If you have 3 GPUs, the real batch size is (3*batch_size_image).
parser.add_argument('--batch_size_image', default=18, type=int, help="batch_size is only on a gpu")

#Trained with VSOD tasks.
parser.add_argument('--video_epochs', default=12, type=int)
# If you have 3 GPUs, the real batch size is (3*batch_size_video).
parser.add_argument('--batch_size_video', default=4, type=int, help="batch_size is only on a gpu")

parser.add_argument('--load_model_path', default="", type=str, help="if you wish to resume training after an interruption, set the model path")

parser.add_argument('--train_size', default=[288, 288], help="[height, width]")
parser.add_argument('--train_image_dir', default="/path/to/dataset/train", type=str)
parser.add_argument('--train_image_datasets', default=[ 'DUTS','DAVIS', 'DAVSOD'])
parser.add_argument('--train_video_dir', default="/path/to/dataset/train", type=str)
parser.add_argument('--train_video_datasets', default=['DAVIS', 'DAVSOD'])
parser.add_argument('--video_time_clips', default=4, type=int)
parser.add_argument('--time_interval', default=1, type=int)
parser.add_argument('--run_type', default="train_model", type=str)

# difference between windows and ubutun
if platform.system().lower() == 'windows':
    backend = 'gloo'
    num_workers = 0
else:
    backend = "nccl"
    num_workers = 6
def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt.item()

def main():
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))


def main_worker(local_rank, nprocs, args):
    args.local_rank = local_rank
    dist.init_process_group(backend=backend, init_method='tcp://127.0.0.1:24568', world_size=args.nprocs,
                            rank=local_rank)

    model = MMN_VSOD()
    if args.load_model_path is not None and args.load_model_path != "":
        checkpoint = torch.load(args.load_model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint, strict=False)
        print(f"load checkpoint {args.load_model_path} success")

    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    criterion = torch.nn.BCELoss().cuda(local_rank)
    ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True).cuda(local_rank)
    iou_loss = pytorch_iou.IOU(size_average=True).cuda(local_rank)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    resnet_params, other_params = [], []
    for name, param in model.named_parameters():
        if ('encoder' in name) and ('reduce' not in name):
            resnet_params.append(param)
        else:
            other_params.append(param)
    optimizer_image = torch.optim.Adam([{'params': resnet_params, 'lr': 1e-5}, {'params': other_params}], lr=1e-4)
    optimizer_video = torch.optim.Adam([{'params': resnet_params, 'lr': 1e-5}, {'params': other_params}], lr=1e-4)
    scheduler_image = torch.optim.lr_scheduler.StepLR(optimizer_image, step_size=8, gamma=0.5)
    scheduler_video = torch.optim.lr_scheduler.StepLR(optimizer_video, step_size=8, gamma=0.5)

    image_train_dataset = ImageTrainDataset(train_image_dir=args.train_image_dir, train_image_datasets=args.train_image_datasets,train_size=args.train_size)
    image_train_sampler = DistributedSampler(image_train_dataset)
    image_train_loader = torch.utils.data.DataLoader(image_train_dataset,batch_size=args.batch_size_image , pin_memory=True,drop_last=True,
                                                     sampler=image_train_sampler, num_workers=num_workers)

    video_train_dataset = VideoTrainDataset(train_video_dir=args.train_video_dir, train_video_datasets=args.train_video_datasets,train_size=args.train_size, time_interval=args.time_interval,video_time_clips = args.video_time_clips)
    video_train_sampler = DistributedSampler(video_train_dataset)
    video_train_loader = torch.utils.data.DataLoader(video_train_dataset, batch_size=args.batch_size_video, pin_memory=True,
                                                     drop_last=True, sampler=video_train_sampler, num_workers=num_workers)
    if local_rank == 0:
        print("stage1: ISOD task")
    for epoch in range(args.image_epochs):
        image_train_sampler.set_epoch(epoch)
        train_with_image(image_train_loader, model, criterion, ssim_loss, iou_loss, optimizer_image, epoch, local_rank, args)
        scheduler_image.step()
    if local_rank == 0:
        print("stage2: VSOD task")
    for epoch in range(args.video_epochs):
        video_train_sampler.set_epoch(epoch)
        train_with_video(video_train_loader, model, criterion, ssim_loss, iou_loss, optimizer_video, epoch,local_rank, args)
        scheduler_video.step()
        dist.barrier()
        if local_rank == 0 and (epoch > args.video_epochs//2):
            if not os.path.exists(os.path.join(os.getcwd(), args.run_type)):
                os.makedirs(os.path.join(os.getcwd(), args.run_type))
            torch.save({'model_state_dict': model.module.state_dict()}, f"{args.run_type}/epoch{epoch}.pt")
        dist.barrier()


def train_with_image(image_train_loader, model, criterion, ssim_loss, iou_loss, optimizer_image, epoch, local_rank, args):
    model.train()
    loss_mean_bce = 0.
    loss_mean_ssim = 0.
    loss_mean_iou = 0.
    if local_rank == 0:
        pbar = tqdm(total=len(image_train_loader))
    for i, (inputs, labels) in enumerate(image_train_loader):
        inputs = inputs.to(local_rank, non_blocking=True).unsqueeze(1)
        labels = labels.to(local_rank, non_blocking=True)

        optimizer_image.zero_grad()
        preds = model(inputs)
        preds = preds.squeeze(1)
        loss = criterion(preds, labels)
        loss_ssim = 1 - ssim_loss(preds, labels)
        loss_iou = iou_loss(preds, labels)

        total = loss + loss_ssim + loss_iou
        torch.distributed.barrier()
        total.backward()
        optimizer_image.step()

        reduce_loss = reduce_mean(loss, args.nprocs)
        reduce_loss_ssim = reduce_mean(loss_ssim, args.nprocs)
        reduce_loss_iou = reduce_mean(loss_iou, args.nprocs)
        if local_rank == 0:
            pbar.update(1)
            loss_mean_bce += reduce_loss
            loss_mean_ssim += reduce_loss_ssim
            loss_mean_iou += reduce_loss_iou
    if local_rank == 0:
        pbar.close()
        loss_mean_bce /= (i + 1)
        loss_mean_ssim /= (i + 1)
        loss_mean_iou /= (i + 1)
        with open(f'log_{args.run_type}.txt', 'a') as file_object:
            information = "loss-image-epoch{}:BCE:{}---SSIM:{}---iou:{}---lr:{}\n".format(epoch, loss_mean_bce, loss_mean_ssim, loss_mean_iou,str([optimizer_image.param_groups[0]['lr'],optimizer_image.param_groups[1]['lr']]))
            print(information)
            file_object.write(information)
def train_with_video(video_train_loader, model, criterion, ssim_loss, iou_loss, optimizer_video, epoch, local_rank, args):
    model.train()
    loss_mean_bce = 0.
    loss_mean_ssim = 0.
    loss_mean_iou = 0.
    loss_mean_motion =0.
    length = len(video_train_loader)
    if local_rank == 0:
        pbar = tqdm(total=length)
    for i, (inputs, labels) in enumerate(video_train_loader):
        inputs = inputs.to(local_rank, non_blocking=True)
        labels = labels.to(local_rank, non_blocking=True)
        motion = torch.zeros_like(labels[:, 0])
        motion[(labels[:, 1] - labels[:, 2]) != 0] = 1
        optimizer_video.zero_grad()
        outputs, move_outputs = model(inputs)
        loss = criterion(outputs, labels)
        motion_loss = criterion(move_outputs, motion)

        loss_ssim1 = 1 - ssim_loss(outputs[:, 0], labels[:, 0])
        loss_ssim2 = 1 - ssim_loss(outputs[:, 1], labels[:, 1])
        loss_ssim3 = 1 - ssim_loss(outputs[:, 2], labels[:, 2])
        loss_ssim4 = 1 - ssim_loss(outputs[:, 3], labels[:, 3])
        loss_ssim = (loss_ssim1 + loss_ssim2 + loss_ssim3 +loss_ssim4) / 4

        loss_iou1 = iou_loss(outputs[:, 0], labels[:, 0])
        loss_iou2 = iou_loss(outputs[:, 1], labels[:, 1])
        loss_iou3 = iou_loss(outputs[:, 2], labels[:, 2])
        loss_iou4 = iou_loss(outputs[:, 3], labels[:, 3])
        loss_iou = (loss_iou1 + loss_iou2 + loss_iou3 + loss_iou4) / 4

        total = loss + loss_ssim + loss_iou + motion_loss
        torch.distributed.barrier()
        total.backward()
        optimizer_video.step()


        reduce_loss = reduce_mean(loss, args.nprocs)
        reduce_loss_ssim = reduce_mean(loss_ssim, args.nprocs)
        reduce_loss_iou = reduce_mean(loss_iou, args.nprocs)
        reduce_loss_shift = reduce_mean(motion_loss, args.nprocs)


        if local_rank == 0:
            pbar.update(1)
            loss_mean_bce += reduce_loss
            loss_mean_ssim += reduce_loss_ssim
            loss_mean_iou += reduce_loss_iou
            loss_mean_motion += reduce_loss_shift
        dist.barrier()

    if local_rank == 0:
        pbar.close()
        loss_mean_bce /= (i + 1)
        loss_mean_ssim /= (i + 1)
        loss_mean_iou /= (i + 1)
        loss_mean_motion /= (i + 1)
        with open(f'log_{args.run_type}.txt', 'a') as file_object:
            information = "loss-video-epoch{}:BCE:{}---SSIM:{}---iou:{}--motion{}---lr:{}\n".format(epoch, loss_mean_bce,loss_mean_ssim,loss_mean_iou,loss_mean_motion,str([optimizer_video.param_groups[0]['lr'],optimizer_video.param_groups[1]['lr']]))
            print(information)
            file_object.write(information)



if __name__ == '__main__':
    main()
