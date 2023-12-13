import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import torch.nn.functional as F
import torch.multiprocessing as mp
import argparse
from dataloader import *
from model import MyModel
import platform

parser = argparse.ArgumentParser(description='VSOD Testing')

# If you have 3 GPUs, the real batch size is (3*batch_size_eval).
parser.add_argument('--batch_size_eval', default=18, type=int, help="batch_size is on a gpu")

parser.add_argument('--test_size', default=[288, 288], type=list, help="[height, width]")

# path to the weight file
parser.add_argument('--load_model_path', default="MMN_VSOD.pt", type=str)

# path to the parent directory of dataset
parser.add_argument('--test_video_dir', default="/path/to/dataset/test", type=str)
# name of the dataset
parser.add_argument('--test_video_datasets', default=['FBMS','DAVSOD', 'DAVIS','VISAL'], type=list)

parser.add_argument('--video_time_clips', default=4, type=int)
parser.add_argument('--time_interval', default=1, type=int)
parser.add_argument('--run_type', default="test_results", type=str)

#difference between windows and ubutun
if platform.system().lower() == 'windows':
    backend = 'gloo'
    num_workers = 0
else:
    backend = "nccl"
    num_workers = 6


def main():
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))


def main_worker(local_rank, nprocs, args):
    args.local_rank = local_rank
    dist.init_process_group(backend=backend, init_method='tcp://127.0.0.1:24568', world_size=args.nprocs, rank=local_rank)

    model = MyModel()
    if args.load_model_path is not None and args.load_model_path != "":
        checkpoint = torch.load(args.load_model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint, strict=False)
        print(f"load checkpoint {args.load_model_path} success")
    else:
        exit("model weights not found!")

    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)

    test_dataset = TestDataset(test_video_dir=args.test_video_dir, test_video_datasets=args.test_video_datasets,test_size=args.test_size, time_interval=args.time_interval,video_time_clips = args.video_time_clips,test_all=True)
    test_sampler = DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size_eval, pin_memory=True,
                                                     drop_last=False, shuffle=False, sampler=test_sampler, num_workers=num_workers)

    val_times = [None for _ in range(args.nprocs)]
    if local_rank == 0:
        val_time = time.strftime("%H-%M-%S")
    else:
        val_time = None
    dist.all_gather_object(val_times, val_time)

    prefix = f"val-time-{val_times[0]}"
    val(test_loader, model, local_rank, args, prefix)


def val(test_loader, model, local_rank, args, sava_folder):
    model.eval()
    sava_folder = os.path.join(args.run_type,sava_folder)
    if local_rank == 0:
        for dataset in args.test_video_datasets:
            for i in os.listdir(os.path.join(args.test_video_dir, dataset)):
                os.makedirs(os.path.join(sava_folder, dataset, i))
        pbar = tqdm(total=len(test_loader))
    dist.barrier()
    with torch.no_grad():
        for batch, (inputs, sizes, paths) in (enumerate(test_loader)):
            outputs = model(inputs)
            batch_size,T = outputs.size(0),outputs.size(1)
            for idx in range(batch_size):
                pred = outputs[idx]
                for t in range(T):
                    now = F.interpolate(pred[t].unsqueeze(0), sizes[idx,t].tolist(),mode="bilinear").squeeze().detach().cpu().numpy()
                    now = np.uint8(now*255)
                    cv2.imwrite(os.path.join(sava_folder,paths[t][idx]), now)
            if local_rank == 0:
                pbar.update(1)
        dist.barrier()

        if local_rank == 0:
            pbar.close()
            print(f"Finished! Results are saved in {sava_folder}")


if __name__ == '__main__':
    main()