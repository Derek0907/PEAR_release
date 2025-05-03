import os
import argparse
import time
import torch
import yaml
import torch.utils.data as Data
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datasets.ego4ddataset import Ego4dDataset
from utils1.train_one_epoch import train_one_epoch
from utils1.eval_one_epoch import evaluate
from utils1.utils import get_lr_scheduler, set_optimizer_lr, seed_everything, save_args
from utils1.data_io import read_data
from models.MANO.configs import get_config
from models.pear_pipeline import Net
from multi_train_utils.distributed_utils import init_distributed_mode, cleanup
from collections import defaultdict

def main(args):

    seed_everything(args.seed)

    # Initialize distributed training environment
    init_distributed_mode(args=args)

    current_time = str(int(time.time()))
    rank = args.rank
    save_root = os.path.join(args.save_root, current_time)
    device = torch.device(args.device)
    batch_size = args.batch_size
    
    # Save training arguments and create save directory (only on rank 0)
    if rank == 0:
        if args.save_args:
            save_root = os.path.join(args.save_root, current_time)
            os.makedirs(save_root, exist_ok=True)
            save_args(args, save_root)
            
        # Create metrics log file for this seed
        metrics_file = os.path.join(args.save_root, f"metrics_seed_{args.seed}.txt")
        with open(metrics_file, "w") as f:
            f.write(f"Seed: {args.seed}\n")
            f.write(f"Status: Running\n")
            f.write("Validation metrics:\n")
            
    # Load dataset configuration from YAML file
    with open(args.yaml_path, 'r') as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)


    # Initialize TensorBoard (only on rank 0)
    if rank == 0: 
        print(args)
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter()
        os.makedirs("./weights", exist_ok=True)

   
    # Load training and validation datasets
    if rank == 0:
        print("Reading train dataset")
    train_data = read_data('train', yaml_data)
    if rank == 0:
        print("Reading val dataset")
    val_data = read_data('val', yaml_data)


    # Initialize datasets
    train_dataset = Ego4dDataset(*train_data)
    val_dataset = Ego4dDataset(*val_data)

    # Distributed sampling for multi-GPU training
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    
    # Batch sampler for training loader
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)
    nw = 8  # Number of data loader workers
    if rank == 0:
        print(f'Using {nw} dataloader workers per process.')
    # Data loaders
    train_loader = Data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, 
                                   pin_memory=True, num_workers=nw, shuffle=False)

    if rank == 0:
        val_loader = Data.DataLoader(val_dataset, batch_size=batch_size,
                                    shuffle=False, pin_memory=True, num_workers=0)
    else:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        val_loader = Data.DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler,
                                    shuffle=False, pin_memory=True, num_workers=0)
    
    # Initialize model
    model_cfg = args.cfg
    my_model = Net(cfg=get_config(model_cfg))
    model = my_model.to(device)
   
   
    # Freeze all BLIP model parameters.
    for p in model.blip_model.parameters():
        p.requires_grad = False

    # Apply synchronized BatchNorm if specified
    if args.syncBN:
        if rank == 0:
            print("Using synchronized batch norm")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # Wrap model for distributed training
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    
    # Load pretrained weights if specified
    if args.pretrain and os.path.exists(args.weights):
        print("Load pretrained models, and it will recover init weights!")
        weights_dict = torch.load(args.weights, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(load_weights_dict, strict=False)
        
    # Learning rate configuration
    Init_lr = args.Init_lr
    nbs = 4  # Nominal batch size reference
    lr_limit_max = 2e-4 if args.optimizer_type in ['adam', 'adamw'] else 5e-2
    lr_limit_min = 3e-5 if args.optimizer_type in ['adam', 'adamw'] else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * args.Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    # Optimizer setup
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = {
        'adam': optim.Adam(pg, Init_lr_fit, betas=(args.momentum, 0.999), weight_decay=args.weight_decay),
        'adamw': optim.AdamW(pg, Init_lr_fit, betas=(args.momentum, 0.999), weight_decay=args.weight_decay),
        'sgd': optim.SGD(pg, Init_lr_fit, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
    }[args.optimizer_type]

    lr_scheduler_func = get_lr_scheduler(args.lr_decay_type, Init_lr_fit, Min_lr_fit, args.epochs)

    if rank == 0:
        if not args.evaluate_only:
            print("Start training!")
        else:
            print("Start evaluation!")

    eval_results = defaultdict(list)
    for epoch in range(args.epochs):
        # Training phase
        if not args.evaluate_only:
            train_sampler.set_epoch(epoch)

            # Update learning rate
            nbs = 4
            lr_limit_max = 1e-4 if args.optimizer_type in ['adam', 'adamw'] else 5e-2
            lr_limit_min = 5e-5 if args.optimizer_type in ['adam', 'adamw'] else 5e-4
            Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit = min(max(batch_size / nbs * args.Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
            lr_scheduler_func = get_lr_scheduler(args.lr_decay_type, Init_lr_fit, Min_lr_fit, args.epochs)
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            # Train one epoch
            train_loss = train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader,
                                        device=device, epoch=epoch, args=args)

            if rank == 0:
                results_file = os.path.join(save_root, f"metrics_seed_{args.seed}.txt")
                os.makedirs(save_root, exist_ok=True)
                with open(results_file, "a") as f:
                    f.write(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}\n")

            # Validation phase
            if (args.eval_period > 0 and (epoch + 1) % args.eval_period == 0):
                if rank == 0:
                    re, sim, auc, nss, precision, recall, f1, ade1, fde1, ade2, fde2 = evaluate(model=model, data_loader=val_loader,
                                                                            device=device, epoch=epoch, args=args,
                                                                            save_root=save_root)    
                    # Save model
                    if args.save_dict:
                            save_path = os.path.join(save_root, 'weight')
                            os.makedirs(save_path, exist_ok=True)
                            torch.save(model.state_dict(), os.path.join(save_path, f"epoch_{epoch+1}.pth"))
        else:
            # Evaluation-only mode
            re, sim, auc, nss, precision, recall, f1, ade1, fde1, ade2, fde2 = evaluate(model=model, data_loader=val_loader,
                                                                      device=device, epoch=epoch, args=args,
                                                                      save_root=save_root)
            eval_results["PA-MPJPE"].append(re)
            eval_results["sim"].append(sim)
            eval_results["auc-j"].append(auc)
            eval_results["nss"].append(nss)
            eval_results["precision"].append(precision)
            eval_results["recall"].append(recall)
            eval_results["f1"].append(f1)
            eval_results["ade1"].append(ade1)
            eval_results["fde1"].append(fde1)
            eval_results["ade2"].append(ade2)
            eval_results["fde2"].append(fde2)
            
            
    if args.evaluate_only and rank == 0:
        avg_results = {k: sum(v) / len(v) for k, v in eval_results.items()}
        print("\n=== Averaged Evaluation Results ===")
        for k, v in avg_results.items():
            print(f"{k}: {v:.4f}")

    cleanup() 
    return 

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for hand-object interaction anticipation model.")

    # Training configuration
    parser.add_argument('--epochs', type=int, default=65,
                        help='Total number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=12,
                        help='Batch size per GPU for distributed training.')

    # Dataset paths
    parser.add_argument('--yaml_path', type=str, default='cfg/all.yaml',
                        help='Path to YAML config file specifying dataset details.')

    # Model evaluation
    parser.add_argument('--eval_period', type=int, default=1,
                        help='Evaluate the model every N epochs.')

    # Save options
    parser.add_argument('--save_dict', action='store_true',
                        help='Save model weights if evaluation criteria are met.')

    # Model and training configuration
    parser.add_argument('--cfg', type=str, default='cfg/BLIP.yaml',
                        help='Path to model configuration YAML file.')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training (e.g., "cuda", "cuda:0", or "cpu").')

    # Distributed training
    parser.add_argument('--world-size', default=4, type=int,
                        help='Number of processes for distributed training (usually equals number of GPUs).')
    parser.add_argument('--dist-url', default='env://',
                        help='URL used to set up distributed training environment.')

    # Optimizer settings
    parser.add_argument('--optimizer_type', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'],
                        help='Type of optimizer to use.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum value for SGD or betas for Adam optimizers.')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay (L2 penalty) applied to optimizer.')
    parser.add_argument('--lr_decay_type', type=str, default='cos', choices=['cos', 'step'],
                        help='Learning rate decay type: "cos" for cosine decay, "step" for step decay.')
    parser.add_argument('--Init_lr', type=float, default=0.001,
                        help='Base initial learning rate before scaling with batch size.')
    parser.add_argument('--Min_lr', type=float, default=0.0001,
                        help='Minimum learning rate after decay.')

    # BatchNorm and GPU settings
    parser.add_argument('--syncBN', action='store_true',
                        help='Use synchronized Batch Normalization across GPUs.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id for single-process multi-GPU training.')
    parser.add_argument('--seed', type=int, default=3407,
                        help='Random seed.')

    # Argument saving and root paths
    parser.add_argument('--save_args', action='store_true',
                        help='Save training arguments into a file for record.')
    parser.add_argument('--save_root', type=str, default='./',
                        help='Root directory for saving outputs, weights, and visualizations.')

    # Loss and model training type
    parser.add_argument('--int_loss_type', type=str, default='hybrid', choices=['hybrid', 'heatmap', 'hotspot'],
                        help='Type of interaction loss to use.')
    parser.add_argument('--train_type', type=str, default='hybrid', choices=['aff', 'pose', 'hybrid'],
                        help='Type of task to train: affordance only, pose only, or both (hybrid).')
    parser.add_argument('--text_encoder', type=str, default='blip', choices=['clip', 'blip', 'blip-2'],
                        help='Choice of text encoder backbone.')

    # Evaluation and Pretrained
    parser.add_argument('--evaluate_only', action='store_true',
                        help='Only run evaluation without training.')
    parser.add_argument('--pretrain', action='store_true',
                        help='Use pretrained weights for initialization.')
    parser.add_argument('--weights', type=str,
                        help='Path to pretrained model weights.')

    opt = parser.parse_args()
    main(opt)