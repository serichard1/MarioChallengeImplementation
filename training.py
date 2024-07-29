from datetime import date as d
from modules import dataset, augmentations, utils, custom_losses, custom_models, engine, test_metrics, base_models
from pathlib import Path
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
import os
import torch
import torch.backends.cudnn as cudnn

def get_args_parser():

    parser = argparse.ArgumentParser(
        "Task1 - challenge MICCAI - LaBRI - 2024",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False)

    # training hyperparameters
    parser.add_argument('--img_size', nargs='+', type=int,
        help=""" Input size for the images """)
    parser.add_argument('--learning_rate', default=5e-5, type=float, 
        help="""Initial value of the learning rate.""")
    parser.add_argument('--model', default='inceptionresnet', type=str, 
        help="""model to be used.""")
    parser.add_argument('--weight_decay', default=0.05, type=float, 
        help="""Initial value of the weight decay.""")
    parser.add_argument('--batch_size_per_gpu', default=16, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--n_epochs', default=100, type=int, 
        help='Number of epochs of training.')
    parser.add_argument("--patience",  default=5, type=int,
        help='Number of epochs to wait before stopping the training when validation loss stopped decreasing')
    parser.add_argument('--mean', default=[0.1880, 0.1880, 0.1880], type=list,
        help="""OCT imgs mean for normalization""")
    parser.add_argument('--std', default=[0.2244, 0.2244, 0.2244], type=list,
        help="""OCT imgs std for normalization""")
    parser.add_argument('--dropout', default=0.4, type=float, 
        help="""dropout of classi head""")
    

    # training environment
    parser.add_argument('--use_fp16', default=True, action=argparse.BooleanOptionalAction,
        help="""Whether or not to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance.""")
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--avoid_fragmentation', default=False, action=argparse.BooleanOptionalAction, help='Whether or not to set a max split size for memory')
    parser.add_argument('--log_freq', default=10, type=int, help='Log frequency')
    parser.add_argument('--data_path', default="data", type=str)
    parser.add_argument('--output_dir', default="./output_Task1", type=str, 
        help='Path to save tensorboard logs and model checkpoints during training.')
    parser.add_argument('--seed', default=3407, type=int, 
        help='Seed for random number generation.')
    parser.add_argument("--dist_url", default="env://", type=str, 
        help="""url used to set up distributed training; 
        see https://pytorch.org/docs/stable/distributed.html""")

    return parser


def main(args):
    if args.avoid_fragmentation:
        print('AVOID ON')
        print('INFO: setting "PYTORCH_CUDA_ALLOC_CONF" to  "max_split_size_mb:32"')
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
    print('INFO: cuda available: ', torch.cuda.is_available())

    with open(Path(args.output_dir) / '_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2, default=lambda o: '<not serializable>')
    utils.fix_random_seeds(args.seed)
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    transforms_train = augmentations.AugmGeneric(
                                img_size=tuple(args.img_size), 
                                mean = args.mean,
                                std = args.std,
                                )
    
    transforms_test = augmentations.AugmGeneric(
                                img_size=tuple(args.img_size), 
                                mean = args.mean,
                                std = args.std,
                                test=True
                                )
    
    train_set = dataset.TwinSet(args.data_path, transforms=transforms_train, split='train')
    valid_set = dataset.TwinSet(args.data_path, transforms=transforms_test, split='val')
    test_set = dataset.TwinSet(args.data_path, transforms=transforms_test, split='val')

    test_loader = DataLoader(
        test_set,
        shuffle=False,
        batch_size=6,
        drop_last=False)
    
    print('INFO: Dataset correctly loaded')
    print("INFO: Training dataset size: ", len(train_set))
    print("INFO: Validation dataset size: ", len(valid_set))
    print("INFO: Test dataset size: ", len(test_set))

    dataiter = iter(test_loader)
    batch = next(dataiter)

    print('INFO: batches of shape (batch, channels, height, width): ', [k.shape for k in batch])
    print('INFO: labels of shape (batch, channels, height, width): ', batch[-1].shape)


    print('using weighted cross entropy with weights', train_set.class_weights)
    criterion = nn.CrossEntropyLoss(weight=train_set.class_weights)
    criterion2 = custom_losses.FocalLoss(alpha=torch.tensor([0.8, 0.1, 1.]).cuda(), gamma=4)
    cosine_loss = nn.CosineEmbeddingLoss()

    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard_logs'))

    print("INFO: Losses, scaler and logs writer ready.")
    date = d.today()

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory = True,
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_set,
        shuffle=True,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory = True,
        drop_last=True
        )

    m = {'convnext': base_models.convnext_tiny, 'effcnt': base_models.efficientnet_v2_s, 'inceptionresnet': base_models.inception_resnet_v2}
    s = {'convnext': 768, 'effcnt': 1280, 'inceptionresnet': 1536}
    teacher = custom_models.DualVision(backbone=m[args.model](), in_size=s[args.model], drop_ratio_head=0.)
    model = custom_models.DualVision_student(backbone=m[args.model](), in_size=s[args.model], drop_ratio_head=args.dropout)

    original_state_dict = torch.load("./ckpt_bestsofar_2024-07-26.pth")
    teacher.load_state_dict(original_state_dict)
    modified_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in original_state_dict.items() if k in modified_state_dict and v.size() == modified_state_dict[k].size()}

    modified_state_dict.update(filtered_state_dict)

    model.load_state_dict(modified_state_dict)
    print('correctly loaded pretrained...')

    for param in teacher.parameters():
        param.requires_grad = False

    learningr = args.learning_rate
    optimizer = torch.optim.AdamW(
        model.parameters(),
        eps=1e-8, betas=(0.9, 0.999),
        lr=learningr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader), eta_min=5e-7)
    
    mem = torch.cuda.mem_get_info()
    model.cuda()
    teacher.cuda()

    print('INFO: CUDA memory usage before loading model on gpu: free:', utils.GetHumanReadable(mem[0]), ' / total:', utils.GetHumanReadable(mem[1]))
    mem = torch.cuda.mem_get_info()
    print('INFO: CUDA memory usage after loading model: free:', utils.GetHumanReadable(mem[0]), ' / total:', utils.GetHumanReadable(mem[1]))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        print(f"number of GFLOPs: {flops / 1e9}")

    print("INFO: Model successfully loaded and set on gpu(s)")

    min_loss = 100000
    for epoch in range(args.n_epochs):

        train_stats = engine.train_distill_one_epoch(teacher,
                                                     model, 
                                        train_loader, 
                                        criterion,
                                        criterion2,
                                        cosine_loss,
                                        epoch, 
                                        args.n_epochs, 
                                        args.log_freq, 
                                        fp16_scaler, 
                                        optimizer,
                                        scheduler)
        
        valid_stats = engine.valid_distill_one_epoch(teacher,
                                                     model,  
                                        valid_loader, 
                                        criterion,
                                        criterion2,
                                        cosine_loss,
                                        epoch, 
                                        args.n_epochs, 
                                        args.log_freq, 
                                        fp16_scaler)
                
        if valid_stats["loss"] < min_loss:
            min_loss = valid_stats["loss"]
            trigger_times = 0
            utils.save_on_master(model.state_dict(), os.path.join(args.output_dir, f'ckpt_bestsofar_{date}.pth'))
        else:
            trigger_times += 1

        log_stats_train = {**{f'train_{k}': v for k, v in train_stats.items()},
                'epoch': epoch}
        log_stats_valid = {**{f'valid_{k}': v for k, v in valid_stats.items()},
                'epoch': epoch}
                
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats_train) + "\n")
                f.write(json.dumps(log_stats_valid) + "\n")

        writer.add_scalars(f"cs_{date}", {
                                            'Loss/train': train_stats["loss"],
                                            'Loss/validation': valid_stats["loss"],
                                            'Accuracy/train': train_stats["accuracy"],
                                            'Accuracy/valid': valid_stats["accuracy"]
                                            }, epoch)
        
        print(f'LOG: Epoch {epoch}')
        print(f'Train Acc. => {round(train_stats["accuracy"],3)}%', end=' | ')
        print(f'Train Loss => {round(train_stats["loss"],5)}')
        
        print(f'Valid Acc. => {round(valid_stats["accuracy"],3)}%', end=' | ')
        print(f'valid Loss => {round(valid_stats["loss"],5)} (earlystop => {trigger_times}/{args.patience}) \n')

        if trigger_times >= args.patience:
            print('WARNING: Early stop !')
            print(f'Best validation loss was {min_loss}')
            utils.save_on_master(model.state_dict(), os.path.join(args.output_dir, f'ckpt_final_{date}.pth'))
            break

    print(f'INFO: End of training')
    best_ckpt = os.path.join(args.output_dir, f'ckpt_bestsofar_{date}.pth')
    print(f'INFO: Best weights have been saved in: ', {best_ckpt})

    state_dict = torch.load(best_ckpt)
    model.load_state_dict(state_dict)

    raw_output = test_metrics.make_inferences(model, test_loader, fp16_scaler, test_set.instances)
    log_list_results = {f'test_results_{k}': v for k, v in raw_output.items()}
    if utils.is_main_process():
        with (Path(args.output_dir) / "log_lists_raw_results.txt").open("a") as f:
            f.write(json.dumps(log_list_results) + "\n")
        
        test_metrics.export_results(raw_output, ["REDUCED: 0", "STABLE: 1", "WORSENED: 2"], date, args.output_dir)

    torch.cuda.empty_cache()

    print(f'INFO: End of training, all folds completed')
    print(f'> exiting ...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Crossmodal SSL pretraining from retinal images (CFP and OCT) - LaBRI - 2024", parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)