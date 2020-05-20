import torch


def set_device_from_args(args):
    # choose device
    if torch.cuda.is_available() and not args.no_cuda:
        if hasattr(args, 'local_rank'):
            device = f'cuda:{args.local_rank}'
        else:
            device = 'cuda'
    else:
        device = 'cpu'
    setattr(args, 'device', device)


def is_master_process():
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def toggle_grad(model, on_or_off):
    for param in model.parameters():
        param.requires_grad_(on_or_off)
