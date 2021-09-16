import os
import torch


def load_checkpoint(checkpoint_path,
                    gen,
                    dis,
                    optimizer_gen,
                    optimizer_dis,
                    name,
                    ):

    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_dict = torch.load(checkpoint_path + '/gen_' + name, map_location='cpu')
    optimizer_gen.load_state_dict(checkpoint_dict['optimizer'])
    gen.load_state_dict(checkpoint_dict['model'])

    epoch = checkpoint_dict['epoch']
    checkpoint_dict = torch.load(checkpoint_path + '/dis_' + name, map_location='cpu')
    optimizer_dis.load_state_dict(checkpoint_dict['optimizer'])
    dis.load_state_dict(checkpoint_dict['model'])

    print("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, epoch))
    return gen, dis, optimizer_gen, optimizer_dis, epoch


def save_checkpoint(gen,
                    dis,
                    optimizer_gen,
                    optimizer_dis,
                    learning_rate,
                    epoch,
                    checkpoint_path,
                    name,
                    ):
    # model.load_state_dict(model.state_dict())
    os.makedirs(f"{checkpoint_path}", exist_ok=True)
    torch.save({'model': gen.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer_gen.state_dict(),
                'learning_rate': learning_rate}, checkpoint_path + '/gen_' + name)

    torch.save({'model': dis.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer_dis.state_dict(),
                'learning_rate': learning_rate}, checkpoint_path + '/dis_' + name)

