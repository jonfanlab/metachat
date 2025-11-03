import os
import sys
import yaml
import pandas as pd
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast

import argparse
import timeit
from fvcore.nn import FlopCountAnalysis

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from multi_film_angle_dec_fwdadj_sample_learners import *
from multi_film_angle_dec_fwdadj_sample_otf_dataloader import SimulationDataset

from phys import *
import consts

from torch.utils.tensorboard import SummaryWriter
import datetime

import shutil

def plot_helper(data, row, column, titles, path):
    # data is a list of 2d data to be plot
    tot_sub_plot = len(data)
    assert tot_sub_plot <= row*column
    plt.figure(figsize=(12,3))
    for i in range(tot_sub_plot):
        plt.subplot(row, column, i+1)
        ax = plt.gca()
        im = ax.imshow(data[i])
        plt.title(f"{titles[i]}")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    plt.tight_layout()
    plt.savefig(path, transparent=True, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

def regConstScheduler(args, epoch, last_epoch_data_loss, last_epoch_physical_loss):
    '''
    This function scales the physical regularization scaling constant over
    each epoch.
    '''
    if(epoch < args.physics_start_epoch):
        return 0
    else:
        return 0.5 * last_epoch_data_loss / last_epoch_physical_loss


def MAE_loss(a, b):
    return torch.mean(torch.abs(a - b)) / torch.mean(torch.abs(b))

def MSE_loss(a, b):
    return torch.mean((a - b) ** 2)

def plot_sampling_distribution(wavelength_bins, sample_weights, sample_bin_indices, epoch, save_path):
    """Plot the sampling distribution across wavelength bins."""
    plt.figure(figsize=(10, 5))
    
    # Calculate average weight per bin
    bin_weights = np.zeros(len(wavelength_bins))
    bin_counts = np.zeros(len(wavelength_bins))
    
    for idx, bin_idx in enumerate(sample_bin_indices):
        bin_weights[bin_idx] += sample_weights[idx]
        bin_counts[bin_idx] += 1
    
    # Normalize by count to get average weight per bin
    mask = bin_counts > 0
    bin_weights[mask] /= bin_counts[mask]
    
    # Plot
    plt.bar(wavelength_bins[mask], bin_weights[mask], width=20)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Average Sample Weight')
    plt.title(f'Wavelength Sampling Distribution (Epoch {epoch})')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(f'{save_path}/plots/sampling_dist_epoch_{epoch}.png')
    plt.close()

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    print(args)

    if args.cuda_device != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)

    if not os.path.isdir(args.model_saving_path):
        os.makedirs(args.model_saving_path)
        raise ValueError(f"no path found: {args.model_saving_path}")

    if not os.path.isdir(args.model_saving_path+'/plots'):
        os.mkdir(args.model_saving_path+'/plots')

    # Save all relevant source files to checkpoint directory
    current_file = os.path.abspath(__file__)
    files_to_save = [
        current_file,  # This training script
        os.path.join(os.path.dirname(current_file), 'multi_film_angle_dec_fwdadj_sample_learners.py'),
        os.path.join(os.path.dirname(current_file), 'phys.py'),
        os.path.join(os.path.dirname(current_file), 'consts.py'),
        os.path.join(os.path.dirname(current_file), 'multi_film_angle_dec_fwdadj_sample_otf_dataloader.py')
    ]
    
    source_code_dir = os.path.join(args.model_saving_path, 'source_code')
    if not os.path.exists(source_code_dir):
        os.makedirs(source_code_dir)
    
    # Save args to yaml file
    args_dict = vars(args)
    with open(os.path.join(source_code_dir, 'config.yaml'), 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False)
    
    for file_path in files_to_save:
        if os.path.exists(file_path):
            shutil.copy2(file_path, source_code_dir)
        else:
            print(f"Warning: Could not find {file_path}")

    # Create tensorboard writer
    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    tensorboard_path = os.path.join(args.model_saving_path, 'runs', current_time)
    writer = SummaryWriter(tensorboard_path)

    # First, load metadata to determine indices
    metadata = pd.read_parquet(args.metadata_file)
    if args.min_wavelength is not None:
        metadata = metadata[metadata['wavelength'] >= args.min_wavelength]
    if args.max_wavelength is not None:
        metadata = metadata[metadata['wavelength'] <= args.max_wavelength]
    
    # Limit total number of devices if specified
    if args.num_devices > 0 and args.num_devices < len(metadata):
        metadata = metadata.head(args.num_devices)
        print(f"Using {args.num_devices} devices out of {len(metadata)} available")
    
    # Create train/test indices
    all_indices = np.arange(len(metadata))
    np.random.seed(args.seed)
    np.random.shuffle(all_indices)
    train_size = int(0.9 * len(metadata))
    train_indices = all_indices[:train_size]
    test_indices = all_indices[train_size:]

    # Create training dataset first to get scaling factors
    train_ds = SimulationDataset(
        args.patterns_dir, args.patterns_base, args.fields_dir, args.fields_base, 
        args.src_dir, args.src_base, args.fields_dir_adj, args.fields_base_adj, 
        args.src_dir_adj, args.src_base_adj, args.num_devices, args.metadata_file, 
        min_wavelength=args.min_wavelength, max_wavelength=args.max_wavelength,
        indices=train_indices,
        scaling_factors=None  # No scaling factors for training set
    )
    
    # Get scaling factors from training set
    scaling_factors = train_ds.get_scaling_factors()
    
    # Create test dataset with same scaling factors
    test_ds = SimulationDataset(
        args.patterns_dir, args.patterns_base, args.fields_dir, args.fields_base, 
        args.src_dir, args.src_base, args.fields_dir_adj, args.fields_base_adj, 
        args.src_dir_adj, args.src_base_adj, args.num_devices, args.metadata_file, 
        min_wavelength=args.min_wavelength, max_wavelength=args.max_wavelength,
        indices=test_indices,
        scaling_factors=scaling_factors  # Use training set scaling factors
    )

    train_sampler = WeightedRandomSampler(
        weights=train_ds.get_sample_weights(),
        num_samples=min(len(train_ds), 150000),
        replacement=True
    )

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, 
                            num_workers=8, pin_memory=True, persistent_workers=True, generator=g)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True, 
                           num_workers=8, pin_memory=True, persistent_workers=True, generator=g)
    field_scaling_factor = train_ds.scaling_factor

    # Save scaling factors to file
    scaling_factors = {
        'field_scaling_factor': float(field_scaling_factor),
        'src_data_scaling_factor': float(train_ds.src_data_scaling_factor),
        'max_wavelength': float(train_ds.max_wavelength),
        'min_wavelength': float(train_ds.min_wavelength),
        'max_angle': float(train_ds.max_angle),
        'min_angle': float(train_ds.min_angle)
    }
    
    scaling_file = os.path.join(args.model_saving_path, 'scaling_factors.yaml')
    with open(scaling_file, 'w') as f:
        yaml.dump(scaling_factors, f)

    total_steps = args.epoch * len(train_loader)
    update_times = np.log(args.end_lr / args.start_lr) / np.log(0.99)
    lr_update_steps = int(total_steps / update_times)
    print(f"start_lr: {args.start_lr}, end_lr: {args.end_lr}, total_steps: {total_steps}, lr_update_steps: {lr_update_steps}")

    start_epoch = 0
    if (args.continue_train == 1):
        df = pd.read_csv(args.model_saving_path + '/'+'df.csv')
        print("Restoring weights from ", args.model_saving_path+"/last_model.pt", flush=True)
        checkpoint = torch.load(args.model_saving_path+"/last_model.pt")
        start_epoch = checkpoint['epoch'] + 1
        print(f"start_epoch is {start_epoch}")
        model = checkpoint['model']
        optimizer = optim.Adam(model.parameters(), lr=args.start_lr, weight_decay=args.weight_decay)
        optimizer.load_state_dict(checkpoint['optimizer'].state_dict())
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'].state_dict())
        loss_fn = MAE_loss
        
    else:
        df = pd.DataFrame(columns=['epoch', 'lr', 'train_loss', 'train_phys_reg', 'test_loss', 'test_phys_reg'])

        model = UNet(args.network_depth, args.block_depth, args.num_kernels,
                     input_channels=3,  # 1 structure + 2 source channels
                     output_channels=2, 
                     dropout=args.dropout).float()

        loss_fn = MAE_loss
        optimizer = optim.Adam(model.parameters(), lr=args.start_lr, weight_decay=args.weight_decay)
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    
    tmp = filter(lambda x: x.requires_grad, model.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print('Total trainable tensors:', num, flush=True)
    with open(args.model_saving_path + '/'+'config.txt', 'w') as f:
        f.write(f'Total trainable tensors: {num}')
    
    model.cuda()
    FLOPs_recorded = True

    total_step = len(train_loader)
    gradient_count = 0
    best_loss = 1e4

    last_epoch_data_loss = 1
    last_epoch_physical_loss = 10

    running_data_loss = 1
    running_phys_loss = 10
    running_bc_loss = 10

    scaler = GradScaler()
    for step in range(start_epoch, args.epoch):
        epoch_start_time = timeit.default_timer()
        print("epoch: ", step, flush=True)
        reg_norm = regConstScheduler(args, step, last_epoch_data_loss, last_epoch_physical_loss)
        # training
        model.train()

        for idx, sample_batched in enumerate(train_loader):
            gradient_count += 1
            optimizer.zero_grad(set_to_none=True)
            x_batch_train = sample_batched['structure'].cuda(non_blocking=True)
            y_batch_train = sample_batched['field'].cuda(non_blocking=True)
            src_batch_train = sample_batched['src'].cuda(non_blocking=True)
            wavelengths = sample_batched['wavelength'].cuda(non_blocking=True)
            wavelengths_normalized = sample_batched['wavelength_normalized'].cuda(non_blocking=True)
            angles = sample_batched['angle'].cuda(non_blocking=True)
            angles_normalized = sample_batched['angle_normalized'].cuda(non_blocking=True)
            x = x_batch_train[:, :, 1:-1, 1:-1]
            y = y_batch_train[:, :, 1:-1, 1:-1]
            x_shape = x_batch_train.shape

            pattern = torch.cat(((torch.ones([x_shape[0],1,1,x_shape[-1]], dtype=torch.float32).cuda(non_blocking=True) * consts.n_sub**2), x_batch_train), dim=2)

            omegas = 2 * np.pi * consts.C_0 / wavelengths

            if not FLOPs_recorded:
                FLOPs_recorded = True
                flops = FlopCountAnalysis(model, x)
                print("flops per input device: ", flops.total() / 1e9 / args.batch_size, 'G')
                with open(args.model_saving_path + '/'+'config.txt', 'a') as f:
                    f.write(f'\nFLOPs per input device: {flops.total()/1e9/args.batch_size}(G)')

            logits = model(
                torch.cat((x, src_batch_train[:, :, 1:-1, 1:-1]/train_ds.src_data_scaling_factor), dim=1),
                wavelengths_normalized,
                angles_normalized
            )

            loss = MAE_loss(logits, y) + MSE_loss(logits, y)
            with torch.no_grad():
                data_MAE_loss = MAE_loss(logits, y)

            y_padded = torch.zeros_like(y_batch_train)
            y_padded[:, :, 1:-1, 1:-1] = logits
            y_padded[:, :, 0, :] = y_batch_train[:, :, 0, :]
            y_padded[:, :, -1, :] = y_batch_train[:, :, -1, :]
            y_padded[:, :, :, 0] = y_batch_train[:, :, :, 0]
            y_padded[:, :, :, -1] = y_batch_train[:, :, :, -1]

            phys_reg_Hz = H_to_H(
                -y_padded[:, 0] * field_scaling_factor,
                -y_padded[:, 1] * field_scaling_factor,
                src_batch_train[:, 0]/omegas.unsqueeze(-1).unsqueeze(-1),
                src_batch_train[:, 1]/omegas.unsqueeze(-1).unsqueeze(-1),
                consts.dL * 1e-9,
                omegas.unsqueeze(-1).unsqueeze(-1),
                pattern
            )

            phys_reg_mae = MAE_loss(
                phys_reg_Hz[:, :, :, 1:-1] / field_scaling_factor,
                y_padded[:, :, 1:-1, 1:-1]
            )
            loss += reg_norm * phys_reg_mae

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_data_loss = 0.99 * running_data_loss + 0.01 * data_MAE_loss.item()
            running_phys_loss = 0.99 * running_phys_loss + 0.01 * phys_reg_mae.item()

            if (idx + 1) % 50 == 0:
                print('Epoch [{}/{}], Step [{}/{}], data MAE loss: {:.4f}, phys loss: {:.4f}'.format(
                    step + 1,
                    args.epoch,
                    idx + 1,
                    total_step,
                    data_MAE_loss.item(),
                    phys_reg_mae.item()
                    )
                )

            if gradient_count >= lr_update_steps:
                gradient_count = 0
                lr_scheduler.step()

            writer.add_scalar('Loss/train_data', data_MAE_loss.item(), step * len(train_loader) + idx)
            writer.add_scalar('Loss/train_physics', phys_reg_mae.item(), step * len(train_loader) + idx)
            writer.add_scalar('Learning_Rate', lr_scheduler.get_last_lr()[0], step * len(train_loader) + idx)

            # del logits, y_batch_train, src_batch_train, wavelengths, x_batch_train, pattern, omegas, phys_reg_Hz, loss, phys_reg_mae, y_padded
            # torch.cuda.empty_cache()

        checkpoint = {
                    'epoch': step,
                    'model': model,
                    'optimizer': optimizer,
                    'lr_scheduler': lr_scheduler
                 }
        torch.save(checkpoint, args.model_saving_path+"/last_model.pt")

        # Evaluation
        model.eval()

        train_loss = running_data_loss
        train_phys_reg = running_phys_loss

        test_loss = 0.
        test_phys_reg = 0.

        angle_bins = np.arange(-5, 26, 1)  # -5 to +25 in steps of 1
        angle_binned_losses = {i: {'count': 0, 'loss': 0.0} for i in angle_bins}

        wavelength_bins = np.arange(400, 701, 10)  
        wavelength_binned_losses = {i: {'count': 0, 'loss': 0.0} for i in wavelength_bins}

        with torch.no_grad():
            for idx, sample_batched in enumerate(test_loader):
                x_batch_test = sample_batched['structure'].cuda(non_blocking=True)
                y_batch_test = sample_batched['field'].cuda(non_blocking=True)
                src_batch_test = sample_batched['src'].cuda(non_blocking=True)
                wavelengths = sample_batched['wavelength'].cuda(non_blocking=True)
                wavelengths_normalized = sample_batched['wavelength_normalized'].cuda(non_blocking=True)
                angles = sample_batched['angle'].cuda(non_blocking=True)
                angles_normalized = sample_batched['angle_normalized'].cuda(non_blocking=True)
    
                x = x_batch_test[:, :, 1:-1, 1:-1]
                y = y_batch_test[:, :, 1:-1, 1:-1]
                x_shape = x_batch_test.shape
                pattern = torch.cat(((torch.ones([x_shape[0],1,1,x_shape[-1]], dtype=torch.float32).cuda(non_blocking=True) * consts.n_sub**2), x_batch_test), dim=2)

                omegas = 2 * np.pi * consts.C_0 / wavelengths
                logits = model(
                    torch.cat((x, src_batch_test[:, :, 1:-1, 1:-1]/test_ds.src_data_scaling_factor), dim=1),
                    wavelengths_normalized,
                    angles_normalized
                )
                batch_loss = MAE_loss(logits, y).item()
                test_loss += batch_loss

                angles_np = angles.cpu().numpy()
                for i, angle in enumerate(angles_np):
                    bin_idx = int(round(angle))
                    if bin_idx in angle_binned_losses:
                        sample_loss = MAE_loss(logits[i:i+1], y[i:i+1]).item()
                        angle_binned_losses[bin_idx]['loss'] += sample_loss
                        angle_binned_losses[bin_idx]['count'] += 1

                wavelengths_np = wavelengths.cpu().numpy() * 1e9
                for i, wavelength in enumerate(wavelengths_np):
                    bin_idx = wavelength_bins[np.abs(wavelength_bins - wavelength).argmin()]
                    if bin_idx in wavelength_binned_losses:
                        sample_loss = MAE_loss(logits[i:i+1], y[i:i+1]).item()
                        wavelength_binned_losses[bin_idx]['loss'] += sample_loss
                        wavelength_binned_losses[bin_idx]['count'] += 1

                y_padded_test = torch.zeros_like(y_batch_test)
                y_padded_test[:, :, 1:-1, 1:-1] = logits
                y_padded_test[:, :, 0, :] = y_batch_test[:, :, 0, :]
                y_padded_test[:, :, -1, :] = y_batch_test[:, :, -1, :]
                y_padded_test[:, :, :, 0] = y_batch_test[:, :, :, 0]
                y_padded_test[:, :, :, -1] = y_batch_test[:, :, :, -1]

                phys_reg_Hz = H_to_H(
                    -y_padded_test[:, 0] * field_scaling_factor,
                    -y_padded_test[:, 1] * field_scaling_factor,
                    src_batch_test[:, 0]/omegas.unsqueeze(-1).unsqueeze(-1),
                    src_batch_test[:, 1]/omegas.unsqueeze(-1).unsqueeze(-1),
                    consts.dL * 1e-9,
                    omegas.unsqueeze(-1).unsqueeze(-1),
                    pattern
                )
                test_phys_reg += MAE_loss(
                    phys_reg_Hz[:, :, :, 1:-1] / field_scaling_factor,
                    y_padded_test[:, :, 1:-1, 1:-1]
                ).item()

                if idx == 0:
                    data = [
                        x_batch_test[0, 0, :, :].cpu().numpy(),
                        y_batch_test[0, 0, :, :].cpu().numpy(),
                        logits[0, 0, :, :].cpu().numpy(),
                        y_batch_test[0, 0, 1:-1, 1:-1].cpu().numpy() - logits[0, 0, :, :].cpu().numpy(),
                        phys_reg_Hz[0, 0, :, 1:-1].cpu().numpy() / field_scaling_factor
                    ]
                    titles = ["eps", "Hz_gt", "Hz_out", "Hz_error", "FD_H"]
                    plot_helper(data, 1, 5, titles, args.model_saving_path+"/plots/epoch_"+str(step)+".png")

                # del logits, y_batch_test, src_batch_test, wavelengths, x_batch_test, pattern, omegas, phys_reg_Hz, y_padded_test
                # torch.cuda.empty_cache()

        angle_avg_losses = {}
        for angle in angle_binned_losses:
            if angle_binned_losses[angle]['count'] > 0:
                angle_avg_losses[angle] = angle_binned_losses[angle]['loss'] / angle_binned_losses[angle]['count']
            else:
                angle_avg_losses[angle] = float('nan')

        wavelength_avg_losses = {}
        for wavelength in wavelength_binned_losses:
            if wavelength_binned_losses[wavelength]['count'] > 0:
                wavelength_avg_losses[wavelength] = wavelength_binned_losses[wavelength]['loss'] / wavelength_binned_losses[wavelength]['count']
            else:
                wavelength_avg_losses[wavelength] = float('nan')

        for angle, loss in angle_avg_losses.items():
            if not np.isnan(loss):
                writer.add_scalar(f'Loss/test_by_angle/{angle}deg', loss, step)

        for wavelength, loss in wavelength_avg_losses.items():
            if not np.isnan(loss):
                writer.add_scalar(f'Loss/test_by_wavelength/{wavelength}nm', loss, step)

        plt.figure(figsize=(10, 5))
        angles = sorted(angle_avg_losses.keys())
        losses = [angle_avg_losses[a] for a in angles]
        plt.plot(angles, losses, 'b.-')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('MAE Loss')
        plt.title(f'Angle-dependent Loss (Epoch {step})')
        plt.grid(True)
        plt.savefig(f'{args.model_saving_path}/plots/angle_loss_epoch_{step}.png')
        plt.close()

        plt.figure(figsize=(10, 5))
        wavelengths_plot = sorted(wavelength_avg_losses.keys())
        wavelength_losses = [wavelength_avg_losses[w] for w in wavelengths_plot]
        plt.plot(wavelengths_plot, wavelength_losses, 'b.-')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('MAE Loss')
        plt.title(f'Wavelength-dependent Loss (Epoch {step})')
        plt.grid(True)
        plt.savefig(f'{args.model_saving_path}/plots/wavelength_loss_epoch_{step}.png')
        plt.close()

        valid_losses = [l for l in losses if not np.isnan(l)]
        if valid_losses:
            writer.add_scalar('Loss/test_angle_std', np.std(valid_losses), step)
            writer.add_scalar('Loss/test_angle_max', np.max(valid_losses), step)
            writer.add_scalar('Loss/test_angle_min', np.min(valid_losses), step)

        valid_wavelength_losses = [l for l in wavelength_losses if not np.isnan(l)]
        if valid_wavelength_losses:
            writer.add_scalar('Loss/test_wavelength_std', np.std(valid_wavelength_losses), step)
            writer.add_scalar('Loss/test_wavelength_max', np.max(valid_wavelength_losses), step)
            writer.add_scalar('Loss/test_wavelength_min', np.min(valid_wavelength_losses), step)

        test_loss /= len(test_loader)
        test_phys_reg /= len(test_loader)
        last_epoch_data_loss = test_loss
        last_epoch_physical_loss = test_phys_reg

        print('train loss: {:.5f}, test loss: {:.5f}'.format(train_loss, test_loss), flush=True)
        new_df = pd.DataFrame([[step + 1, str(lr_scheduler.get_last_lr()), train_loss, train_phys_reg, test_loss, test_phys_reg]], 
                              columns=['epoch', 'lr', 'train_loss', 'train_phys_reg', 'test_loss', 'test_phys_reg'])
        df = pd.concat([df, new_df])

        df.to_csv(args.model_saving_path + '/'+'df.csv', index=False)

        if(test_loss < best_loss):
            best_loss = test_loss
            checkpoint = {
                            'epoch': step,
                            'model': model,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer,
                            'lr_scheduler': lr_scheduler
                         }
            torch.save(checkpoint, args.model_saving_path+"/best_model.pt")
        epoch_stop_time = timeit.default_timer()
        print("epoch run time:", epoch_stop_time - epoch_start_time)

        writer.add_scalar('Loss/test_data', test_loss, step)
        writer.add_scalar('Loss/test_physics', test_phys_reg, step)

        # ==============================================
        # On-the-fly wavelength-dependent re-weighting
        # ==============================================
        # Recompute training sample weights based on test wavelength losses
        train_wavelength_bins = train_ds.get_wavelength_bins()
        # Collect bin losses aligned with train_wavelength_bins
        bin_losses_list = []
        for wb in train_wavelength_bins:
            val = wavelength_avg_losses.get(wb, np.nan)
            bin_losses_list.append(val)
        bin_losses = np.array(bin_losses_list, dtype=np.float32)
        nan_mask = np.isnan(bin_losses)
        if np.any(nan_mask):
            if np.any(~nan_mask):
                bin_losses[nan_mask] = np.nanmean(bin_losses[~nan_mask])
            else:
                bin_losses[nan_mask] = 1.0  # default if all are NaN

        # Simple weighting scheme: bin weight proportional to bin loss
        bin_weights = bin_losses / np.sum(bin_losses)

        sample_bin_indices = train_ds.get_wavelength_bin_indices()
        new_sample_weights = bin_weights[sample_bin_indices]

        train_ds.update_sample_weights(new_sample_weights)

        # Plot sampling distribution
        plot_sampling_distribution(
            train_ds.get_wavelength_bins(),
            train_ds.get_sample_weights(),
            train_ds.get_wavelength_bin_indices(),
            step,
            args.model_saving_path
        )

        # Create new train_loader with updated weights
        train_sampler = WeightedRandomSampler(
            weights=train_ds.get_sample_weights(),
            num_samples=min(len(train_ds), 150000),
            replacement=True
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, 
                                num_workers=8, pin_memory=True, persistent_workers=True, generator=g)


    writer.close()

if __name__ == '__main__':
    torch.set_default_tensor_type(torch.FloatTensor)

    config_path = sys.argv[1]

    parser = argparse.ArgumentParser(description="Arguments for the controlnet model")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    args = argparse.Namespace(**config)

    main(args)
