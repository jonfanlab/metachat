import os.path
import numpy as np
import torch
import random
from torch.utils.data import Dataset
import consts
import pandas as pd
from datetime import datetime

class SimulationDataset(Dataset):
    def __init__(self, patterns_dir, patterns_base, fields_dir, fields_base, src_dir, src_base,
                 fields_dir_adj, fields_base_adj, src_dir_adj, src_base_adj,
                 num_devices, metadata_file, min_wavelength=None, max_wavelength=None, 
                 indices=None, scaling_factors=None):
        # Load metadata
        self.metadata = pd.read_parquet(metadata_file)
        
        # Filter by wavelength range if provided
        if min_wavelength is not None:
            self.metadata = self.metadata[self.metadata['wavelength'] >= min_wavelength]
        if max_wavelength is not None:
            self.metadata = self.metadata[self.metadata['wavelength'] <= max_wavelength]
        print(f"Found {len(self.metadata)} samples within wavelength range")
        
        # Adjust num_devices if larger than available
        if num_devices > len(self.metadata):
            print(f"Warning: Requested {num_devices} devices, but only {len(self.metadata)} are available.")
            num_devices = len(self.metadata)

        # If indices are provided, use them to select samples
        if indices is not None:
            self.metadata = self.metadata.iloc[indices].reset_index(drop=True)
        else:
            self.metadata = self.metadata.head(num_devices)

        raw_domain_shape = tuple(self.metadata['domain_shape'].iloc[0])
        domain_shape = (raw_domain_shape[1], raw_domain_shape[0])
        
        if not all(self.metadata['domain_shape'].apply(lambda x: tuple(x) == raw_domain_shape)):
            raise ValueError("Not all domain shapes are the same. This implementation assumes uniform domain shapes.")

        total_samples = len(self.metadata) * 2
        self.patterns = np.zeros((total_samples, 1, domain_shape[0], domain_shape[1]), dtype=np.float32)
        self.fields = np.zeros((total_samples, 2, domain_shape[0], domain_shape[1]), dtype=np.float32)
        self.src = np.zeros((total_samples, 2, domain_shape[0], domain_shape[1]), dtype=np.float32)
        self.wavelengths = np.zeros(total_samples, dtype=np.float32)
        self.angles = np.zeros(total_samples, dtype=np.float32)

        grouped_metadata = self.metadata.groupby('file_name')

        sample_index = 0
        for file_name, group in grouped_metadata:
            print(f"Loading file {file_name} at {datetime.now()}")

            patterns_fn = os.path.join(patterns_dir, file_name)
            fields_fn = os.path.join(fields_dir, file_name.replace('eps_r_', fields_base))
            src_fn = os.path.join(src_dir, file_name.replace('eps_r_', src_base))
            fields_adj_fn = os.path.join(fields_dir_adj, file_name.replace('eps_r_', fields_base_adj))
            src_adj_fn = os.path.join(src_dir_adj, file_name.replace('eps_r_', src_base_adj))

            file_indices = group['file_index'].values
            num_samples_curr = len(file_indices)

            curr_patterns = np.expand_dims(np.load(patterns_fn, mmap_mode='r')[file_indices].transpose(0, 2, 1), axis=1)
            curr_fields = np.load(fields_fn, mmap_mode='r')[file_indices].transpose(0, 2, 1)
            curr_src = np.load(src_fn, mmap_mode='r')[file_indices].transpose(0, 2, 1)
            curr_fields_adj = np.load(fields_adj_fn, mmap_mode='r')[file_indices].transpose(0, 2, 1)
            curr_src_adj = np.load(src_adj_fn, mmap_mode='r')[file_indices].transpose(0, 2, 1)

            forward_end = sample_index + num_samples_curr
            self.patterns[sample_index:forward_end] = curr_patterns.astype(np.float32)
            self.fields[sample_index:forward_end] = np.stack([curr_fields.real, curr_fields.imag], axis=1).astype(np.float32)
            self.src[sample_index:forward_end] = np.stack([curr_src.real, curr_src.imag], axis=1).astype(np.float32)
            self.wavelengths[sample_index:forward_end] = group['wavelength'].values
            self.angles[sample_index:forward_end] = group['angle'].values

            adj_start = forward_end
            adj_end = adj_start + num_samples_curr
            self.patterns[adj_start:adj_end] = curr_patterns.astype(np.float32)
            self.fields[adj_start:adj_end] = np.stack([curr_fields_adj.real, curr_fields_adj.imag], axis=1).astype(np.float32)
            self.src[adj_start:adj_end] = np.stack([curr_src_adj.real, curr_src_adj.imag], axis=1).astype(np.float32)
            self.wavelengths[adj_start:adj_end] = group['wavelength'].values
            self.angles[adj_start:adj_end] = group['angle'].values

            sample_index = adj_end

        self.max_wavelength = np.max(self.wavelengths)
        self.min_wavelength = np.min(self.wavelengths)
        self.wavelengths_normalized = (self.wavelengths - self.min_wavelength) / (self.max_wavelength - self.min_wavelength)

        self.max_angle = np.max(self.angles)
        self.min_angle = np.min(self.angles)
        self.angles_normalized = (self.angles - self.min_angle) / (self.max_angle - self.min_angle)

        # Convert wavelengths to nm
        self.wavelengths_nm = self.wavelengths * 1e9
        # Define wavelength bins (increments of 25 nm from 400 to 700 nm)
        self.wavelength_bins = np.arange(400, 701, 25)
        # Compute bin indices for each sample
        self.wavelength_bin_indices = np.digitize(self.wavelengths_nm, self.wavelength_bins, right=True) - 1

        if scaling_factors is not None:
            self.scaling_factor = scaling_factors['field_scaling_factor']
            self.src_data_scaling_factor = scaling_factors['src_data_scaling_factor']
            self.max_wavelength = scaling_factors['max_wavelength']
            self.min_wavelength = scaling_factors['min_wavelength']
            self.max_angle = scaling_factors['max_angle']
            self.min_angle = scaling_factors['min_angle']
        else:
            self.scaling_factor = np.mean(np.abs(self.fields))
            self.src_data_scaling_factor = np.max(np.abs(self.src))*0.125

        self.fields /= self.scaling_factor

        self.wavelengths_normalized = (self.wavelengths - self.min_wavelength) / (self.max_wavelength - self.min_wavelength)
        self.angles_normalized = (self.angles - self.min_angle) / (self.max_angle - self.min_angle)

        # Initialize sample weights
        if indices is not None:
            # Start with uniform weights
            self.sample_weights = np.ones(len(self.patterns), dtype=np.float32) / len(self.patterns)
        else:
            self.sample_weights = None

        print("src_data_scaling_factor:", self.src_data_scaling_factor)
        print("field_scaling_factor:", self.scaling_factor)
        
        print("patterns.shape:", self.patterns.shape, self.patterns.dtype)
        print("fields.shape:", self.fields.shape, self.fields.dtype)
        print("src.shape:", self.src.shape, self.src.dtype)
        print("wavelengths.shape:", self.wavelengths.shape, self.wavelengths.dtype)

    def __len__(self):
        return len(self.patterns)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            'structure': self.patterns[idx],
            'field': self.fields[idx],
            'src': self.src[idx],
            'wavelength': self.wavelengths[idx],
            'wavelength_normalized': self.wavelengths_normalized[idx],
            'angle': self.angles[idx],
            'angle_normalized': self.angles_normalized[idx],
        }

        return sample

    def get_scaling_factors(self):
        return {
            'field_scaling_factor': self.scaling_factor,
            'src_data_scaling_factor': self.src_data_scaling_factor,
            'max_wavelength': self.max_wavelength,
            'min_wavelength': self.min_wavelength,
            'max_angle': self.max_angle,
            'min_angle': self.min_angle
        }

    def get_sample_weights(self):
        return self.sample_weights

    def update_sample_weights(self, new_weights):
        """Update sample weights in the dataset."""
        self.sample_weights = new_weights.astype(np.float32)

    def get_wavelength_bin_indices(self):
        return self.wavelength_bin_indices

    def get_wavelength_bins(self):
        return self.wavelength_bins
