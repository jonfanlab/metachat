import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    def __init__(self, num_features):
        super(FiLM, self).__init__()
        self.num_features = num_features
        self.film_layer = nn.Linear(2, num_features * 2)

    def forward(self, x, wavelength, angle):
        # wavelength shape: (batch_size, 1)
        # angle shape: (batch_size, 1)
        # x shape: (batch_size, num_features, height, width)
        
        # Concatenate conditions
        conditions = torch.cat([wavelength.unsqueeze(1), angle.unsqueeze(1)], dim=1)
        
        # Generate FiLM parameters
        params = self.film_layer(conditions)
        gamma, beta = torch.chunk(params, 2, dim=1)
        
        # Reshape for broadcasting
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        
        return gamma * x + beta


class UNet(nn.Module):
    '''
    UNet architecture
    '''

    def __init__(self, net_depth, block_depth, init_num_kernels, input_channels, output_channels, dropout):

      super(UNet, self).__init__()

      self.input_channels = input_channels  # Should be 3 (1 structure + 2 source)
      self.output_channels = output_channels

      self.init_num_kernels = init_num_kernels
      self.block_depth = block_depth
      self.net_depth = net_depth

      self.conv_layers = nn.ModuleList([])
      self.bn_layers = nn.ModuleList([])
      self.dropout = nn.Dropout2d(p=dropout)
      self.film_layers = nn.ModuleList([])

      # Encoder path (no FiLM layers)
      for d in range(self.net_depth):

        curr_kernels = self.init_num_kernels * 2**d

        for b in range(self.block_depth):

          #Account for first layer taking in the low-channel input
          if(d==0 and b==0):
            self.conv_layers.append(nn.Conv2d(self.input_channels,curr_kernels, kernel_size=3, padding=1))

          else:
            if(b==0):
              prev_kernels = self.init_num_kernels * 2**(d-1)
              self.conv_layers.append(nn.Conv2d(prev_kernels,curr_kernels, kernel_size=3, padding=1))

            else:
              self.conv_layers.append(nn.Conv2d(curr_kernels,curr_kernels, kernel_size=3, padding=1))

          self.bn_layers.append(nn.BatchNorm2d(curr_kernels))

      # Decoder path (with FiLM layers)
      for d in range(self.net_depth-1):
        curr_kernels = self.init_num_kernels * 2**(self.net_depth-d-2)

        for b in range(self.block_depth):

          #Take care of the extra channels from concatenating the upsampled result from the previous block

          if(b==0):
            self.conv_layers.append(nn.Conv2d(curr_kernels*3,curr_kernels, kernel_size=3, padding=1))
          else:
            self.conv_layers.append(nn.Conv2d(curr_kernels,curr_kernels, kernel_size=3, padding=1))

          self.bn_layers.append(nn.BatchNorm2d(curr_kernels))
          self.film_layers.append(FiLM(curr_kernels))

      #One last convolution layer, for calculating the output from the processed input
      self.conv_layers.append(nn.Conv2d(self.init_num_kernels,self.output_channels, kernel_size=3, padding=1))

    #   # Add weight initialization
    #   def init_weights(m):
    #       if isinstance(m, nn.Conv2d):
    #           nn.init.kaiming_normal_(m.weight)
    #           if m.bias is not None:
    #               nn.init.zeros_(m.bias)
    #       elif isinstance(m, nn.Linear):
    #           nn.init.xavier_uniform_(m.weight)
    #           if m.bias is not None:
    #               nn.init.zeros_(m.bias)

    #   self.apply(init_weights)

    #   # Initialize FiLM layers with small weights
    #   for film in self.film_layers:
    #       nn.init.uniform_(film.film_layer.weight, -0.01, 0.01)
    #       nn.init.zeros_(film.film_layer.bias)

    def forward(self, x, wavelength, angle):

      batch_size, _, height, width = x.shape

      shortcut_list = []  #stores shortcut layers
      conv_counter = 0
      bn_counter = 0
      film_counter = 0

      # Encoder path (no FiLM)
      for d in range(self.net_depth):

        for b in range(self.block_depth):
          features = self.conv_layers[conv_counter](x if (d == 0 and b == 0) else features)
          conv_counter+=1

          #The first convolution result of each block is added as a residual connection at the end
          if(b==0):
            res_connection = features

          features = self.bn_layers[bn_counter](features)
          bn_counter+=1

          if(b==self.block_depth-1):
            features = features + res_connection

          features = F.leaky_relu(features)
          features = self.dropout(features)

          if(b==self.block_depth-1 and d!=self.net_depth-1):
            shortcut_list.append(features)

        if(d != self.net_depth-1):
          features = F.max_pool2d(input=features, kernel_size=2)

      # Decoder path (with FiLM)
      for d in range(self.net_depth-1):

        #Convention: (N,C,H,W)
        features = torch.cat(( F.interpolate(features, scale_factor=(2,2), mode='nearest') , shortcut_list.pop() ),dim=1)

        for b in range(self.block_depth):

          features = self.conv_layers[conv_counter](features)
          conv_counter+=1

          if(b==0):
            res_connection = features

          features = self.bn_layers[bn_counter](features)
          features = self.film_layers[film_counter](features, wavelength, angle)
          bn_counter+=1
          film_counter+=1

          if(b==self.block_depth-1):
            features = features + res_connection

          features = F.leaky_relu(features)
          features = self.dropout(features)

      output = self.conv_layers[conv_counter](features)

      return output