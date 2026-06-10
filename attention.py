import torch.nn as nn

class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=3, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

                                                  
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

                    
        self.mlp = nn.Sequential(
                                                                          
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),                                    
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        return x
