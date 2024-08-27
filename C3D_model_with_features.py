import torch
import torch.nn as nn
import torch.nn.functional as F
from fds import FDS


            
class C3D(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, use_feature=False, use_buffer = True, cls = 46):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 64)
        self.fc8 = nn.Linear(64, 1)

        # if use_buffer:
        self.FDS = FDS(
                feature_dim= 8192 , bucket_num = cls, bucket_start=0,
                start_update=0, start_smooth=1, kernel="gaussian", ks=3, sigma=1, momentum=0.5
            )

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()       

        self.__init_weight()
        self.use_feature = use_feature
        self.use_buffer = use_buffer

    def forward(self, x, epoch, targets=None):

        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        
        x = x.reshape(-1, 8192)
        feature = x
        if self.use_feature:
            # if self.training:
            #     feature = self.ks.new_smooth(x,targets)
            # else:
            #     feature = self.ks.test_smooth(x)
            x = self.relu(self.fc6(x))
            x = self.dropout(x)
            # feature = x
            x = self.fc7(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc8(x)

            if self.training and self.use_buffer:
                
                y = self.FDS.smoothed_mean_last_epoch
                y = self.relu(self.fc6(y))
                y = self.dropout(y)
                y = self.fc7(y)
                y = self.relu(y)
                y = self.dropout(y)
                y = self.fc8(y)

                return feature,x.squeeze(-1),y.squeeze(-1)
            else:
                return feature,x.squeeze(-1)



               
        else:
            x = self.relu(self.fc6(x))
            x = self.dropout(x)
            x = self.fc7(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc8(x)
            return x.squeeze(-1)



    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



if __name__ == "__main__":
    inputs = torch.rand(3, 3, 16, 112, 112)
    net = C3D(pretrained = True)

    p= net.forward(inputs)
    # print(feature.size())
    print(p)

    