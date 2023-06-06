import torch
import torch.nn as nn
import torch.nn.functional as F
 

class CRNN(nn.Module):
    def __init__(self, cnn_output_height, lstm_hidden_size, lstm_num_layers, num_classes):
        super(CRNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.mp1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.mp2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.mp3 = nn.MaxPool2d((1,2), stride=2)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.conv5_bn = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.conv6_bn = nn.BatchNorm2d(512)
        self.mp6 = nn.MaxPool2d((1,2), stride=2)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=(2, 2), stride=1, padding=0)
        
        self.lstm_input_size = cnn_output_height * 512
        self.lstm = nn.LSTM(
            self.lstm_input_size,
            lstm_hidden_size,
            lstm_num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(lstm_hidden_size * 2, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.conv1(x)
        out = F.leaky_relu(out)
        out = self.mp1(out)
        
        out = self.conv2(out)
        out = F.leaky_relu(out)
        out = self.mp2(out)
        
        out = self.conv3(out)
        out = F.leaky_relu(out)
        out = self.conv4(out)
        out = F.leaky_relu(out)
        out = self.mp3(out)
        
        out = self.conv5(out)
        out = self.conv5_bn(out)
        out = F.leaky_relu(out)
        
        out = self.conv6(out)
        out = self.conv6_bn(out)
        out = F.leaky_relu(out)
        out = self.mp6(out)

        out = self.conv7(out)
        out = F.leaky_relu(out)
        # out.shape [64, 512, 1, 7]
        out = out.permute(0, 3, 2, 1)
        out = out.reshape(batch_size, -1, self.lstm_input_size)
        out, _ = self.lstm(out)
        out = torch.stack(
            [F.log_softmax(self.fc(out[i]), dim=-1) for i in range(out.shape[0])]
        )
        return out