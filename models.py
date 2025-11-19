
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

# %%
# Define a basic Residual Block
class BasicBlock(nn.Module):
    expansion = 1  # Output channels same as input
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # Residual connection
        out = F.relu(out)
        return out

# Define ResNet Architecture
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):  # Default CIFAR-10 classification
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Define ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
       
        x = self.fc(x)
        return x

# Instantiate ResNet-18 Model
def resnet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


class ResNetCTC(nn.Module):
    def __init__(self, num_classes=10):
        """
        num_classes: number of actual distinct labels W (excluding blank).
        We add +1 for blank token automatically.
        """
        super(ResNetCTC, self).__init__()
        # Load ResNet-18 backbone
        resnet = models.resnet18(pretrained=False)

        # Modify first conv layer to accept 1 channel
        resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Keep only convolutional blocks
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        # Pool on height axis only (H' → 1)
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, None))  # (H', T') → (1, T')

        # Linear classifier for each time step
        self.fc = nn.Linear(512, num_classes + 1)  # +1 for CTC blank

    def forward(self, x):
        """
        x: (B, 1, H, T), e.g., (5, 1, 60, 820)
        """
        features = self.backbone(x)  # (B, 512, H', T')

        pooled = self.spatial_pool(features)  # (B, 512, 1, T')
        pooled = pooled.squeeze(2)           # (B, 512, T')

        # Transpose to (T', B, 512)
        pooled = pooled.permute(2, 0, 1)

        out = self.fc(pooled)  # (T', B, num_classes + 1)

        return out


class ResNetWindowLSTMCTC(nn.Module):
    """
    Input:  x (B, C, H, T)
    Steps:
      - Slide windows along time (size=window_size, step=window_step)
      - Encode each window with ResNet18 -> (B, 512)
      - Stack across windows -> (B, Tw, 512)
      - RNN over window sequence -> (B, Tw, rdim)
      - CTC head -> (Tw, B, V+1)  (last index is blank)
      - (optional) aux per-window head -> (B, Tw, V)

    Args:
      num_classes: V (without blank)
      in_ch: input channels
      window_size, window_step: temporal windowing
      rnn_hidden: LSTM hidden size
      rnn_layers: number of LSTM layers
      bidir: bidirectional LSTM
      use_aux: if True, also output per-window aux logits (B, Tw, V)
      weights: torchvision weights enum or None (use None if in_ch != 3)
    """
    def __init__(self,
                 num_classes: int,
                 in_ch: int = 4,
                 window_size: int = 100,
                 window_step: int = 50,
                 rnn_hidden: int = 256,
                 rnn_layers: int = 1,
                 bidir: bool = True,
                 dropout: float = 0.1,
                 use_aux: bool = False,
                 weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.in_ch = in_ch
        self.window_size = window_size
        self.window_step = window_step
        self.use_aux = use_aux

        # ----- ResNet18 backbone -----
        # Use weights only if in_ch==3
        m = resnet18(weights=weights if in_ch == 3 else None)
        # patch conv1 for custom channels
        m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # if you wanted to "inflate" ImageNet weights to extra channels, do it here.

        # cut off the classifier; keep feature extractor
        self.backbone = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))   # (B,512,1,1)
        feat_dim = 512

        # ----- RNN over window features -----
        rdim = rnn_hidden * (2 if bidir else 1)
        self.rnn = nn.LSTM(input_size=feat_dim,
                           hidden_size=rnn_hidden,
                           num_layers=rnn_layers,
                           batch_first=True,
                           bidirectional=bidir,
                           dropout=(dropout if rnn_layers > 1 else 0.0))

        # ----- Heads -----
        self.ctc_fc = nn.Linear(rdim, num_classes + 1)     # +1 for CTC blank
        self.aux_fc = nn.Linear(rdim, num_classes) if use_aux else None

    @staticmethod
    def _make_windows(x, win, step):
        """
        x: (B, C, H, T) -> list of windows (B, C, H, win)
        """
        B, C, H, T = x.shape
        if T < win:
            raise ValueError(f"T={T} < window_size={win}")
        starts = range(0, T - win + 1, step)
        return [x[:, :, :, s:s+win] for s in starts], len(list(starts))

    def encode_window(self, w):
        """
        w: (B, C, H, Win) -> (B, 512)
        """
        f = self.backbone(w)               # (B,512,h',w')
        p = self.spatial_pool(f).flatten(1)  # (B,512)
        return p

    def forward(self, x):
        """
        x: (B, C, H, T)
        Returns:
          ctc_logits: (Tw, B, V+1)
          aux_logits: (B, Tw, V)  if use_aux else None
          Tw: int
        """
        B, C, H, T = x.shape
        windows, Tw = self._make_windows(x, self.window_size, self.window_step)  # Tw windows

        # Encode all windows with a single pass (concat along batch dim)
        if Tw == 1:
            enc = self.encode_window(windows[0])              # (B,512)
        else:
            enc = self.encode_window(torch.cat(windows, dim=0))  # (Tw*B,512)

        # reshape to (B, Tw, 512)
        feats = enc.view(Tw, B, -1).transpose(0, 1)           # (B,Tw,512)

        # RNN over the window sequence
        z, _ = self.rnn(feats)                                # (B,Tw,rdim)

        # Heads
        ctc_logits = self.ctc_fc(z).transpose(0, 1)           # (Tw,B,V+1)
        aux_logits = self.aux_fc(z) if self.aux_fc is not None else None  # (B,Tw,V) or None

        return ctc_logits, aux_logits, Tw


class TemporalResNetCTC(nn.Module):
    def __init__(self, num_classes, stride_time=(2, 2, 2, 2), in_ch =4):
        """
        Args:
            num_classes: number of labels + 1 (for CTC blank)
            stride_time: tuple of length 4, stride along time-axis for each ResNet stage
        """
        super().__init__()
        self.in_channels = 64

        # ----- 1. Initial Conv Layer -----
        # You can change kernel_size to increase temporal window (7 -> larger context)
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=1)  
        # ↑ stride=(1,2): only downsample time-axis slightly here

        # ----- 2. ResNet Layers -----
        self.layer1 = self._make_layer(64, 2, stride=(1, stride_time[0]))  # Stage 1
        self.layer2 = self._make_layer(128, 2, stride=(1, stride_time[1])) # Stage 2
        self.layer3 = self._make_layer(256, 2, stride=(1, stride_time[2])) # Stage 3
        self.layer4 = self._make_layer(512, 2, stride=(1, stride_time[3])) # Stage 4

        # ----- 3. Adaptive Average Pool (Height only) -----
        self.avgpool_height = nn.AdaptiveAvgPool2d((1, None))  
        # keeps time-axis intact but pools height to 1

        # ----- 4. Linear Decoder for CTC -----
        self.fc = nn.Linear(512, num_classes + 1)

    def _make_layer(self, out_channels, blocks, stride):
        """Create a ResNet layer with temporal stride control"""
        downsample = None
        if stride != (1, 1) or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = [BasicBlock(self.in_channels, out_channels, stride=stride, downsample=downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Input: x -> (B, 1, H, T)
        Output: logits for CTC -> (T', B, num_classes + 1)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Pool height only
        x = self.avgpool_height(x)  # (B, 512, 1, T')
        x = x.squeeze(2)            # (B, 512, T')

        # Permute for CTC: (T', B, C)
        x = x.permute(2, 0, 1)      # (T', B, 512)
        logits = self.fc(x)         # (T', B, num_classes + 1)

        return logits


class ResNetWindowCTC(nn.Module):
    def __init__(self, in_ch =4, num_classes=10, window_size=100, window_step=50):
        super().__init__()
        self.num_classes = num_classes
        self.window_size = window_size
        self.window_step = window_step
        self.in_ch = in_ch

        # Use ResNet-18 backbone
        resnet = models.resnet18(pretrained=False)
        #resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        resnet.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))  # Pool H x W → (1, 1)
        self.fc = nn.Linear(512, num_classes + 1)  # +1 for CTC blank

    def encode_window(self, x):
        """
        x: (B, 1, H, window_size)
        """
        features = self.backbone(x)         # (B, 512, H', T')
        pooled = self.spatial_pool(features)  # (B, 512, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (B, 512)
        return pooled

    def forward(self, x):
        """
        x: (B, 1, H, T)
        """
        B, C, H, T = x.shape
        windows = []

        # Create windows
        for start in range(0, T - self.window_size + 1, self.window_step):
            end = start + self.window_size
            window = x[:, :, :, start:end]  # (B, 1, H, window_size)
            windows.append(window)

        # Stack all windows: (num_windows * B, 1, H, window_size)
        windows = torch.cat(windows, dim=0)

        # Encode each window
        feats = self.encode_window(windows)  # (num_windows * B, 512)

        # Reshape to (num_windows, B, feature_dim)
        num_windows = len(range(0, T - self.window_size + 1, self.window_step))
        feats = feats.view(num_windows, B, -1)

        # Linear decode
        logits = self.fc(feats)  # (num_windows, B, num_classes + 1)

        return logits


# %%
def greedy_decode(log_probs, label_dic_reverse, blank_index):
    max_probs = log_probs.argmax(dim=2)
    predictions = max_probs.transpose(0, 1)
    decoded_strings = []

    for seq in predictions:
        seq = seq.cpu().numpy().tolist()
        prev = -1
        decoded = []
        for s in seq:
            if s != prev and s != blank_index:
                decoded.append(label_dic_reverse[s])
            prev = s
        decoded_strings.append(" ".join(decoded))
    return decoded_strings


def greedy_decode_blank(log_probs, label_dic_reverse, blank_index):
    max_probs = log_probs.argmax(dim=2)
    predictions = max_probs.transpose(0, 1)
    decoded_strings = []
    decoded_strings_raw = []

    for seq in predictions:
        seq = seq.cpu().numpy().tolist()
        prev = -1
        decoded = []
        decoded_raw = []
        for s in seq:
            if s != prev and s != blank_index:
                decoded.append(label_dic_reverse[s])
            prev = s
            if s == blank_index:
                decoded_raw.append("#")
            if s != blank_index:
                decoded_raw.append(label_dic_reverse[s])
        
        decoded_strings.append(" ".join(decoded))
        decoded_strings_raw.append(" ".join(decoded_raw))

    return decoded_strings, decoded_strings_raw


def wer(ref, hyp):
    """
    ref: list of words (ground truth)
    hyp: list of words (prediction)
    """
    # Initialize matrix
    d = np.zeros((len(ref)+1, len(hyp)+1), dtype=np.uint32)
    for i in range(len(ref)+1):
        d[i][0] = i
    for j in range(len(hyp)+1):
        d[0][j] = j

    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            if ref[i-1] == hyp[j-1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(
                d[i-1][j] + 1,      # deletion
                d[i][j-1] + 1,      # insertion
                d[i-1][j-1] + cost  # substitution
            )
    return d[len(ref)][len(hyp)] / float(len(ref))



def ctc_loss_weighted(logits_TBV, tgt_list, device, blank_index):
    """
    logits_TBV: (T', B, V+1) raw logits from model
    tgt_list  : list of 1D LongTensors (targets per sample, no blanks)
    blank_index: index used for blank (usually V)
    returns scalar loss (weighted mean over batch) and per-sample losses
    """
    Tprime, B, _ = logits_TBV.shape

    # CTC expects log-probs FP32
    logp = logits_TBV.log_softmax(2).float()
    input_lengths  = torch.full((B,), Tprime, dtype=torch.long, device=device)
    target_lengths = torch.tensor([t.numel() for t in tgt_list], dtype=torch.long, device=device)
    targets_concat = (torch.cat([t.to(device) for t in tgt_list])
                      if B > 0 else torch.empty(0, dtype=torch.long, device=device))

    ctc = nn.CTCLoss(blank=blank_index, zero_infinity=True, reduction='none')
    per_sample = ctc(logp, targets_concat, input_lengths, target_lengths)  # (B,)
    return per_sample

def sample_weight_from_target_ids(tgt_ids_1d: torch.Tensor,
                                  class_rarity: torch.Tensor,
                                  agg="max",
                                  empty_default=1.0,
                                  clip=(0.5, 10.0)) -> torch.Tensor:
    """
    tgt_ids_1d: 1D LongTensor (no blanks, no 'none')
    class_rarity: Tensor[V]
    returns: scalar tensor weight
    """
    if tgt_ids_1d.numel() == 0:
        w = torch.tensor(empty_default, dtype=torch.float32)
    else:
        vals = class_rarity[tgt_ids_1d]
        if   agg == "mean": w = vals.mean()
        elif agg == "sum":  w = (vals.sum() / tgt_ids_1d.numel())
        else:               w = vals.max()
    lo, hi = clip
    return torch.clamp(w, lo, hi)




class DualResNetClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(DualResNetClassifier, self).__init__()

        # ResNet18 for First Input Modality
        self.resnet1 = models.resnet18(num_classes=10)
        self.resnet1.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Change input channels to 1
        self.resnet1.fc = nn.Identity()  # Remove final FC layer

        # ResNet18 for Second Input Modality
        self.resnet2 = models.resnet18(num_classes=10)
        self.resnet2.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Change input channels to 4
        self.resnet2.fc = nn.Identity()  # Remove final FC layer

        # Batch Normalization for Normalizing Feature Vectors
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)

        # Fully Connected Layer for Classification
        self.fc = nn.Linear(512 + 512, num_classes)  # Combine ResNet1 (512) + ResNet2 (512)

    def normalize(self, x):
        """Normalize the feature map to the range [0, 1]"""
        min_val = torch.min(x, dim=-1, keepdim=True)[0]
        max_val = torch.max(x, dim=-1, keepdim=True)[0]
        return (x - min_val) / (max_val - min_val)
    
    def forward(self, x1, x2):
        # Extract Features
        feat1 = self.resnet1(x1)  # (B, 512, H', W')
        feat2 = self.resnet2(x2)  # (B, 512, H', W')

        # Flatten Features
        feat1 = feat1.view(feat1.size(0), -1)  # (B, 512)
        feat2 = feat2.view(feat2.size(0), -1)  # (B, 512)

        # Normalize Features
        feat1 = self.bn1(feat1)  # Batch normalization
        feat2 = self.bn2(feat2)  # Batch normalization

        # Alternative: L2 Normalization (Uncomment if needed)
        # feat1 = F.normalize(feat1, p=2, dim=1)  # L2 Normalization
        # feat2 = F.normalize(feat2, p=2, dim=1)

        feat1 = self.normalize(feat1)  # (B, 512)
        feat2 = self.normalize(feat2)  # (B, 512)

        # Concatenate Features
        fused_features = torch.cat([feat1, feat2], dim=1)  # (B, 1024)

        fused_features = self.normalize(fused_features)   # (B, 1024)

        # Fully Connected Layer for Classification
        logits = self.fc(fused_features)  # (B, num_classes)

        return logits
    


class Imu1dImage2dModel(nn.Module):
    def __init__(self, num_classes=10):
        super(Imu1dImage2dModel, self).__init__()

        # 1D CNN for IMU data

        self.imu_cnn = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Pool to 1 to reduce to [batch, 128, 1]
        )

        # ResNet18 for Second Input Modality
        self.resnet2 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet2.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Change input channels to 4
        self.resnet2.fc = nn.Identity()  # Remove final FC layer

        # Batch Normalization for Normalizing Feature Vectors
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(512)

        # Fully Connected Layer for Classification
        self.fc = nn.Linear(128 + 512, num_classes)  # Combine ResNet1 (512) + ResNet2 (512)

    def normalize(self, x):
        """Normalize the feature map to the range [0, 1]"""
        min_val = torch.min(x, dim=-1, keepdim=True)[0]
        max_val = torch.max(x, dim=-1, keepdim=True)[0]
        return (x - min_val) / (max_val - min_val)
    
    def forward(self, x1, x2):
        # Extract Features
        B, C, H, W =  x1.shape # Input 1: 1D CNN for sequential IMU data (shape: [B, C, T]).
        x1 = x1.reshape(B, H, W) # Height 3 - Channel, W - Time
        feat1 = self.imu_cnn(x1)
        feat1 = feat1.view(feat1.size(0), -1)   # Flatten to [batch, 128]

        feat2 = self.resnet2(x2)  # (B, 512, H', W')
        #print("AAA", feat1.shape, feat2.shape)

        # Flatten Features
        feat1 = feat1.view(feat1.size(0), -1)  # (B, 512)
        feat2 = feat2.view(feat2.size(0), -1)  # (B, 512)
        #print("BBB", feat1.shape, feat2.shape)

        # Normalize Features
        feat1 = self.bn1(feat1)  # Batch normalization
        feat2 = self.bn2(feat2)  # Batch normalization

        # Alternative: L2 Normalization (Uncomment if needed)
        # feat1 = F.normalize(feat1, p=2, dim=1)  # L2 Normalization
        # feat2 = F.normalize(feat2, p=2, dim=1)

        feat1 = self.normalize(feat1)  # (B, 128)
        feat2 = self.normalize(feat2)  # (B, 512)

        # Concatenate Features
        fused_features = torch.cat([feat1, feat2], dim=1)  # (B, 640)

        fused_features = self.normalize(fused_features)   # (B, 640)

        # Fully Connected Layer for Classification
        logits = self.fc(fused_features)  # (B, num_classes)

        return logits


class IMUEncoder1D(nn.Module):
    """(B, C_imu, Win) -> (B, F_imu)"""
    def __init__(self, in_ch=6, feat=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 5, stride=2, padding=2),   nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, feat, 3, stride=1, padding=1), nn.BatchNorm1d(feat), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
    def forward(self, x):  # (B,C_imu,Win)
        return self.net(x)

class Imu1dImage2dCTC(nn.Module):
    """
    CTC-ready late-fusion:
      Inputs:
        x_img: (B, C_img, H, T)
        x_imu: (B, C_imu, T)  or (B, C_imu, 1, T)
      Windowing along T with (window_size, window_step).
      Output:
        logits: (Tw, B, V+1)  (+1 for CTC blank)
    """
    def __init__(self,
                 num_classes: int,            # V (no blank)
                 C_img: int = 3,
                 C_imu: int = 6,
                 window_size: int = 100,
                 window_step: int = 50,
                 imu_feat: int = 128,
                 fc_hidden: int = None,
                 use_pretrained_rgb: bool = True):
        super().__init__()
        self.window_size = window_size
        self.window_step = window_step
        self.num_classes = num_classes

        # --- Visual backbone (ResNet-18) ---
        # Use pretrained weights only if C_img==3
        #weights = ResNet18_Weights.DEFAULT if (use_pretrained_rgb and C_img == 3) else None
        m = resnet18()
        if C_img != 3:
            # replace conv1 to match channel count (no pretrained for non-3ch)
            m.conv1 = nn.Conv2d(C_img, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.backbone = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool,
                                      m.layer1, m.layer2, m.layer3, m.layer4)
        self.pool2d = nn.AdaptiveAvgPool2d((1,1))  # -> (B,512)
        vis_feat = 512

        # --- IMU branch ---
        self.imu = IMUEncoder1D(in_ch=C_imu, feat=imu_feat)

        # --- Fusion head to CTC vocab (+ blank) ---
        fused_dim = vis_feat + imu_feat
        out_dim = num_classes + 1  # +1 for blank
        if fc_hidden is None:
            self.head = nn.Linear(fused_dim, out_dim)
        else:
            self.head = nn.Sequential(
                nn.Linear(fused_dim, fc_hidden), nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(fc_hidden, out_dim)
            )

    @staticmethod
    def _starts(T, win, step):
        if T < win:
            raise ValueError(f"T ({T}) < window_size ({win})")
        return list(range(0, T - win + 1, step))

    def _encode_vis_win(self, x_win):  # (B,C_img,H,Win) -> (B,512)
        f = self.backbone(x_win)
        return self.pool2d(f).flatten(1)

    def forward(self, x_img, x_imu):
        """
        x_img: (B,C_img,H,T)
        x_imu: (B,C_imu,T)  or (B,C_imu,1,T)
        returns logits: (Tw,B,V+1)
        """
        if x_imu.dim() == 4 and x_imu.size(2) == 1:
            x_imu = x_imu.squeeze(2)  # (B,C_imu,T)

        B, C, H, T = x_img.shape
        assert x_imu.shape[0] == B and x_imu.shape[-1] == T, "x_img and x_imu must share B and T"

        starts = self._starts(T, self.window_size, self.window_step)
        Tw = len(starts)

        # Build all windows in batch (concat on batch dim)
        x_img_w = torch.cat([x_img[:, :, :, s:s+self.window_size] for s in starts], dim=0)  # (Tw*B,C_img,H,Win)
        x_imu_w = torch.cat([x_imu[:, :,      s:s+self.window_size] for s in starts], dim=0)  # (Tw*B,C_imu,Win)

        v = self._encode_vis_win(x_img_w)      # (Tw*B,512)
        u = self.imu(x_imu_w)                  # (Tw*B,imu_feat)
        z = torch.cat([v,u], dim=1)            # (Tw*B,512+imu_feat)

        logits_flat = self.head(z)             # (Tw*B,V+1)
        logits = logits_flat.view(Tw, B, -1)   # (Tw,B,V+1)  ← CTC-ready
        return logits