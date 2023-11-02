import timm
import torch.nn as nn


vgg = [#"vgg11",
     "vgg11_bn", 
     "vgg13", 
     "vgg13_bn",
     "vgg16", 
     "vgg16_bn",
     "vgg19",
     "vgg19_bn"]

# ResNet モデル
resnet = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

# Wide ResNet モデル
wide_resnet = ["wide_resnet50_2", "wide_resnet101_2"]

# EfficientNet モデル
efficientnet = ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3","efficientnet_b4", "efficientnet_b5"]





# Vision Transformer モデル
vit = ["vit_base_patch8_224", "vit_base_patch16_224", "vit_base_patch32_224", "vit_large_patch16_224","vit_large_patch32_224"]



available_model_timm = [vgg, resnet, wide_resnet, efficientnet,  vit]


#ファインチューニング用クラス
class FineTuning(nn.Module):
    def __init__(self, model, num_features,num_classes):
        super().__init__()
        self.model = model
        # 最終層の出力ユニットを追加する
        if num_features < 1024:
            if num_features < 512:
                one_liner_out = num_features
            else:
                one_liner_out = 512
        elif num_features < 2048:
            one_liner_out = int(num_features / 2)
        else:
            one_liner_out = 1024


        self.add_layer = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=one_liner_out),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=one_liner_out, out_features=num_classes),
        )        
        

    def forward(self, x):
        x = self.model(x)
        x = self.add_layer(x)
        return x




