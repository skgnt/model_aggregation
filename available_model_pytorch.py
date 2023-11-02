import torch
import torch.nn as nn
import torchvision

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
efficientnet = ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
                "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7"]

# AlexNet モデル
alexnet = ["alexnet"]

# Inception モデル
inception = ["inception_v3"]

# GoogLeNet モデル
googlenet = ["googlenet"]

# Swin Transformer モデル
swin = ["swin_b", "swin_s", "swin_t", "swin_v2_b", "swin_v2_s", "swin_v2_t"]

# Vision Transformer モデル
vit = ["vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32"]


available_model_pytorch = [vgg, resnet, wide_resnet, efficientnet,alexnet, inception, 
                    googlenet, swin, vit]

def model_output_plus(model , model_name,num_classes,transform):
     if model_name in efficientnet:
          in_features=list(model.children())[-1][1].in_features
     elif model_name in alexnet:
          in_features=9216
     else:
          try:
               in_features = list(model.children())[-1][0].in_features
          except:
               in_features = list(model.children())[-1].in_features

     if in_features < 1024:
          if in_features < 512:
               one_liner_out = in_features
          else:
               one_liner_out = 512
     else:
          one_liner_out = 1024

     torch.manual_seed(0)

     add_layer = nn.Sequential(
          nn.Linear(in_features=in_features, out_features=one_liner_out),
          nn.ReLU(),
          nn.Dropout(p=0.5),
          nn.Linear(in_features=one_liner_out, out_features=num_classes),
     )

     


     # 3. 最終層の変更(指定されたモデルによって最終層の形式が異なるため)
     if model_name in vgg:
          model.classifier = add_layer
     elif model_name in resnet:
          model.fc = add_layer

     elif model_name in wide_resnet:
          model.fc = add_layer

     elif model_name in efficientnet:
          model.classifier = add_layer

     elif model_name in alexnet:
          model.classifier = add_layer
     elif model_name in inception:
          model.fc=add_layer
          model.aux_logits = False 
          transform = {"train": torchvision.transforms.Compose([
                                   torchvision.transforms.Resize((299, 299)),
                                   torchvision.transforms.RandomHorizontalFlip(),
                                   torchvision.transforms.RandomVerticalFlip(),
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.5,), (0.5,))
                    ]), "val":  torchvision.transforms.Compose([
                              torchvision.transforms.Resize((299, 299)),
                              torchvision.transforms.ToTensor(),
                              torchvision.transforms.Normalize((0.5,), (0.5,))
                    ])}
     

     elif model_name in googlenet:
          model.fc = add_layer

     elif model_name in swin:
          model.head = add_layer

     elif model_name in vit:
          model.heads = add_layer
     return model,transform
