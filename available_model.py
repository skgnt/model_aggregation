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


model_kind_list = [vgg, resnet, wide_resnet, efficientnet,alexnet, inception, 
                    googlenet, swin, vit]