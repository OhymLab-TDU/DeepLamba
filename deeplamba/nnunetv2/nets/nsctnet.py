import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import timm
from PIL import Image
import numpy as np
from timm.models.swin_transformer import SwinTransformerBlock
from nsct_gpu import myNSCTd, myNSCTr
import cupy as cp
from PIL import Image, ImageEnhance
from einops import rearrange, reduce
import torch.multiprocessing as mp
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

# 设置多进程启动方法 muti threads
mp.set_start_method('spawn', force=True)
    
class Nsct_swinTransformer(nn.Module):
    def __init__(
            self,
            in_channels: int,
            input_resolution: tuple,
            depth: int=2,
            levels: list=[2, 2],
            dfilt: str='pkva',
            pfilt: str='pyr',
            type: str='NSCT',
            original_shape: tuple=(1, 3, 224, 224),
            level_map: int=9,
        ):
        super().__init__()
        self.in_channels = in_channels
        self.input_resolution = input_resolution
        self.depth = depth
        self.levels = levels
        self.dfilt = dfilt
        self.pfilt = pfilt
        self.type = type
        self.original_shape = original_shape
        self.level_map = level_map

        self.linear = nn.Linear(level_map*in_channels+in_channels, level_map*in_channels)

        # self.conv1_D = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        # self.conv2_D = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1)


        self.swinT1 = nn.ModuleList(
            [SwinTransformerBlock(dim=level_map*in_channels+in_channels, input_resolution=input_resolution) for _ in range(depth)]
        )

        self.conv1x1 = nn.Conv2d(in_channels=level_map*in_channels+in_channels, out_channels=in_channels, kernel_size=1, stride=1)

        


    def CT(self, image, level_map_CT=9):
        B, C, H, W = image.shape

        # 使用多进程并行化
        with mp.Pool(processes=mp.cpu_count()) as pool:
            transformed_list = pool.starmap(self._process_image, [(image[b, c].detach(),) for b in range(B) for c in range(C)])
        
        transformed_list = [transformed_list[i * C:(i + 1) * C] for i in range(B)]
        transformed = cp.stack([cp.stack(batch, axis=0) for batch in transformed_list], axis=0)
        transformed = transformed.transpose(0, 2, 1, 3, 4).reshape(B, C*level_map_CT, H, W)

        transformed = torch.utils.dlpack.from_dlpack(transformed.toDlpack()) 
        return transformed.to('cpu')
    
    def _process_image(self, image):
        return self.nsctDec(image)
    
    
    def ICT(self, processed, original_shape, level_map_ICT=9):
            B, C, H, W = original_shape
            to_inverse = processed.reshape(B, level_map_ICT, C, H, W).permute(0, 2, 1, 3, 4)
            to_inverse_list = [[to_inverse[b, c].detach().tolist() for c in range(C)] for b in range(B)]

            result = cp.zeros((B, C, H, W))

            # 使用多进程并行化
            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = pool.starmap(self._process_inverse, [(to_inverse_list[b][c],) for b in range(B) for c in range(C)])
            
            for b in range(B):
                for c in range(C):
                    result[b, c] = results[b * C + c]

            result = torch.utils.dlpack.from_dlpack(result.toDlpack())
            return result.to('cpu').float()
    
    def _process_inverse(self, to_inverse):
        if not isinstance(to_inverse, cp.ndarray):
            to_inverse = cp.array(to_inverse)
        # 在子进程中初始化CUDA设备
        cp.cuda.Device().use()
        return self.nsctRec(to_inverse)
        
    
    def nsctDec(self, x):
        x = cp.asarray(x) # 将转换而成的numpy数组转换为cupy数组
        [Insp, Insct] = myNSCTd(x, self.levels, self.pfilt, self.dfilt, self.type)
        result = [Insct[0]]

        for sublist in Insct[1:]:
            if isinstance(sublist, list):
                for item in sublist:
                    result.append(item)
            else:
                result.append(sublist)
        result = cp.asarray(result)

        return result
    
    def nsctRec(self, x, second_layer_num=5):
        first_element = x[0]

        second_element = x[1:second_layer_num]
        third_element = x[second_layer_num:]

        first_element = cp.asarray(first_element)
        second_element = cp.asarray(second_element)
        third_element = cp.asarray(third_element)

        original_format = [first_element, second_element, third_element]
        result = myNSCTr(original_format, self.levels, self.pfilt, self.dfilt, self.type)

        return result



    def forward(self, x):
        # x1 = self.conv1_D(x)
        # x2 = self.conv2_D(x1)
        x2 = nn.ReLU()(x)
        x_skip = x2
        x_trans = self.CT(x2)
        x3 = torch.cat((x_trans, x_skip), dim=1)
        x3 = x3.permute(0, 2, 3, 1).contiguous()
        for block in self.swinT1:
            x3 = block(x3)
        x3 = self.linear(x3)
        y = x3.permute(0, 3, 1, 2).contiguous()
        y_skip = y
        y_trans = self.ICT(y, self.original_shape)
        y = torch.cat((y_trans, y_skip), dim=1)
        y = self.conv1x1(y)
        return y


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        #         self.depthwise_conv = nn.Conv2d(4 * dim, 4 * dim, kernel_size=1, groups=4 * dim)
        #         self.pointwise_conv = nn.Conv2d(4 * dim, 2 * dim, kernel_size=1)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // 2, W // 2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        # x = x.permute(0, 3, 1, 2)
        # x = self.depthwise_conv(x)
        # x = self.pointwise_conv(x)

        return x
    

class PatchExpand(nn.Module):
    """
    Reference: https://arxiv.org/pdf/2105.05537.pdf
    """

    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # B, C, H, W ==> B, H, W, C
        x = self.expand(x)
        B, H, W, C = x.shape

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        x = x.reshape(B, H * 2, W * 2, C // 4)

        return x

class FinalPatchExpand_X4(nn.Module):
    """
    Reference:
        - GitHub: https://github.com/HuCaoFighting/Swin-Unet/blob/main/networks/swin_transformer_unet_skip_expand_decoder_sys.py
        - Paper: https://arxiv.org/pdf/2105.05537.pdf
    """

    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        # self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # H, W = self.input_resolution
        x = x.permute(0, 2, 3, 1)  # B, C, H, W ==> B, H, W, C
        x = self.expand(x)
        B, H, W, C = x.shape
        # B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        # x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        x = x.reshape(B, H * self.dim_scale, W * self.dim_scale, self.output_dim)

        return x  #.permute(0, 3, 1, 2)

class nsctnet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            depth_level: list=[2, 2, 2, 1],
            input_shape: tuple=(4, 3, 224, 224),
        ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth_level = depth_level
        self.input_shape = input_shape

        self.patchembed = PatchEmbed2D(in_chans=in_channels)
        self.ConvEn =nn.ModuleDict({
            'conv1_3x3': nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            'conv1_1x1': nn.Conv2d(in_channels=96, out_channels=16, kernel_size=1, stride=1),
            'conv2_3x3': nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1),
            'conv2_1x1': nn.Conv2d(in_channels=192, out_channels=16, kernel_size=1, stride=1),
            'conv3_3x3': nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            'conv3_1x1': nn.Conv2d(in_channels=384, out_channels=16, kernel_size=1, stride=1),
            'conv4_3x3': nn.Conv2d(in_channels=768, out_channels=768, kernel_size=3, stride=1, padding=1),
            'conv4_1x1': nn.Conv2d(in_channels=768, out_channels=16, kernel_size=1, stride=1),
        })
        self.nsct_swin1 = nn.ModuleList(
            [Nsct_swinTransformer(in_channels=16,
                                   input_resolution=(input_shape[2]//4, input_shape[3]//4), 
                                   original_shape=(input_shape[0], 16, input_shape[2]//4, input_shape[3]//4))
                                   for _ in range(depth_level[0])
                                   ])
        
        self.patchmerge1 = PatchMerging2D(dim=96)

        self.nsct_swin2 = nn.ModuleList(
            [Nsct_swinTransformer(in_channels=16, 
                                  input_resolution=(input_shape[2]//8, input_shape[3]//8), 
                                  original_shape=(input_shape[0], 16, input_shape[2]//8, input_shape[3]//8))
                                  for _ in range(depth_level[1])
                                  ])
        self.patchmerge2 = PatchMerging2D(dim=192)
        self.nsct_swin3 = nn.ModuleList(
            [Nsct_swinTransformer(in_channels=16, 
                                  input_resolution=(input_shape[2]//16, input_shape[3]//16), 
                                  original_shape=(input_shape[0], 16, input_shape[2]//16, input_shape[3]//16))
                                  for _ in range(depth_level[2])
                                  ])
        self.patchmerge3 = PatchMerging2D(dim=384)
        self.nsct_swin4 = nn.ModuleList(
            [Nsct_swinTransformer(in_channels=16, 
                                  input_resolution=(input_shape[2]//32, input_shape[3]//32), 
                                  original_shape=(input_shape[0], 16, input_shape[2]//32, input_shape[3]//32))
                                  for _ in range(depth_level[3])
                                  ])

        self.ConvDe = nn.ModuleDict({
            'conv4_3x3' : nn.Conv2d(in_channels=768+16, out_channels=768, kernel_size=3, stride=1, padding=1),
            'conv3_1x1' : nn.Conv2d(in_channels=384+16, out_channels=384, kernel_size=1, stride=1),
            'conv3_3x3' : nn.Conv2d(in_channels=384+384, out_channels=384, kernel_size=3, stride=1, padding=1),
            'conv2_1x1' : nn.Conv2d(in_channels=192+16, out_channels=192, kernel_size=1, stride=1),
            'conv2_3x3' : nn.Conv2d(in_channels=192+192, out_channels=192, kernel_size=3, stride=1, padding=1),
            'conv1_1x1' : nn.Conv2d(in_channels=96+16, out_channels=96, kernel_size=1, stride=1),
            'conv1_3x3' : nn.Conv2d(in_channels=96+96, out_channels=96, kernel_size=3, stride=1, padding=1),
        })
        
        self.patchexpand3 = PatchExpand(input_resolution=(input_shape[2]//32, input_shape[3]//32), dim=768)
        self.patchexpand2 = PatchExpand(input_resolution=(input_shape[2]//16, input_shape[3]//16), dim=384)
        self.patchexpand1 = PatchExpand(input_resolution=(input_shape[2]//8, input_shape[3]//8), dim=192)
        self.patchexpand0 = FinalPatchExpand_X4(input_resolution=None, dim=96)

        self.convEnd = nn.Conv2d(in_channels=96, out_channels=out_channels, kernel_size=1, stride=1)

    
    def forward(self, x):
        x = self.patchembed(x)

        x1 = x.permute(0, 3, 1, 2).contiguous()
        x1 = self.ConvEn['conv1_3x3'](x1)
        x1_1 = self.ConvEn['conv1_1x1'](x1)
        x1 = nn.ReLU()(x1)
        x1_1 = nn.ReLU()(x1_1)
        for block in self.nsct_swin1:
            x1_1 = block(x1_1)
        
        x = self.patchmerge1(x)

        x2 = x.permute(0, 3, 1, 2).contiguous()
        x2 = self.ConvEn['conv2_3x3'](x2)
        x2_1 = self.ConvEn['conv2_1x1'](x2)
        x2 = nn.ReLU()(x2)
        x2_1 = nn.ReLU()(x2_1)
        for block in self.nsct_swin2:
            x2_1 = block(x2_1)
        
        x = self.patchmerge2(x)

        x3 = x.permute(0, 3, 1, 2).contiguous()
        x3 = self.ConvEn['conv3_3x3'](x3)
        x3_1 = self.ConvEn['conv3_1x1'](x3)
        x3 = nn.ReLU()(x3)
        x3_1 = nn.ReLU()(x3_1)
        for block in self.nsct_swin3:
            x3_1 = block(x3_1)

        x = self.patchmerge3(x)

        x4 = x.permute(0, 3, 1, 2).contiguous()
        x4 = self.ConvEn['conv4_3x3'](x4)
        x4_1 = self.ConvEn['conv4_1x1'](x4)
        x4 = nn.ReLU()(x4)
        x4_1 = nn.ReLU()(x4_1)
        for block in self.nsct_swin4:
            x4_1 = block(x4_1)
        
        y4 = torch.cat((x4_1, x4), dim=1)
        y4 = self.ConvDe['conv4_3x3'](y4)

        y3 = self.patchexpand3(y4)
        y3 =y3.permute(0, 3, 1, 2).contiguous()
        x33_1 = torch.cat((x3_1, x3), dim=1)
        x33_1 = self.ConvDe['conv3_1x1'](x33_1)
        y3 = torch.cat((x33_1, y3), dim=1)
        y3 = self.ConvDe['conv3_3x3'](y3)

        y2 = self.patchexpand2(y3)
        y2 = y2.permute(0, 3, 1, 2).contiguous()
        x22_1 = torch.cat((x2_1, x2), dim=1)
        x22_1 = self.ConvDe['conv2_1x1'](x22_1)
        y2 = torch.cat((x22_1, y2), dim=1)
        y2 = self.ConvDe['conv2_3x3'](y2)

        y1 = self.patchexpand1(y2)
        y1 = y1.permute(0, 3, 1, 2).contiguous()
        x11_1 = torch.cat((x1_1, x1), dim=1)
        x11_1 = self.ConvDe['conv1_1x1'](x11_1)
        y1 = torch.cat((x11_1, y1), dim=1)
        y1 = self.ConvDe['conv1_3x3'](y1)

        y0 = self.patchexpand0(y1)
        y0 = y0.permute(0, 3, 1, 2).contiguous()
        y0 = self.convEnd(y0)

        return y0
    

def get_nsctnet_from_plans(
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        deep_supervision: bool = False,
):
    label_manager = plans_manager.get_label_manager(dataset_json)

    model = nsctnet(
        in_channels=num_input_channels,
        out_channels=label_manager.get_num_classes(),
        depth_level=[2,2,2,1],
        input_shape=(12, num_input_channels, 512, 512)
    )

    return model


if __name__ == '__main__':
    torch.manual_seed(1)
    input = torch.randn(1, 3, 224, 224)
    # model = Nsct_swinTransformer(in_channels=32, input_resolution=(512, 512), original_shape=(4, 32, 512, 512))
    model = nsctnet(in_channels=3, out_channels=2, input_shape=(1, 3, 224, 224))


    # 确保模型的参数有梯度
    for param in model.parameters():
        param.requires_grad = True

    # 前向传播
    output = model(input)
    print(model(input).shape)

    # 创建一个简单的损失函数
    target = torch.randn_like(output)  # 目标张量
    loss_fn = nn.MSELoss()
    loss = loss_fn(output, target)
    print("损失计算完成")

    # 反向传播
    loss.backward()

    print("梯度计算完成")

    # # 检查 conv3 层的梯度
    # if model.conv1_D.weight.grad is not None:
    #     print("conv1 层的权重梯度存在")
    # else:
    #     print("conv1 层的权重梯度不存在")

    # if model.conv1_D.bias.grad is not None:
    #     print("conv1 层的偏置梯度存在")
    # else:
    #     print("conv1 层的偏置梯度不存在")
    # print(model.conv1_D.weight.grad)

    # image_path = '/home/shizhe/New project/nsct_github/000000002149.jpg'  # 替换为你的图像路径
    # image = Image.open(image_path).convert('RGB')
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor()
    # ])
    # input_tensor = transform(image).unsqueeze(0)  # 增加batch维度

    # # 创建模型并进行前向传播
    # model = Nsct_swinTransformer(in_channels=3, input_resolution=(224, 224))
    # output = model(input_tensor)

    # # 输出特征图
    # print(output.shape)
    # output_image = output.squeeze(0).permute(1, 2, 0).detach().numpy()
    # output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())  # 归一化到0-1
    # output_image = (output_image * 255).astype(np.uint8)  # 转换为uint8类型

    # # 保存或显示输出特征图
    # output_image = Image.fromarray(output_image)
    # output_image.save('/home/shizhe/New project/nsct_github/feature_map/output_feature_map.jpg')  # 保存特征图
