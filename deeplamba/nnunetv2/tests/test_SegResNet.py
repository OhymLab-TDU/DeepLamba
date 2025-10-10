from monai.networks.nets import SegResNet

import re
import time
import math
import numpy as np
from functools import partial
from typing import Optional, Union, Type, List, Tuple, Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
# from torchsummary import summary
from fvcore.nn import FlopCountAnalysis, flop_count, parameter_count_table
from thop import profile

def test_segresnet_deeplamba_style():
    """仿照Deeplamba的测试方式测试SegResNet模型"""
    print("=" * 60)
    print("SegResNet 测试 (仿照Deeplamba风格)")
    print("=" * 60)
    
    # 创建2D SegResNet模型，使用与Deeplamba相同的输出通道数
    model = SegResNet(
        spatial_dims=2,
        init_filters=64,
        in_channels=3,  # 与Deeplamba相同的输入通道数
        out_channels=13,  # 与Deeplamba相同的输出通道数
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1]
    )
    
    # 计算参数量 (仿照Deeplamba的方式)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')
    
    # 使用与Deeplamba完全相同的输入尺寸
    x = torch.rand(4, 3, 320, 320)
    
    if torch.cuda.is_available():
        x = x.cuda()
        model = model.cuda()
    
    print(f"Input shape: {x.shape}")
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # 计算FLOPs (仿照Deeplamba的方式)
    try:
        Gflops, unsupported = flop_count(model=model, inputs=(x,))
        print(f"FLOPs: {Gflops}")
    except Exception as e:
        print(f"FLOPs calculation failed: {e}")
        # 备用方案
        try:
            flops_thop, _ = profile(model, inputs=(x,), verbose=False)
            print(f"FLOPs (thop): {flops_thop / 1e9:.2f} G")
        except Exception as e2:
            print(f"thop calculation also failed: {e2}")
    
    # 参数详细表格 (仿照Deeplamba)
    print("\nParameter count table:")
    print(parameter_count_table(model))
    
    return model, total_params

def test_segresnet_2d():
    """测试2D SegResNet模型"""
    print("=" * 60)
    print("测试 2D SegResNet 模型")
    print("=" * 60)
    
    # 创建2D模型
    model = SegResNet(
        spatial_dims=2,
        init_filters=64,
        in_channels=1,
        out_channels=2,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1]
    )
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f'总参数量: {total_params:,}')
    print(f'参数大小: {total_params * 4 / 1024 / 1024:.2f} MB')
    
    # 使用与Deeplamba相同的空间尺寸，但调整通道数
    x = torch.rand(4, 1, 320, 320)  # 使用320x320尺寸
    
    if torch.cuda.is_available():
        x = x.cuda()
        model = model.cuda()
        print("使用GPU进行测试")
    else:
        print("使用CPU进行测试")
    
    # 测试前向传播
    print(f"输入形状: {x.shape}")
    output = model(x)
    print(f"输出形状: {output.shape}")
    
    # 计算FLOPs
    try:
        flops = FlopCountAnalysis(model, x)
        print(f"FLOPs: {flops.total() / 1e9:.2f} G")
    except Exception as e:
        print(f"FLOPs计算失败: {e}")
    
    # 使用thop计算FLOPs（备用方案）
    try:
        flops_thop, params_thop = profile(model, inputs=(x,), verbose=False)
        print(f"FLOPs (thop): {flops_thop / 1e9:.2f} G")
        print(f"参数量 (thop): {params_thop:,}")
    except Exception as e:
        print(f"thop计算失败: {e}")
    
    # 详细参数表
    print("\n参数详细信息:")
    print(parameter_count_table(model))

def test_segresnet_3d():
    """测试3D SegResNet模型"""
    print("\n" + "=" * 60)
    print("测试 3D SegResNet 模型")
    print("=" * 60)
    
    # 创建3D模型
    model = SegResNet(
        spatial_dims=3,
        init_filters=64,
        in_channels=1,
        out_channels=2,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1]
    )
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f'总参数量: {total_params:,}')
    print(f'参数大小: {total_params * 4 / 1024 / 1024:.2f} MB')
    
    # 3D输入，使用合理的尺寸避免内存问题
    x = torch.rand(2, 1, 64, 160, 160)  # 减小batch size和深度，但保持较大的H,W
    
    if torch.cuda.is_available():
        x = x.cuda()
        model = model.cuda()
        print("使用GPU进行测试")
    else:
        print("使用CPU进行测试")
    
    # 测试前向传播
    print(f"输入形状: {x.shape}")
    output = model(x)
    print(f"输出形状: {output.shape}")
    
    # 计算FLOPs
    try:
        flops = FlopCountAnalysis(model, x)
        print(f"FLOPs: {flops.total() / 1e9:.2f} G")
    except Exception as e:
        print(f"FLOPs计算失败: {e}")
    
    # 使用thop计算FLOPs（备用方案）
    try:
        flops_thop, params_thop = profile(model, inputs=(x,), verbose=False)
        print(f"FLOPs (thop): {flops_thop / 1e9:.2f} G")
        print(f"参数量 (thop): {params_thop:,}")
    except Exception as e:
        print(f"thop计算失败: {e}")
    
    # 详细参数表
    print("\n参数详细信息:")
    print(parameter_count_table(model))

def compare_with_deeplamba():
    """与Deeplamba进行直接对比"""
    print("\n" + "=" * 60)
    print("SegResNet vs Deeplamba 对比测试")
    print("=" * 60)
    
    print("测试配置:")
    print("- 输入尺寸: (4, 3, 320, 320)")
    print("- 输出通道: 13")
    print("- 空间维度: 2D")
    print()
    
    # SegResNet配置
    segresnet_model = SegResNet(
        spatial_dims=2,
        init_filters=64,
        in_channels=3,
        out_channels=13,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1]
    )
    
    # 计算SegResNet参数量
    segresnet_params = sum(p.numel() for p in segresnet_model.parameters())
    
    # 创建相同的输入
    x = torch.rand(4, 3, 320, 320)
    
    if torch.cuda.is_available():
        x = x.cuda()
        segresnet_model = segresnet_model.cuda()
    
    # 测试SegResNet
    print("SegResNet 结果:")
    print("-" * 30)
    print(f"参数量: {segresnet_params:,}")
    print(f"模型大小: {segresnet_params * 4 / 1024 / 1024:.2f} MB")
    
    segresnet_output = segresnet_model(x)
    print(f"输出形状: {segresnet_output.shape}")
    
    # 计算SegResNet FLOPs
    try:
        flops_thop, _ = profile(segresnet_model, inputs=(x,), verbose=False)
        segresnet_flops = flops_thop / 1e9
        print(f"FLOPs: {segresnet_flops:.2f} G")
    except Exception as e:
        print(f"FLOPs计算失败: {e}")
        segresnet_flops = 0
    
    # 这里可以添加Deeplamba的对比（如果需要）
    print(f"\nDeeplamba 参考值 (来自原始测试):")
    print("-" * 30)
    print("参数量: [需要运行Deeplamba获取]")
    print("FLOPs: [需要运行Deeplamba获取]")
    
    return {
        'segresnet_params': segresnet_params,
        'segresnet_flops': segresnet_flops
    }

def test_segresnet_variants():
    """测试不同配置的SegResNet模型"""
    print("\n" + "=" * 60)
    print("测试不同配置的 SegResNet 模型")
    print("=" * 60)
    
    configs = [
        {
            'name': 'SegResNet-Small (init_filters=32)',
            'spatial_dims': 2,
            'init_filters': 32,
            'in_channels': 3,
            'out_channels': 13,
            'blocks_down': [1, 2, 2, 4],
            'blocks_up': [1, 1, 1],
            'input_shape': (4, 3, 320, 320)  # 使用与Deeplamba相同的尺寸
        },
        {
            'name': 'SegResNet-Standard (init_filters=64)',
            'spatial_dims': 2,
            'init_filters': 64,
            'in_channels': 3,
            'out_channels': 13,
            'blocks_down': [1, 2, 2, 4],
            'blocks_up': [1, 1, 1],
            'input_shape': (4, 3, 320, 320)
        },
        {
            'name': 'SegResNet-Large (init_filters=128)',
            'spatial_dims': 2,
            'init_filters': 128,
            'in_channels': 3,
            'out_channels': 13,
            'blocks_down': [1, 2, 2, 4],
            'blocks_up': [1, 1, 1],
            'input_shape': (4, 3, 320, 320)
        }
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{config['name']}:")
        print("-" * 40)
        
        # 创建模型
        model = SegResNet(
            spatial_dims=config['spatial_dims'],
            init_filters=config['init_filters'],
            in_channels=config['in_channels'],
            out_channels=config['out_channels'],
            blocks_down=config['blocks_down'],
            blocks_up=config['blocks_up']
        )
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f'参数量: {total_params:,}')
        print(f'模型大小: {total_params * 4 / 1024 / 1024:.2f} MB')
        
        # 创建输入
        x = torch.rand(*config['input_shape'])
        
        # 测试前向传播
        try:
            output = model(x)
            print(f'输入: {x.shape} -> 输出: {output.shape}')
            
            # 计算FLOPs
            try:
                flops_thop, _ = profile(model, inputs=(x,), verbose=False)
                flops_g = flops_thop / 1e9
                print(f'FLOPs: {flops_g:.2f} G')
                
                results.append({
                    'name': config['name'],
                    'params': total_params,
                    'flops': flops_g
                })
            except:
                print('FLOPs计算失败')
                results.append({
                    'name': config['name'],
                    'params': total_params,
                    'flops': 0
                })
                
        except Exception as e:
            print(f'前向传播失败: {e}')
    
    # 打印对比表格
    print("\n" + "=" * 60)
    print("配置对比总结")
    print("=" * 60)
    print(f"{'模型':<30} {'参数量':<15} {'FLOPs(G)':<10}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['name']:<30} {result['params']:<15,} {result['flops']:<10.2f}")

def benchmark_inference_speed():
    """测试推理速度"""
    print("\n" + "=" * 60)
    print("SegResNet 推理速度测试")
    print("=" * 60)
    
    # 创建模型
    model = SegResNet(
        spatial_dims=3,
        init_filters=64,
        in_channels=1,
        out_channels=2,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1]
    )
    
    model.eval()
    
    # 创建输入
    x = torch.rand(1, 1, 64, 64, 64)
    
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
        device_name = "GPU"
    else:
        device_name = "CPU"
    
    print(f"在{device_name}上测试推理速度...")
    
    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    
    # 测试推理时间
    num_runs = 50
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    
    print(f"平均推理时间: {avg_time * 1000:.2f} ms")
    print(f"推理速度: {1 / avg_time:.2f} FPS")

if __name__ == "__main__":
    print("SegResNet 模型测试")
    print("=" * 60)
    
    # 主要测试：仿照Deeplamba风格
    print("1. 仿照Deeplamba风格的测试")
    model, total_params = test_segresnet_deeplamba_style()
    
    # 与Deeplamba直接对比
    print("\n2. 与Deeplamba对比测试")
    compare_results = compare_with_deeplamba()
    
    # 测试不同配置的SegResNet
    print("\n3. 不同配置对比测试")
    test_segresnet_variants()
    
    # 传统的2D和3D测试
    print("\n4. 传统2D/3D测试")
    test_segresnet_2d()
    test_segresnet_3d()
    
    # 推理速度测试
    print("\n5. 推理速度测试")
    benchmark_inference_speed()
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)