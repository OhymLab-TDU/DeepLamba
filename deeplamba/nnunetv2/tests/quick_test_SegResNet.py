#!/usr/bin/env python3
"""
快速测试SegResNet模型参数量和FLOPs
仿照Deeplamba的测试方式
"""

import torch
import torch.nn as nn
from monai.networks.nets import SegResNet
from fvcore.nn import FlopCountAnalysis, flop_count, parameter_count_table
from thop import profile

def quick_test_segresnet():
    """快速测试SegResNet模型，仿照Deeplamba的测试方式"""
    
    print("=" * 60)
    print("SegResNet 快速参数测试")
    print("=" * 60)
    
    # 创建SegResNet模型 (仿照原始配置)
    model = SegResNet(
        spatial_dims=3,
        init_filters=64,
        in_channels=1,
        out_channels=2,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1]
    )
    
    # 计算总参数量 (仿照Deeplamba的方式)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')
    print(f'Model size: {total_params * 4 / 1024 / 1024:.2f} MB')
    
    # 创建输入张量
    x = torch.rand(2, 1, 64, 64, 64)  # batch_size, channels, D, H, W
    
    if torch.cuda.is_available():
        x = x.cuda()
        model = model.cuda()
        print("Using CUDA")
    else:
        print("Using CPU")
    
    # 测试前向传播
    print(f"Input shape: {x.shape}")
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # 计算FLOPs (使用fvcore，仿照Deeplamba)
    try:
        flops = FlopCountAnalysis(model, x)
        total_flops = flops.total()
        print(f"FLOPs: {total_flops / 1e9:.2f} G")
        
        # 详细的FLOPs信息
        flops_dict = flop_count(model, (x,))
        print("FLOPs breakdown:")
        for op, count in flops_dict[0].items():
            if count > 0:
                print(f"  {op}: {count / 1e9:.2f} G")
                
    except Exception as e:
        print(f"FLOPs calculation failed: {e}")
    
    # 使用thop作为备用方案
    try:
        flops_thop, params_thop = profile(model, inputs=(x,), verbose=False)
        print(f"FLOPs (thop): {flops_thop / 1e9:.2f} G")
        print(f"Parameters (thop): {params_thop:,}")
    except Exception as e:
        print(f"thop calculation failed: {e}")
    
    # 参数详细表格 (仿照Deeplamba)
    print("\nParameter count table:")
    print(parameter_count_table(model))
    
    return model, total_params

def compare_segresnet_configs():
    """比较不同配置的SegResNet模型"""
    
    print("\n" + "=" * 60)
    print("SegResNet 配置对比")
    print("=" * 60)
    
    configs = [
        {
            'name': 'SegResNet-32',
            'init_filters': 32,
            'input_size': (1, 1, 64, 64, 64)
        },
        {
            'name': 'SegResNet-64 (标准)',
            'init_filters': 64,
            'input_size': (1, 1, 64, 64, 64)
        },
        {
            'name': 'SegResNet-128',
            'init_filters': 128,
            'input_size': (1, 1, 64, 64, 64)
        }
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{config['name']}:")
        print("-" * 30)
        
        # 创建模型
        model = SegResNet(
            spatial_dims=3,
            init_filters=config['init_filters'],
            in_channels=1,
            out_channels=2,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1]
        )
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        model_size_mb = total_params * 4 / 1024 / 1024
        
        print(f"参数量: {total_params:,}")
        print(f"模型大小: {model_size_mb:.2f} MB")
        
        # 创建输入并计算FLOPs
        x = torch.rand(*config['input_size'])
        
        try:
            # 使用thop计算FLOPs
            flops, _ = profile(model, inputs=(x,), verbose=False)
            flops_g = flops / 1e9
            print(f"FLOPs: {flops_g:.2f} G")
            
            results.append({
                'name': config['name'],
                'params': total_params,
                'size_mb': model_size_mb,
                'flops_g': flops_g
            })
            
        except Exception as e:
            print(f"FLOPs计算失败: {e}")
            results.append({
                'name': config['name'],
                'params': total_params,
                'size_mb': model_size_mb,
                'flops_g': 0
            })
    
    # 打印对比表格
    print("\n" + "=" * 60)
    print("配置对比总结")
    print("=" * 60)
    print(f"{'模型':<20} {'参数量':<15} {'大小(MB)':<10} {'FLOPs(G)':<10}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['name']:<20} {result['params']:<15,} {result['size_mb']:<10.2f} {result['flops_g']:<10.2f}")

def test_2d_vs_3d():
    """对比2D和3D SegResNet"""
    
    print("\n" + "=" * 60)
    print("2D vs 3D SegResNet 对比")
    print("=" * 60)
    
    # 2D模型
    print("2D SegResNet:")
    print("-" * 20)
    model_2d = SegResNet(
        spatial_dims=2,
        init_filters=64,
        in_channels=1,
        out_channels=2,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1]
    )
    
    params_2d = sum(p.numel() for p in model_2d.parameters())
    print(f"参数量: {params_2d:,}")
    
    x_2d = torch.rand(1, 1, 256, 256)
    try:
        flops_2d, _ = profile(model_2d, inputs=(x_2d,), verbose=False)
        print(f"FLOPs: {flops_2d / 1e9:.2f} G")
    except:
        print("FLOPs计算失败")
    
    # 3D模型
    print("\n3D SegResNet:")
    print("-" * 20)
    model_3d = SegResNet(
        spatial_dims=3,
        init_filters=64,
        in_channels=1,
        out_channels=2,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1]
    )
    
    params_3d = sum(p.numel() for p in model_3d.parameters())
    print(f"参数量: {params_3d:,}")
    
    x_3d = torch.rand(1, 1, 64, 64, 64)
    try:
        flops_3d, _ = profile(model_3d, inputs=(x_3d,), verbose=False)
        print(f"FLOPs: {flops_3d / 1e9:.2f} G")
    except:
        print("FLOPs计算失败")
    
    print(f"\n参数量比例 (3D/2D): {params_3d / params_2d:.2f}x")

if __name__ == '__main__':
    # 主要测试 (仿照Deeplamba的main函数结构)
    model, total_params = quick_test_segresnet()
    
    # 配置对比
    compare_segresnet_configs()
    
    # 2D vs 3D对比
    test_2d_vs_3d()
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60) 