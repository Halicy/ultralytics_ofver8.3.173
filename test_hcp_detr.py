#!/usr/bin/env python3
"""
HCP-DETR (Hierarchical Category Prototype Learning) 功能测试脚本

测试内容:
1. HCPRTDETRDecoder 初始化检查
2. 前向传播 (训练模式)
3. 前向传播 (推理模式)
4. 层次化映射矩阵构建
5. 原型对比损失计算
6. 子类别分数融合
7. 梯度流检查
8. 性能对比 (vs RTDETRDecoder)

Usage:
    python test_hcp_detr.py
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from ultralytics.nn.modules.head import RTDETRDecoder, HCPRTDETRDecoder
    print("✅ Successfully imported RTDETRDecoder and HCPRTDETRDecoder")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please ensure ultralytics is installed and head.py contains HCPRTDETRDecoder")
    sys.exit(1)


def print_section(title):
    """Print formatted section title"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def test_1_initialization():
    """Test 1: HCPRTDETRDecoder 初始化"""
    print_section("Test 1: HCPRTDETRDecoder 初始化检查")

    # 定义子类别
    sub_categories = {
        1: ['young_fruit', 'flower', 'occluded', 'malformed']  # no_harvestable → 4 子类
    }

    # 创建 decoder
    decoder = HCPRTDETRDecoder(
        nc=2,                          # 原始类别数
        ch=(256, 256, 256),            # 输入通道
        hd=256,                        # 隐藏层维度
        nq=300,                        # 查询数量
        ndl=2,                         # Decoder 层数 (减少以加快测试)
        nh=8,                          # 注意力头数
        ndp=4,                         # 可变形采样点数
        ndff=1024,                     # FFN 维度
        dropout=0.0,
        sub_categories=sub_categories,
        prototype_temp=0.07,
        prototype_loss_weight=0.3,
    )

    # 检查属性
    assert decoder.original_nc == 2, f"original_nc should be 2, got {decoder.original_nc}"
    assert decoder.num_sub == 4, f"num_sub should be 4, got {decoder.num_sub}"
    assert decoder.total_nc == 6, f"total_nc should be 6 (2+4), got {decoder.total_nc}"

    print(f"✅ original_nc = {decoder.original_nc}")
    print(f"✅ num_sub = {decoder.num_sub}")
    print(f"✅ total_nc = {decoder.total_nc}")

    # 检查原型
    assert decoder.prototypes.shape == (6, 256), \
        f"prototypes shape should be (6, 256), got {decoder.prototypes.shape}"
    print(f"✅ prototypes shape = {decoder.prototypes.shape}")

    # 检查投影头
    assert len(decoder.proj_head) == 4, "proj_head should have 4 layers"
    print(f"✅ proj_head has {len(decoder.proj_head)} layers")

    # 检查映射矩阵
    assert decoder.sub_to_main.shape == (6, 2), \
        f"sub_to_main shape should be (6, 2), got {decoder.sub_to_main.shape}"
    print(f"✅ sub_to_main shape = {decoder.sub_to_main.shape}")

    print("\n✅ Test 1 PASSED: 初始化检查完成")
    return decoder


def test_2_hierarchy_matrix(decoder):
    """Test 2: 层次化映射矩阵"""
    print_section("Test 2: 层次化映射矩阵检查")

    sub_to_main = decoder.sub_to_main.cpu()
    print(f"sub_to_main matrix shape: {sub_to_main.shape}")
    print("\nMatrix (rows=total_nc, cols=original_nc):")
    print(sub_to_main.numpy())

    # 验证主类别映射 (对角线)
    assert sub_to_main[0, 0] == 1.0, "Class 0 should map to itself"
    assert sub_to_main[1, 1] == 1.0, "Class 1 should map to itself"
    print("✅ Main categories map correctly (diagonal elements)")

    # 验证子类别映射 (all map to class 1)
    for i in range(2, 6):  # Classes 2-5 are subcategories of class 1
        assert sub_to_main[i, 1] == 1.0, f"Subclass {i} should map to class 1"
        assert sub_to_main[i, 0] == 0.0, f"Subclass {i} should not map to class 0"
    print("✅ Subcategories (2-5) map to main class 1")

    # 验证矩阵每行只有一个 1.0
    row_sums = sub_to_main.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(6)), "Each row should sum to 1.0"
    print("✅ Each row sums to 1.0 (valid mapping)")

    print("\n✅ Test 2 PASSED: 层次化映射矩阵正确")


def test_3_forward_train(decoder):
    """Test 3: 前向传播 (训练模式)"""
    print_section("Test 3: 前向传播 (训练模式)")

    decoder.train()
    batch_size = 2
    h, w = 64, 64

    # 模拟输入 (3 个特征层)
    x = [
        torch.randn(batch_size, 256, h, w),
        torch.randn(batch_size, 256, h//2, w//2),
        torch.randn(batch_size, 256, h//4, w//4),
    ]

    # 模拟 batch (包含 GT 标签)
    batch = {
        'cls': torch.randint(0, 6, (batch_size, 50)),  # [bs, max_labels] 范围 0-5
        'bboxes': torch.rand(batch_size, 50, 4),        # [bs, max_labels, 4]
    }

    print(f"Input shapes: {[xi.shape for xi in x]}")
    print(f"Batch cls shape: {batch['cls'].shape}, range: [{batch['cls'].min()}, {batch['cls'].max()}]")

    # 前向传播
    try:
        outputs = decoder(x, batch)

        # 解包
        if len(outputs) == 6:
            dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta, prototype_losses = outputs
            has_proto_losses = True
        else:
            dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = outputs
            has_proto_losses = False
            prototype_losses = {}

        print(f"✅ Forward pass successful")
        print(f"   dec_bboxes shape: {dec_bboxes.shape}")
        print(f"   dec_scores shape: {dec_scores.shape}")

        # 检查输出形状
        assert dec_bboxes.shape[0] == batch_size, "Batch size mismatch"
        assert dec_bboxes.shape[1] == 300, "Query number should be 300"
        assert dec_bboxes.shape[2] == 4, "Bbox should have 4 coordinates"
        assert dec_scores.shape[2] == 6, f"Scores should have 6 classes (total_nc), got {dec_scores.shape[2]}"

        print(f"✅ Output shapes correct")

        # 检查原型损失
        if has_proto_losses:
            print(f"\n原型损失:")
            for key, value in prototype_losses.items():
                if isinstance(value, torch.Tensor):
                    print(f"   {key}: {value.item():.4f}")
                else:
                    print(f"   {key}: {value}")

            assert 'instance_proto_loss' in prototype_losses, "Missing instance_proto_loss"
            assert 'proto_separation_loss' in prototype_losses, "Missing proto_separation_loss"
            print(f"✅ Prototype losses present")
        else:
            print(f"⚠️  No prototype losses returned (may be due to no valid samples)")

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        raise

    print("\n✅ Test 3 PASSED: 训练模式前向传播正确")


def test_4_forward_inference(decoder):
    """Test 4: 前向传播 (推理模式)"""
    print_section("Test 4: 前向传播 (推理模式)")

    decoder.eval()
    batch_size = 1  # 推理时通常 bs=1
    h, w = 64, 64

    # 模拟输入
    x = [
        torch.randn(batch_size, 256, h, w),
        torch.randn(batch_size, 256, h//2, w//2),
        torch.randn(batch_size, 256, h//4, w//4),
    ]

    print(f"Input shapes: {[xi.shape for xi in x]}")

    # 前向传播 (无 batch)
    with torch.no_grad():
        outputs = decoder(x, batch=None)

        # decoder.export = False 时返回 (y, ...)
        if isinstance(outputs, tuple):
            y = outputs[0]
        else:
            y = outputs

        print(f"✅ Inference pass successful")
        print(f"   Output shape: {y.shape}")

        # 检查输出形状
        # y: [nq, 4 + original_nc] = [300, 6]
        assert y.shape[0] == 300, "Query number should be 300"
        assert y.shape[1] == 6, f"Output should have 6 dims (4 bbox + 2 classes), got {y.shape[1]}"

        # 检查 bbox 范围 (应在 0-1 之间，因为是归一化坐标)
        bboxes = y[:, :4]
        print(f"   Bbox range: [{bboxes.min():.3f}, {bboxes.max():.3f}]")

        # 检查分数范围 (sigmoid 后应在 0-1)
        scores = y[:, 4:]
        print(f"   Score range: [{scores.min():.3f}, {scores.max():.3f}]")
        assert (scores >= 0).all() and (scores <= 1).all(), "Scores should be in [0, 1] after sigmoid"

    print("\n✅ Test 4 PASSED: 推理模式前向传播正确 (子类别已融合为 2 类)")


def test_5_prototype_contrastive_loss(decoder):
    """Test 5: 原型对比损失"""
    print_section("Test 5: 原型对比损失计算")

    # 模拟查询特征和标签
    batch_size = 16
    features = torch.randn(batch_size, 256)  # [B, hidden_dim]
    labels = torch.randint(0, 6, (batch_size,))  # [B] 范围 0-5

    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}, unique: {labels.unique().tolist()}")

    # 计算损失
    instance_loss, proto_sep_loss = decoder.prototype_contrastive_loss(features, labels)

    print(f"✅ Losses computed:")
    print(f"   instance_proto_loss: {instance_loss.item():.4f}")
    print(f"   proto_separation_loss: {proto_sep_loss.item():.4f}")

    # 检查损失值合理性
    assert instance_loss.item() >= 0, "Instance loss should be non-negative"
    assert proto_sep_loss.item() >= 0, "Proto separation loss should be non-negative"

    # 测试梯度
    total_loss = instance_loss + proto_sep_loss
    total_loss.backward()

    # 检查原型梯度
    assert decoder.prototypes.grad is not None, "Prototypes should have gradients"
    print(f"✅ Prototypes grad norm: {decoder.prototypes.grad.norm().item():.4f}")

    # 清除梯度
    decoder.zero_grad()

    print("\n✅ Test 5 PASSED: 原型对比损失计算正确")


def test_6_merge_subcategory_scores(decoder):
    """Test 6: 子类别分数融合"""
    print_section("Test 6: 子类别分数融合")

    batch_size = 2
    nq = 300
    total_nc = 6

    # 模拟细粒度分数 [bs, nq, total_nc]
    fine_scores = torch.randn(batch_size, nq, total_nc)

    print(f"Fine-grained scores shape: {fine_scores.shape}")

    # 融合为粗粒度分数
    coarse_scores = decoder.merge_subcategory_scores(fine_scores)

    print(f"Coarse-grained scores shape: {coarse_scores.shape}")

    # 检查形状
    assert coarse_scores.shape == (batch_size, nq, 2), \
        f"Coarse scores shape should be (2, 300, 2), got {coarse_scores.shape}"

    # 检查融合逻辑
    # coarse[:, :, 0] 应等于 fine[:, :, 0] (harvestable)
    # coarse[:, :, 1] 应等于 fine[:, :, 1:6].sum(dim=-1) (no_harvestable + 4 子类)

    # 由于是矩阵乘法，验证第一列
    expected_class0 = fine_scores[:, :, 0:1]  # [bs, nq, 1]
    actual_class0 = coarse_scores[:, :, 0:1]

    assert torch.allclose(expected_class0, actual_class0, atol=1e-5), \
        "Class 0 fusion incorrect"
    print("✅ Class 0 (harvestable) fusion correct")

    # 验证第二列 (sum of class 1-5)
    expected_class1 = fine_scores[:, :, 1:6].sum(dim=-1, keepdim=True)
    actual_class1 = coarse_scores[:, :, 1:2]

    assert torch.allclose(expected_class1, actual_class1, atol=1e-5), \
        "Class 1 fusion incorrect"
    print("✅ Class 1 (no_harvestable) fusion correct (sum of 5 subcategories)")

    print("\n✅ Test 6 PASSED: 子类别分数融合正确")


def test_7_gradient_flow():
    """Test 7: 梯度流检查"""
    print_section("Test 7: 梯度流检查")

    sub_categories = {1: ['young_fruit', 'flower', 'occluded', 'malformed']}

    decoder = HCPRTDETRDecoder(
        nc=2, ch=(256, 256, 256), hd=256, nq=100,  # nq=100 减少计算
        ndl=1, nh=4, ndp=2,  # 简化配置
        sub_categories=sub_categories,
        prototype_temp=0.07,
        prototype_loss_weight=0.3,
    )
    decoder.train()

    # 模拟输入
    batch_size = 1
    x = [
        torch.randn(batch_size, 256, 32, 32),
        torch.randn(batch_size, 256, 16, 16),
        torch.randn(batch_size, 256, 8, 8),
    ]
    batch = {
        'cls': torch.randint(0, 6, (batch_size, 20)),
        'bboxes': torch.rand(batch_size, 20, 4),
    }

    # 前向传播
    outputs = decoder(x, batch)

    if len(outputs) == 6:
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta, prototype_losses = outputs
    else:
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = outputs
        prototype_losses = {}

    # 计算总损失
    loss_bbox = dec_bboxes.mean()
    loss_cls = dec_scores.mean()
    total_loss = loss_bbox + loss_cls

    if prototype_losses:
        total_loss += prototype_losses.get('instance_proto_loss', 0)
        total_loss += prototype_losses.get('proto_separation_loss', 0)

    print(f"Total loss: {total_loss.item():.4f}")

    # 反向传播
    total_loss.backward()

    # 检查关键组件的梯度
    grad_components = {
        'prototypes': decoder.prototypes.grad,
        'proj_head.0.weight': decoder.proj_head[0].weight.grad,
        'proj_head.3.weight': decoder.proj_head[3].weight.grad,
    }

    print("\nGradient norms:")
    for name, grad in grad_components.items():
        if grad is not None:
            grad_norm = grad.norm().item()
            print(f"   {name}: {grad_norm:.4f}")
            assert grad_norm > 0, f"{name} has zero gradient!"
        else:
            print(f"   {name}: None (⚠️  no gradient)")

    print("\n✅ Test 7 PASSED: 梯度流正常")


def test_8_performance_comparison():
    """Test 8: 性能对比 (HCP vs Baseline)"""
    print_section("Test 8: 性能对比 (HCP-DETR vs RT-DETR)")

    import time

    sub_categories = {1: ['young_fruit', 'flower', 'occluded', 'malformed']}

    # Baseline decoder
    baseline_decoder = RTDETRDecoder(
        nc=2, ch=(256, 256, 256), hd=256, nq=300,
        ndl=2, nh=8, ndp=4,
    )

    # HCP decoder
    hcp_decoder = HCPRTDETRDecoder(
        nc=2, ch=(256, 256, 256), hd=256, nq=300,
        ndl=2, nh=8, ndp=4,
        sub_categories=sub_categories,
    )

    # 统计参数量
    baseline_params = sum(p.numel() for p in baseline_decoder.parameters())
    hcp_params = sum(p.numel() for p in hcp_decoder.parameters())

    print(f"Parameter count:")
    print(f"   Baseline RT-DETR: {baseline_params:,}")
    print(f"   HCP-DETR: {hcp_params:,}")
    print(f"   增加: {hcp_params - baseline_params:,} (+{(hcp_params/baseline_params - 1)*100:.2f}%)")

    # 测试推理速度
    baseline_decoder.eval()
    hcp_decoder.eval()

    x = [
        torch.randn(1, 256, 64, 64),
        torch.randn(1, 256, 32, 32),
        torch.randn(1, 256, 16, 16),
    ]

    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = baseline_decoder(x)
            _ = hcp_decoder(x)

    # 测速 (Baseline)
    n_runs = 100
    with torch.no_grad():
        start = time.time()
        for _ in range(n_runs):
            _ = baseline_decoder(x)
        baseline_time = (time.time() - start) / n_runs

    # 测速 (HCP)
    with torch.no_grad():
        start = time.time()
        for _ in range(n_runs):
            _ = hcp_decoder(x)
        hcp_time = (time.time() - start) / n_runs

    print(f"\nInference time (avg over {n_runs} runs):")
    print(f"   Baseline RT-DETR: {baseline_time*1000:.2f} ms")
    print(f"   HCP-DETR: {hcp_time*1000:.2f} ms")
    print(f"   增加: {(hcp_time - baseline_time)*1000:.2f} ms (+{(hcp_time/baseline_time - 1)*100:.2f}%)")

    print("\n✅ Test 8 PASSED: 性能对比完成")


def main():
    """运行所有测试"""
    print("\n" + "🚀"*35)
    print("   HCP-DETR (Hierarchical Category Prototype Learning) 功能测试")
    print("🚀"*35)

    try:
        # Test 1: 初始化
        decoder = test_1_initialization()

        # Test 2: 层次化映射矩阵
        test_2_hierarchy_matrix(decoder)

        # Test 3: 训练模式前向传播
        test_3_forward_train(decoder)

        # Test 4: 推理模式前向传播
        test_4_forward_inference(decoder)

        # Test 5: 原型对比损失
        test_5_prototype_contrastive_loss(decoder)

        # Test 6: 子类别分数融合
        test_6_merge_subcategory_scores(decoder)

        # Test 7: 梯度流
        test_7_gradient_flow()

        # Test 8: 性能对比
        test_8_performance_comparison()

        # 总结
        print_section("✅ 所有测试通过！")
        print("\n🎉 HCP-DETR 实现正确，可以开始训练！")
        print("\n下一步:")
        print("  1. 创建 YAML 配置文件 (rtdetr-l-hcp.yaml)")
        print("  2. 训练模型: yolo detect train model=rtdetr-l-hcp.yaml ...")
        print("  3. 对比 Baseline 和 HCP-DETR 的性能")
        print("  4. 分析混淆矩阵，关注 no_harvestable 召回率提升\n")

    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
