    def test_invalid_reduction_parameter_g2(self):
        """测试G2组损失函数的无效reduction参数"""
        # 根据执行日志，PyTorch在初始化时不会验证reduction参数
        # 而是在实际调用损失函数时通过_Reduction.get_enum验证
        # 所以我们需要在调用时捕获错误
        
        # 创建带有无效reduction参数的损失函数实例
        loss_fn = nn.BCELoss(reduction='invalid')
        
        # 创建测试数据
        input_tensor = torch.rand(2, 3)
        target_tensor = torch.randint(0, 2, (2, 3), dtype=torch.float32)
        
        # 在调用时应该抛出ValueError
        with pytest.raises(ValueError, match=".*is not a valid value for reduction.*"):
            loss_fn(input_tensor, target_tensor)
        
        # 测试其他损失函数
        loss_fn = nn.NLLLoss(reduction='wrong')
        input_tensor = F.log_softmax(torch.randn(2, 5), dim=1)
        target_tensor = torch.randint(0, 5, (2,), dtype=torch.long)
        with pytest.raises(ValueError, match=".*is not a valid value for reduction.*"):
            loss_fn(input_tensor, target_tensor)
        
        loss_fn = nn.KLDivLoss(reduction='bad')
        input_tensor = F.log_softmax(torch.randn(2, 5), dim=1)
        target_tensor = F.softmax(torch.randn(2, 5), dim=1)
        with pytest.raises(ValueError, match=".*is not a valid value for reduction.*"):
            loss_fn(input_tensor, target_tensor)
        
        # 测试一些边界情况
        # 空字符串
        loss_fn = nn.BCELoss(reduction='')
        with pytest.raises(ValueError):
            loss_fn(input_tensor, target_tensor)
        
        # None值（应该使用默认值'mean'，而不是引发错误）
        # 注意：reduction参数有默认值'mean'，所以传递None应该使用默认值
        loss_fn = nn.BCELoss(reduction=None)
        assert loss_fn.reduction == 'mean', f"Default reduction should be 'mean', got {loss_fn.reduction}"
    
    def test_shape_mismatch_errors_g2(self):
        """测试G2组损失函数的形状不匹配错误"""
        # BCELoss形状不匹配
        loss_fn = nn.BCELoss()
        input_tensor = torch.rand(2, 3, 4)
        target_tensor = torch.rand(2, 3, 5)  # 形状不匹配
        
        # 根据binary_cross_entropy源码，形状不匹配会抛出ValueError
        # 但根据执行日志，实际抛出的是RuntimeError
        # 错误消息是："Using a target size (torch.Size([2, 3, 5])) that is different to the input size (torch.Size([2, 3, 4]))"
        with pytest.raises((ValueError, RuntimeError)):
            loss_fn(input_tensor, target_tensor)
        
        # NLLLoss形状不匹配
        loss_fn = nn.NLLLoss()
        input_tensor = torch.randn(2, 5, 3, 4)  # log probabilities
        target_tensor = torch.randint(0, 5, (2, 3, 5))  # 形状不匹配
        
        # 根据执行日志，NLLLoss形状不匹配抛出RuntimeError
        # 错误消息是："size mismatch (got input: [2, 5, 3, 4] , target: [2, 3, 5]"
        with pytest.raises(RuntimeError, match=".*size mismatch.*"):
            loss_fn(input_tensor, target_tensor)
        
        # KLDivLoss形状不匹配
        loss_fn = nn.KLDivLoss()
        input_tensor = F.log_softmax(torch.randn(3, 4), dim=1)
        target_tensor = F.softmax(torch.randn(3, 5), dim=1)  # 形状不匹配
        
        with pytest.raises(RuntimeError, match=".*must match.*size.*tensor.*"):
            loss_fn(input_tensor, target_tensor)
    
    def test_bceloss_invalid_probability_range(self):
        """测试BCELoss输入概率范围验证"""
        loss_fn = nn.BCELoss()
        
        # 根据执行日志，BCELoss要求输入在[0,1]范围内
        # 如果输入超出范围，会抛出RuntimeError: "all elements of input should be between 0 and 1"
        
        # 输入超出[0,1]范围 - BCELoss应该抛出RuntimeError
        input_tensor = torch.tensor([[1.5, -0.5], [0.3, 0.7]])
        target_tensor = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        
        # BCELoss应该对超出范围的输入抛出RuntimeError
        with pytest.raises(RuntimeError, match=".*all elements of input should be between 0 and 1.*"):
            loss_fn(input_tensor, target_tensor)
        
        # 测试边界值0和1应该是有效的
        valid_input = torch.tensor([[0.0, 1.0], [0.5, 0.5]])
        valid_target = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        
        # 边界值应该能正常计算
        loss = loss_fn(valid_input, valid_target)
        assert torch.isfinite(loss), f"BCELoss should handle boundary values (0 and 1)"
        assert loss.item() >= 0, f"BCE loss should be non-negative, got {loss.item()}"
        
        # 测试接近边界但有效的值
        eps = 1e-12
        near_boundary_input = torch.tensor([[eps, 1.0 - eps], [0.25, 0.75]])
        near_boundary_loss = loss_fn(near_boundary_input, valid_target)
        assert torch.isfinite(near_boundary_loss), f"BCELoss should handle values near boundaries"
        
        # 验证手动计算（使用clamp）与BCELoss在有效范围内的结果一致
        # 注意：BCELoss内部可能使用不同的数值稳定方法
        input_clamped = torch.clamp(valid_input, eps, 1 - eps)
        manual_bce = -(valid_target * torch.log(input_clamped) + 
                      (1 - valid_target) * torch.log(1 - input_clamped))
        manual_loss = torch.mean(manual_bce)
        
        # 由于数值方法可能不同，使用合理的容差
        assert torch.allclose(loss, manual_loss, rtol=1e-4, atol=1e-5), \
            f"BCELoss should produce similar results to clamped manual calculation: {loss} vs {manual_loss}"
    
    def test_nllloss_invalid_target_indices(self):
        """测试NLLLoss无效目标索引"""
        loss_fn = nn.NLLLoss()
        input_tensor = F.log_softmax(torch.randn(3, 5), dim=1)  # 3个样本，5个类别
        
        # 目标索引超出范围
        target_tensor = torch.tensor([5, 1, 2], dtype=torch.long)  # 5超出范围[0,4]
        
        # NLLLoss应该对无效索引抛出错误
        # 注意：具体错误类型可能因PyTorch版本而异
        with pytest.raises((RuntimeError, IndexError)):
            loss_fn(input_tensor, target_tensor)
        
        # 测试负索引
        target_tensor = torch.tensor([-1, 1, 2], dtype=torch.long)
        with pytest.raises((RuntimeError, IndexError)):
            loss_fn(input_tensor, target_tensor)
    
    def test_kldivloss_log_target_parameter(self):
        """测试KLDivLoss的log_target参数"""
        # 测试log_target=True的情况
        loss_fn = nn.KLDivLoss(reduction='mean', log_target=True)
        
        # 创建log probabilities输入
        input_tensor = F.log_softmax(torch.randn(3, 4), dim=1)
        target_tensor = F.log_softmax(torch.randn(3, 4), dim=1)
        
        # 应该能正常计算
        loss = loss_fn(input_tensor, target_tensor)
        assert torch.isfinite(loss), f"KLDivLoss with log_target=True should work, got {loss}"
        
        # 验证损失值是非负的（KL散度总是非负）
        assert loss.item() >= 0, f"KL divergence should be non-negative, got {loss.item()}"
        
        # 测试log_target=False的情况
        loss_fn_no_log = nn.KLDivLoss(reduction='mean', log_target=False)
        target_prob = F.softmax(torch.randn(3, 4), dim=1)
        
        loss_no_log = loss_fn_no_log(input_tensor, target_prob)
        assert torch.isfinite(loss_no_log), f"KLDivLoss with log_target=False should work"
        
        # 验证两种模式在数学上应该等价（当target是log probabilities时）
        # 注意：由于数值精度，可能会有微小差异
        target_exp = torch.exp(target_tensor)
        loss_manual = loss_fn_no_log(input_tensor, target_exp)
        assert torch.allclose(loss, loss_manual, rtol=1e-5, atol=1e-5), \
            f"log_target=True should be equivalent to exp(target) with log_target=False"
    
    def test_loss_functions_are_callable_g2(self):
        """测试G2组损失函数是可调用的"""
        # 测试所有G2组的损失函数
        loss_classes = [nn.BCELoss, nn.NLLLoss, nn.KLDivLoss]
        
        for loss_class in loss_classes:
            # 创建实例
            loss_fn = loss_class()
            
            # 创建测试数据
            if loss_class == nn.NLLLoss:
                input_tensor = F.log_softmax(torch.randn(2, 5), dim=1)
                target_tensor = torch.randint(0, 5, (2,), dtype=torch.long)
            elif loss_class == nn.KLDivLoss:
                input_tensor = F.log_softmax(torch.randn(2, 5), dim=1)
                target_tensor = F.softmax(torch.randn(2, 5), dim=1)
            else:  # BCELoss
                input_tensor = torch.rand(2, 3)
                target_tensor = torch.randint(0, 2, (2, 3), dtype=torch.float32)
            
            # 验证可调用
            assert callable(loss_fn), f"{loss_class.__name__} should be callable"
            
            # 验证调用返回张量
            result = loss_fn(input_tensor, target_tensor)
            assert isinstance(result, torch.Tensor), \
                f"{loss_class.__name__} should return a Tensor, got {type(result)}"
            
            # 验证返回的张量有正确的属性
            assert result.dtype in [torch.float32, torch.float64], \
                f"Loss should return float tensor, got {result.dtype}"
            assert result.requires_grad == False, \
                f"Loss output should not require grad by default"
    
    def test_batchmean_reduction_kldivloss(self):
        """测试KLDivLoss的batchmean reduction模式"""
        # KLDivLoss支持特殊的'batchmean' reduction
        loss_fn = nn.KLDivLoss(reduction='batchmean')
        
        # 创建输入和目标（log probabilities和probabilities）
        input_tensor = F.log_softmax(torch.randn(3, 4), dim=1)
        target_tensor = F.softmax(torch.randn(3, 4), dim=1)
        
        # 计算损失
        loss = loss_fn(input_tensor, target_tensor)
        
        # 验证输出是标量
        assert loss.shape == torch.Size([]), f"Expected scalar for batchmean reduction, got shape {loss.shape}"
        
        # 验证有限值
        assert torch.isfinite(loss), f"KLDivLoss batchmean should produce finite value, got {loss}"
        
        # 验证batchmean是sum除以batch size
        sum_loss_fn = nn.KLDivLoss(reduction='sum')
        sum_loss = sum_loss_fn(input_tensor, target_tensor)
        expected_batchmean = sum_loss / 3  # batch size = 3
        
        assert torch.allclose(loss, expected_batchmean, rtol=1e-5, atol=1e-5), \
            f"batchmean should be sum divided by batch size: expected {expected_batchmean}, got {loss}"
        
        # 验证batchmean与其他reduction模式的关系
        none_loss_fn = nn.KLDivLoss(reduction='none')
        none_loss = none_loss_fn(input_tensor, target_tensor)
        
        # batchmean应该是所有元素的和除以batch size
        manual_batchmean = torch.sum(none_loss) / 3
        assert torch.allclose(loss, manual_batchmean, rtol=1e-5, atol=1e-5), \
            f"batchmean should equal sum(none) / batch_size"
        
        # 验证batchmean是mean乘以元素数量除以batch size
        mean_loss_fn = nn.KLDivLoss(reduction='mean')
        mean_loss = mean_loss_fn(input_tensor, target_tensor)
        total_elements = 3 * 4  # batch_size * num_elements_per_sample
        expected_from_mean = mean_loss * total_elements / 3
        assert torch.allclose(loss, expected_from_mean, rtol=1e-5, atol=1e-5), \
            f"batchmean should equal mean * total_elements / batch_size"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])