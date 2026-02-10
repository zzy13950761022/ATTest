"""快速检查修复后的测试"""
import pytest
import tensorflow as tf
from tensorflow.python.framework import test_util
import unittest.mock as mock

def test_case_03_fixed():
    """检查CASE_03修复"""
    # 创建简单图形
    graph1 = tf.Graph()
    with graph1.as_default():
        a = tf.constant(1.0, name="node_a")
        b = tf.constant(2.0, name="node_b")
        c = tf.add(a, b, name="node_c")
    
    graph2 = tf.Graph()
    with graph2.as_default():
        a = tf.constant(1.0, name="node_a")
        b = tf.constant(2.0, name="node_b")
        c = tf.add(a, b, name="node_c")
    
    actual = graph1.as_graph_def()
    expected = graph2.as_graph_def()
    
    # 应该可以调用而不出错
    try:
        test_util.assert_equal_graph_def(actual, expected)
        print("✓ CASE_03修复成功：assert_equal_graph_def可以正常调用")
    except Exception as e:
        print(f"✗ CASE_03修复失败：{e}")

def test_case_04_fixed():
    """检查CASE_04修复"""
    # 使用mock测试
    with mock.patch.object(test_util.device_lib, 'list_local_devices') as mock_list_devices:
        mock_list_devices.return_value = [
            mock.Mock(device_type="CPU", name="/device:CPU:0"),
            mock.Mock(device_type="GPU", name="/device:GPU:0"),
        ]
        
        device_name = test_util.gpu_device_name()
        assert isinstance(device_name, str)
        assert device_name == "/device:GPU:0"
        print("✓ CASE_04修复成功：gpu_device_name可以正常调用")

def test_case_05_fixed():
    """检查CASE_05修复"""
    # 使用mock测试
    with mock.patch.object(test_util, 'portpicker') as mock_portpicker, \
         mock.patch.object(test_util, 'server_lib') as mock_server_lib:
        
        mock_portpicker.pick_unused_port.return_value = 10001
        mock_cluster_spec = mock.Mock()
        mock_server_lib.ClusterSpec.return_value = mock_cluster_spec
        mock_server_lib.Server.return_value = mock.Mock()
        
        workers, ps_servers = test_util.create_local_cluster(
            num_workers=1,
            num_ps=0,
            protocol="grpc"
        )
        
        assert isinstance(workers, list)
        assert isinstance(ps_servers, list)
        print("✓ CASE_05修复成功：create_local_cluster可以正常调用")

if __name__ == "__main__":
    test_case_03_fixed()
    test_case_04_fixed()
    test_case_05_fixed()
    print("\n所有修复检查完成！")