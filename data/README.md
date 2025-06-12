# 样例数据

本目录包含用于测试星语自闭症早期筛查系统的样例音频数据。

## 数据说明

样例数据包含两类音频文件：

1. **自闭症儿童语音**：
   - `ASDchild2.wav`
   - `ASDchild3.wav`
   - `ASDChild_60.wav`

2. **正常发育儿童语音**：
   - `NormalChild_30.wav`
   - `NormalChild_60.wav`
   - `NormalChild_89.wav`

## 数据来源

这些样例数据是从我们收集的更大数据集中精选出来的，经过了去标识化处理，可以安全地用于演示和测试目的。原始数据在获取时已经获得了相关监护人的知情同意。

## 使用方法

您可以使用这些样例数据来测试系统的各项功能：

```bash
# 音频处理
python ../audio_processing.py --audio ASDchild2.wav --denoise --extract --output ../results

# 自闭症风险预测
python ../predict.py --audio NormalChild_30.wav --model ../model_training/model_output/best_model.pth --output ../results
```

## 注意事项

- 这些样例数据仅用于演示和测试目的，不应用于临床诊断
- 请尊重数据隐私，不要将这些音频用于其他目的
- 如果您需要更多数据用于研究，请联系我们获取完整数据集 