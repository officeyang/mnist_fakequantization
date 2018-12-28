# mnist_fakequantization

这里是mnist的量化训练，代码中添加了伪量化操作，并完成了训练，生成含有伪量化节点的.pb文件，可供toco指令使用，生成量化的tflite。
具体步骤介绍及代码分析可见博客：https://blog.csdn.net/angela_12/article/details/85000072#commentsedit

这里：
mnist_build_network.py：为模型创建
mnist_fakequantize.py ：为量化训练（先运行这个训练）
mnist_fakequantize_freeze.py：为固化冻结（再运行固化）
