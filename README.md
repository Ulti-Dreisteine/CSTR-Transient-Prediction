## CSTR瞬态数据预测建模
### 建模目的
* CSTR反应器在运行过程中经常发生操作变动，过程变量如产品浓度和反应温度等也随之发生变化和波动，从一个稳态经过瞬态过渡到另一个稳态，原来的学习模型
  希望能够采集到各个操作参数输入与模型稳态输出的一一对应关系，但这是不现实的，实际过程中由于操作变动频繁可能根本无法获得稳态数据，只能获得模型在
  短期的瞬态变化过程数据
* 模型中的CSTR反应器为非等温反应器，温度于反应放热耦合，具体建模参考references中相关材料
* 本模型考虑结合CSTR反应器操作参数和过程变量的历史记录，对其将来行为进行预测，以便对模型未来的稳态状态进行估计

### 模型结构
模型共分为analysis、config、files、graphs、lib、mods、refrences、trash等目录，其中，lib为模型主要过程命令，config中为模型配置文件，
mods提供了数据分析和建模等相关模块函数，files用于放置模型过程中的重要数据，graphs用于保存模型重要图表，refrences为参考文献，历史代码和数据放入
trash中

### 模型运行
1.  运行cstr反应器模型
    * 运行lib/running_cstr.py, 生成过程变量数据和对应的操作参数数据记录
        * 过程变量：反应物浓度ca和反应温度T，操作参数：反应物进口浓度ca_0, 反应物进口温度T_0和进口流量q
        * 程序中使用seq_len_range设置每次操作变动后的一套操作参数的运行时间长度范围，模型会随机地在这个范围中取整数作为下一次变动后的运行时长
        * 在每次操作参数变动时模型会随机选择一个操作参数进行改变，变动范围在config/config.yml中设置
        * 数据保存于files中用于之后的分析
        
2.  数据相关性分析
    * 运行analysis/ccf.py，对cstr过程变量数据和操作参数之间的相关性进行分析
        * 相关性结果用于之后模型样本选取过程中
        * ccf交叉分析结果图和数据分别保存于graphs和files中
    * 运行analysis/graph_net.py（可选），该功能用于展示过程变量和操作参数之间的影响强弱关系
        * 图片保存于graphs中

3. 样本构建和模型训练
    * 使用mods/build_train_and_test_samples.py生成建模用的训练集、验证集和测试集数据，具体参数参考config/config.yml
    * 神经网络模型位于mods/nn_models.py，提供生成和载入模型的函数
    * 运行lib/train_nn.py，生成训练数据然后使用神经网络进行学习，模型学习过程中的loss和相关数据分别保存于graphs和files中
    
4. 模型评估
    * 首先运行analysis/model_evaluations.py对神经网络在未来不同时刻的预测效果进行评估，并生成nn_evaluation_results.csv数据放入files中
    * 然后运行analysis/single_sample_prediction.py使用学习好的模型对某一样本进行预测，并根据nn_evaluation_results.csv查看预测曲线和
      误差区间

### 模型效果
* 模型基本能预测500个样本以内的瞬态变化，对稳态跟踪准确

### TODO: