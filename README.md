## 基于CRNN_Tensorflow框架的字符识别模型：(CTC 和Seq2seq)  


## 安装  

请参照这个[链接](https://github.com/MaybeShewill-CV/CRNN_Tensorflow)，配置必要的运行环境

## 训练  

1、 基于数据集Synth 90k生成TFRecord文件，注意使用STN网络的话将生成不同的数据文件：  

```
cd tools
conda activate your_env
python Write tfrecords tools.py --cfg your_config_file
```  

2、 打开tools文件夹，运行train.py训练模型：  

```
python train.py --cfg your_config_file
```

## 测试  

1、运行test.py，指定训练所使用的配置文件，以及模型对应的checkpoint：  

```
python test.py --cfg your_config_file --weights_path your_weight_path
```
