## requirements
```python
pip install pyltp==0.1.9.1
# 哈工大做的分句、分词、词性标注的工具，也可以用jieba
pip install gensim
# 包含word2vec模型
```
如果使用pyltp要下载对应（与1.9版本对应的）的模型文件：
https://pan.baidu.com/share/link?shareid=1988562907&uk=2738088569#list/path=%2F
## 参数介绍

* data_path:存放pdf的目录
* words_dim:词向量的维度
* text_save_path：临时存放pdf转换为txt的位置，转换格式用json逐行读取，每行是一个pdf，每一行存放格式是[[分好词的句子]，[分好词的句子]...]
* is_first_time:第一次执行转换pdf为txt，之后执行直接读txt
* min_count：word2vec参数，最少出现次数，少于的词不纳入考虑
* num_process：用于处理pdf和训练w2v的CPU数量
* model_save_path：存放模型的位置
* ltp_path：存放用于分词的ltp模型的位置

## 目录结构
目录结构：

* -data : 存放pdf数据
* -tmp_text：存放pdf转换为txt的文件的位置
* -main.py