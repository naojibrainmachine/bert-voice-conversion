# bert-voice-conversion
A voice conversion program which is consist of 3 bert. 

介绍(intro)  
-
这个说话人替换(voice conversion)采用非平行数据(No-parallel)的方式进行训练，解决方案主要来源于2016年的Lifa Sun等人的[论文](https://www.researchgate.net/publication/307434911_Phonetic_posteriorgrams_for_many-to-one_voice_conversion_without_parallel_data_training)。模型结构主要分为2部分，一部分是音素分类模型，另一部分是语音合成模型。

音素分类(phoneme classification)  
-
音素分类模型是用来对相同发音的单个音素作分类，以梅尔倒谱系数(MFCCs)产生一张[ppgs](https://www.researchgate.net/publication/307434911_Phonetic_posteriorgrams_for_many-to-one_voice_conversion_without_parallel_data_training)图,然后再给语音合成模型。模型采用[timit](https://github.com/philipperemy/timit)音素分类数据集,主要是对12层的bert作直接训练，不采用预训练方式。

语音合成(speech synthesis)  
-
语音合成是通过训练数据直接拟合目标数据(甲)的声音特征梅尔谱(Mel Bank Features)和振幅谱，这个拟合所采用的目标函数是均方误差(mse)。用目标数据(甲)得到的模型去预测一个目标数据(乙)，则合成的音频数据是发出目标数据(甲)声音的目标数据(乙)说话的内容。语音合成分两部分，一部分是6层的bert，去寻找一个从ppgs图到梅尔谱的恒等变换，再把这个学习到的梅尔谱作为输入，输入到一个6层的bert，学习一个从梅尔谱到振幅谱的恒等变换。为了学习到一个优秀的合成效果，采用预训练方式，先让模型学习到共同人声特征信息，所以在这里使用了[LibriSpeech](http://www.openslr.org/12/)的train-clean-100和train-clean-360两个数据集,一共约10万条音频数据，每一条约10秒。预训练完成后用目标数据(甲)对模型进行几个论次的微调。

转换  
-
由于语音合成数据最后得到的是振幅谱，需要转化成wav音频数据，所以采用Griffin-Limd来获得最后的wav音频数据。运行convert.py之前需要把训练过的参数[训练过的参数(提取码：bpx5)](https://pan.baidu.com/s/1rosXmM9q6KIHz8rTReljqg)放置到ckp中。


参考  
-
1.[The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](https://github.com/andabi/deep-voice-conversion)  
2.[deep-voice-conversion](https://github.com/andabi/deep-voice-conversion)  
3.[Phonetic posteriorgrams for many-to-one voice conversion without parallel data training](https://www.researchgate.net/publication/307434911_Phonetic_posteriorgrams_for_many-to-one_voice_conversion_without_parallel_data_training)
