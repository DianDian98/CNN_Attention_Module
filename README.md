# CNN_Attention_Module

## Create Model

```python  
from Model_ResNet import *
```
```python
model = resnet18(num_classes=nc, Attention='SE', AttPos='STD')  
```  
### Available Model
resnet18  
resnet34  
resnet50  
resnet101  
resnet152  
resnext50_32x4d  
resnext101_32x8d  
wide_resnet50_2  
wide_resnet101_2
### Available attention module
'SE' : Squeeze-and-Excitation Module  
'GC' : Global Context Module  
'CBAM' : Convolutional Block Attention Module  
'BAM' : Bottleneck Attention Module  
'TA' : Triplet Attention Module
### Available module position
'STD'  
'PRE'  
'POST'  
'ID'  

## Reference:  
1. Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu. Squeeze-and-Excitation Networks. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Dec 2018.
2. Yue Cao, Jiarui Xu, Stephen Lin, Fangyun Wei, Han Hu. GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), Oct  2019.
3. Sanghyun Woo, Jongchan Park, Joon-Young Lee, In So Kweon. CBAM: Convolutional Block Attention Module. Proceedings of the European Conference on Computer Vision (ECCV), Jul 2018.
4. Jongchan Park, Sanghyun Woo, Joon-Young Lee, In So Kweon. BAM: Bottleneck Attention Module. BMVC. Jul 2018.
5. Diganta Misra, Trikay Nalamada, Ajay Uppili Arasanipalai, Qibin Hou. Rotate to Attend: Convolutional Triplet Attention Module. Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), Nov 2021.