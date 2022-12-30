import numpy as np
import matplotlib.pyplot as plt
import random

# 准备数据
x_data = ['0','1','2','3','4','5','6','7']
y_data = [114,117,116,117,116,116,122,117]

# 正确显示中文和负号
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 画图，plt.bar()可以画柱状图
for i in range(len(x_data)):
	plt.bar(x_data[i], y_data[i])
# 设置图片名称
# plt.title("data analyse")
# 设置x轴标签名
plt.xlabel("state")
# 设置y轴标签名
# plt.ylabel("行为")
# 显示
plt.show()
# ============================================================================================

# import os
# #输入想要存储图像的路径
# os.chdir('C:/Users/pwx/desktop')

# import matplotlib.pyplot as plt 
# import numpy as np 
# #改变绘图风格
# import seaborn as sns
# sns.set(color_codes=True)
 
 
# cell = ["safe driving", "texting-R", "phone-R", "texting-L",  
# 		"phone-L", "operation radio", "drinking", "reaching behind", 
# 		"make up", "talking"]
# pvalue = [2489,2267,2317,2346,2326,2312,2325,2002,1911,2129]
 
 
# width = 0.20
# index = np.arange(len(cell)) 
# p1 = np.arange(0,len(cell),0.01)
# p2 = 0.05 + p1*0
 
# q1 = np.arange(0,len(cell),0.01)
# q2 = 0.1 + p1*0
 
# figsize = (10,8)#调整绘制图片的比例
# plt.rcParams["font.sans-serif"] = ["SimHei"]
# plt.rcParams["axes.unicode_minus"] = False
# plt.bar(index, pvalue, width,color="#87CEFA") #绘制柱状图
# #plt.xlabel('cell type') #x轴
# plt.ylabel('') #y轴
# # plt.title('数据分析') #图像的名称
# plt.xticks(index, cell,fontsize=10) #将横坐标用cell替换,fontsize用来调整字体的大小
# # plt.savefig('test.png',dpi = 400) #保存图像，dpi可以调整图像的像素大小
# plt.show()
# ================================================================================================

# from matplotlib import pyplot as plt
# fig = plt.figure(figsize=(5,5))

# plt.subplot(2,2,1)
# x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# y = [0.4688,0.6094,0.6406,0.6719,0.6719,0.7500,0.8125,0.8281,0.9062,0.9688,0.9844,1.0000,1.0000,1.0000,1.0000]
# plt.plot(x,y,color='c')
# plt.title('val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('')
# plt.xticks(x)

# plt.subplot(2,2,2)
# x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# y = [9.0457,9.5392,7.3995,4.9300,4.3271,2.1739,2.0195,1.3983,0.3279,0.1400,0.0629,0.0027,0.0052,0.0012,0.0006]
# plt.plot(x,y,color='c')
# plt.title('val_loss')
# plt.xlabel('Epoch')
# plt.ylabel('')
# plt.xticks(x)

# plt.subplot(2,2,3)
# x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
# y = [0.6671,0.7139,0.8293,0.9303,0.9519,0.9411,0.9327,0.9219,0.9339,0.9219,0.9267,0.9315,0.9339,0.9159]
# plt.plot(x,y,color='c')
# plt.title('val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('')
# plt.xticks(x)

# plt.subplot(2,2,4)
# x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
# y = [3.9770,1.4400,0.8570,0.2864,0.1960,0.1985,0.2364,0.2447,0.2489,0.2539,0.2791,0.2851,0.2430,0.2945]
# plt.plot(x,y,color='c')
# plt.title('val_loss')
# plt.xlabel('Epoch')
# plt.ylabel('')
# plt.xticks(x)

plt.suptitle('InceptionV3',fontweight ="bold")
plt.tight_layout()
plt.show()