# 参赛作品名
熊猫头表情生成器
## 作品简介
使用Wechaty+Paddlehub制作的表情包机器人，只要给它发送照片和视频，它就会提取你的人脸来制作你专属的熊猫头表情，还可以定制文字，玩起来吧~
[aistudio](https://aistudio.baidu.com/aistudio/projectdetail/1869462)
[B站视频](https://www.bilibili.com/video/BV1NK4y1N7m5/)
## 使用方式
JS端：   
set WECHATY_LOG=verbose   
set WECHATY_PUPPET=wechaty-puppet-wechat   
npx ts-node examples\\advanced\\Panda-Face-bot.js


Django端
python manage.py runserver 0.0.0.0:8080


> 比赛介绍+赛题重点难点剖析
#### 比赛要求使用Wechaty以及Paddlehub,对于熟悉Paddlehub的我来说难点大概只剩下Wechaty这端.项目的完成大概分为三个部分:
#### 1.Wechaty的部署
#### 2.Wechaty端处理接受到的消息
#### 3.Wechaty调用Paddlehub

> 思路介绍+个人方案亮点
#### 在一开始的时候,有多个想法想要做,表情包生成器,语音玩游戏,你言我语故事机器人,OCR相关
#### 最后选择先做表情包生成器,因为这个模型我比较熟悉,哈哈哈,还有就是这个最简单,比较能直观的看到结果
#### 现在大概介绍一下表情包生成器做了哪些事:
#### 1.接收到人脸照片/视频会提取人脸,制作对应的熊猫头表情包(经测试,gif图的部分还有一点bug,接着调吧~)
#### 2.输入文字会给表情包的底部加上想要的文字
#### 有了这个机器人就可以非常方便的制作想要的表情包了,怼人的能力笋尖爆炸~

> 具体方案分享（思路框架图、思路步骤详述、模型应用+调优过程、部分代码展示）   
#### 框架图   
![](https://ai-studio-static-online.cdn.bcebos.com/05d27434542e4bf29a80d451a4c993c972c8ab6a5e6948438fc5c7b287bf90ab)   
#### 因为python版本的Wechaty我真的搞不定，所以不得已使用这种方式：使用接口简单的js版本的Wechaty，然后使用Django启动一个服务器，在Wechaty端接收到文字/图片/视频时，会向django端发送一个get请求，django端调用Paddlehub，完成人脸信息处理，进行response给Wechaty端。   
#### Wechaty端,我直接魔改的官方的advance例子中的media-file-bot.js文件,增加了两个模块:    
#### 1.保存文字的模块,要针对不同的信息发出者,保存其对应的文字,结果保存在一个sender.json文件中    
#### 2.网络请求模块,因为我需要调用自己的django服务器,所以写了这一个模块,给自己的端口上发请求,然后处理返回的结果,返回表情包,或者告诉用户我没有在你发送的图片/视频中发现人脸    
##### 这部分的部署,我先尝试了python版本的Wechaty,但是后续的使用不太顺手,api方面不太清晰,所以最终还是使用了js的版本.js版本的部署极其简单,只要照着github中的readme就可以,一次成功.当然这得益于自己的账号可以使用网页登录的方式登录微信.如果你的账号不可以,那你可能还是需要使用padlocal之类的方法了.    

#### django服务器端,则非常的简单,在view中添加一个处理请求的方法就好,这部分处理一下请求中的数据,根据数据去调用pandaFace脚本就好.

#### pandaFace脚本则是处理图像的主文件,提供了处理图片和视频的方法.大致的流程如下
#### 1.熊猫头表情包的人脸去除和人脸定位
#### 2.接收到的图片的人脸的提取和缩放,处理到灰度图,适当添加对比和亮度
#### 3.把2中的人脸根据1中的定位,贴到1中的熊猫头上
#### 4.添加文字

> 模型应用结果分析（如：实验结果的量化分析、不同模型结果的对比分析、调参优化过程分析）   
#### 亮度调节对比
![](https://ai-studio-static-online.cdn.bcebos.com/8a9086cf256c4bb5969c847d1fdd8581fb614935966f49feb2bac2d74722f75b)   
#### 对比度调节对比
![](https://ai-studio-static-online.cdn.bcebos.com/98f037bc672f4ebb9226f7aeb7e3a522cd1637220f7045e59d807c262d47b3a9)    
#### gamma方式调节histogram调节对比
![](https://ai-studio-static-online.cdn.bcebos.com/92a7c02f5f354539bb2afd95c6a9e01bd548488a241b49658a799561a37880bb)
#### 这部分试了很多个参数,但是对于不同的图片不太好做到统一,所以想到了直方图匹配(histogram match),但是这部分实验的图找不到了,之后补上,直方图匹配的代码也在脚本中提供了,但是默认并没有开启.
#### 直方图匹配的方法可以把人脸映射到和原版的表情包大概同一个色调,但是缺点是,直方图匹配的方式会损失细节,这样会出现跃迁式的像素变化,很丑,所以可能采用直方图匹配之后要做更多的处理.
    
#### 针对文字的方面,也通过缩放等手段,通过简单的规则生成了.

> 总结+改进完善方向    
#### 最终版本中没有开启直方图匹配(提供了这部分的代码,只是没有开启),但是还是可以尝试,直方图匹配后的结果图像"质量"更差,~~但是表情包的精髓不就是糊吗?糊到一堆水印,糊到满是包浆.~~
#### 最后统一采用了2.2的对比和3的亮度,如果要改进的话,应该还是要提升一下直方图匹配的效果.
#### 除了人脸处理模块,其他的部分需要改进的大概是舍弃掉这种js+django服务器的模式,这样的效果有点慢.
#### 在频繁给django发送请求后,server端的cpu和内存"爆炸",这部分不确定是django的问题还是Paddlehub的问题,之前一直遇到,现在好像突然没了????(迷惑)

> 飞桨使用体验+给其他选手学习飞桨的建议
#### Paddlehub好,真滴好~好好使用吧,对于写趣味项目来说,已经很够用了,但是目前还都只在图像方面做,之后可能要多试试文本和语音方面.
#### 谈不上建议,多多观察生活吧,很多效果都是多个模型组合的效果~

# 个人简介

> 百度飞桨开发者技术专家 PPDE

> 百度飞桨官方帮帮团、答疑团成员

> 国立清华大学18届硕士

> 以前不懂事，现在只想搞钱～欢迎一起搞哈哈哈

我在AI Studio上获得至尊等级，点亮9个徽章，来互关呀！！！<br>
[https://aistudio.baidu.com/aistudio/personalcenter/thirdview/311006]( https://aistudio.baidu.com/aistudio/personalcenter/thirdview/311006)

B站ID： 玖尾妖熊

### 其他趣味项目：   
#### [如何变身超级赛亚人(一)--帅气的发型](https://aistudio.baidu.com/aistudio/projectdetail/1180050)
#### [【AI创造营】是极客就坚持一百秒？](https://aistudio.baidu.com/aistudio/projectdetail/1609763)    
#### [在Aistudio，每个人都可以是影流之主[飞桨PaddleSeg]](https://aistudio.baidu.com/aistudio/projectdetail/1173812)       
#### [愣着干嘛？快来使用DQN划船啊](https://aistudio.baidu.com/aistudio/projectdetail/621831)    
#### [利用PaddleSeg偷天换日～](https://aistudio.baidu.com/aistudio/projectdetail/1403330)    
