# Fight Collaborate

# Intro：

项目分为七个部分，如下图所示，（除了Entity为抽象概念，不关联到具体项目）。

<img src="https://ws4.sinaimg.cn/large/006tNc79ly1fzk8q1ku73j31xs0ksq5a.jpg" width = "700px"/>


# Components：

1. text analysis：进行nlp分词，抽取意象和意象的动作。

1. Material Library：素材库，即意象素材，分为背景库，人物库，动作库。例如：月亮，楼阁，山水图片等；老年、中年、青年的人物；走路、跑步、举臂的动作。



1. Montion Transfer：动作迁移，将指定动作，迁移给指定人物（让人物做指定动作，例如举杯邀明月）。使用[EDN模型，Everybody Dance Now
]，代码在[EverybodyDanceNow_reproduce_pytorch]文件夹。训练时长P100在4h左右。

    <img src="https://ws2.sinaimg.cn/large/006tNc79ly1fzkfwufky4j31n70mxmyp.jpg" width = "500px"/>
    <br/>
    <img src="https://ws2.sinaimg.cn/large/006tNc79ly1fzkfwu911lj31m50k8myj.jpg" width = "500px"/>

1. Segmentation：人物分割，将人物从背景中分离出来（实现将透明背景的人物合成到动画中）。使用[MaskRCNN模型]，具体为[Mask_RCNN]文件夹。分割时长，P100在5min/100帧左右。

1. Style Transfer：风格迁移，将风景摄影图迁移成为中国水墨画风格。使用[CycleGAN模型]，代码在[cycle]文件夹中。训练时长，P100在Xh左右。

    <img src="https://ws1.sinaimg.cn/large/006tNc79ly1fzkfrdhqanj31qw0s2twv.jpg" width = "500px"/>

1. Animation Synthesis：动画合成，使用逻辑代码，进行[素材定位，效果添加（缩放，平移），图片帧合并成视频]，进行动画的生成。

    <a href="https://drive.google.com/open?id=1s3_PyIyuE3N5ua9x5ZpuAlcaTP94Hc8F">
    <img src="https://ws3.sinaimg.cn/large/006tNc79ly1fzkf2cemymj30u00u0wnl.jpg" width = "300px"/>
    </a>
    
    <a href="https://drive.google.com/open?id=1-tozNZNTcBmiHhRUf_MAt4Oj92JSyVb7">
    <img src="https://ws1.sinaimg.cn/large/006tNc79ly1fzkf2equ8nj30u00u0k4v.jpg" width = "300px"/>
    </a>

1. Web Display：提供项目的介绍，生成视频的观看，生成视频接口的提供（暂无）。

    <img src="https://ws4.sinaimg.cn/large/006tNc79ly1fzkg1ztrekj31jy0u0q7y.jpg" width = "500px"/>



# presentation：

1. step1: 出三个视频，分别让大家猜是哪首诗。

1. step2:讲技术流程，首先：1、text2entity，获取名词以及动词对应素材库中的素材。2、分割：将人的素材分割出来进行第三步处理，其他素材直接第四步。3、素材动态化，利用open pose + gan + face gan 生成动态素材。4、中国风风格迁移：将背景利用cyclegan（+人物）渲染成中国水墨风格。5、上述素材按照一定顺序整合成为动画（位置+缩放），输出为视频。6、前端展示，后端将视频存入数据库。

1. step3:项目意义：1、教育（故事+图片动画）。2、娱乐（体感交互+趣味竞猜）。3、弘扬传统文化。4、NLP+CV

1. step4: 展望：1、丰富素材库。2、交互（语音+动作）。3、生成动画的画面逻辑（素材的物理意义）。

# TODO：

1. 前端：2个页面：1、展示页面；2、列表页面。（陈）

1. nlp：分词。（燕）

1. 找古诗对应的素材（名词+动词）：背景+人物：水、山、桥、房屋、月亮等，人物：老年中年青年男女。（邓）

1. 后端：1、根据名词找到素材，对素材进行上述2、3、4步骤处理，写个统一的脚本；（合）2、生成动画，将其添加适当位移缩放，合并成为最终的动画。（陈）

1. github维护，分支操作 （燕）

1. poster，ppt，视频制作 （合）

1. 文末彩蛋，中国风抖音。录凯哥读诗（合）

