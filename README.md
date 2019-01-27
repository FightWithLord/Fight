# Fight Collaborate

# Intro：

项目分为七个部分，如下图所示，（除了Entity为抽象概念，不关联到具体项目）。

<img src="https://ws4.sinaimg.cn/large/006tNc79ly1fzk8q1ku73j31xs0ksq5a.jpg" width = "700px"/>


# Components：

1. text analysis: process word cutting, POS tagging, to extracting ‘poem image’ and the action of people.

1. Material Library: Poem image material is divided into 3 category, those are background database, character database, action database separatly. For example: moon, pavilion, landscape. Old, middle-aged, young people. Walking, running, arm movement etc.

    <img src="https://ws1.sinaimg.cn/large/006tNc79ly1fzlj7f2lexj30uw0fvtar.jpg" width = "500px"/>

1. Montion Transfer: Transfer the specific action to the specific character (for example, ‘raise hand to invite the moon’, which is an action in a poem). Using EDN Model (Everybody Dance Now), contains 4 components, which are a pose detection (openpose project), pose normalization, a GAN model mapping from pose images to a target subject's appearance, another GAN model adding additional realistic face synthesis. The model is open-source. code in[EverybodyDanceNow_reproduce_pytorch]folder。

    <img src="https://ws2.sinaimg.cn/large/006tNc79ly1fzkfwufky4j31n70mxmyp.jpg" width = "500px"/>
    <br/>
    <img src="https://ws2.sinaimg.cn/large/006tNc79ly1fzkfwu911lj31m50k8myj.jpg" width = "500px"/>

1. Segmentation: Separate characters from the background (realizing that synthesize the characters with transparent backgrounds into animations). The model is an open-source implementation of Mask R-CNN. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

1. Style Transfer: Tranfer scenery picture into the Chinese ancient ink painting style. Using cycle GAN model, which introducing a cycle consistency loss to constraint image transfer result and transfer back result. The dataset of source domain and target domain were crawling by ourselves. We propose a super-resolution method of positive stacked bottom for dealing the low resolution problem. The cycle GAN model is open-source. code in [cycle] folder.

    <img src="https://ws1.sinaimg.cn/large/006tNc79ly1fzkfrdhqanj31qw0s2twv.jpg" width = "500px"/>

1. Animation Synthesis: Synthesis process contains 3 component, that is material location, effect addition (zooming, shifting), frames to video. There can be some hard rules now, and more NLP comprehension in future.

    <a href="https://drive.google.com/open?id=1s3_PyIyuE3N5ua9x5ZpuAlcaTP94Hc8F">
    <img src="https://ws3.sinaimg.cn/large/006tNc79ly1fzkf2cemymj30u00u0wnl.jpg" width = "200px"/>
    </a>
    
    <a href="https://drive.google.com/open?id=1-tozNZNTcBmiHhRUf_MAt4Oj92JSyVb7">
    <img src="https://ws1.sinaimg.cn/large/006tNc79ly1fzkf2equ8nj30u00u0k4v.jpg" width = "200px"/>
    </a>

1. Web Display: Providing project introduction, video list for watching.

    <img src="https://ws4.sinaimg.cn/large/006tNc79ly1fzkg1ztrekj31jy0u0q7y.jpg" width = "400px"/>

1. utils：The crawler used by the project, as well as the super-resolution method based on the bottom of a positive chip, are in the [utils] folder.

    <img src="https://ws3.sinaimg.cn/large/006tNc79ly1fzkgezm4grj31ps0tu41f.jpg" width = "400px"/>

# presentation：

1. step1: 出三个视频，分别让大家猜是哪首诗。

1. step2:讲技术流程，首先：1、text2entity，获取名词以及动词对应素材库中的素材。2、分割：将人的素材分割出来进行第三步处理，其他素材直接第四步。3、素材动态化，利用open pose + gan + face gan 生成动态素材。4、中国风风格迁移：将背景利用cyclegan（+人物）渲染成中国水墨风格。5、上述素材按照一定顺序整合成为动画（位置+缩放），输出为视频。6、前端展示，后端将视频存入数据库。

1. step3:项目意义：1、教育（故事+图片动画）。2、娱乐（体感交互+趣味竞猜）。3、弘扬传统文化。4、NLP+CV

1. step4: 展望：1、丰富素材库。2、交互（语音+动作）。3、生成动画的画面逻辑（素材的物理意义）。

1. PPT Link:
    
    [[ppt download link]](https://drive.google.com/file/d/1iZsU7W2ic8zwOXUGtq6i9LLJAnK3ieVS/view?usp=sharing)

1. poster
    
    [[poster download link]](https://drive.google.com/file/d/1M_r4NBz63712jqkPsT15mRc9DAniJoag/view?usp=sharing)
    
    <img src="https://ws3.sinaimg.cn/large/006tNc79ly1fzkw86lyzvj30u0190tfs.jpg" width = "350px"/>

# TODO：

1. 前端：2个页面：1、展示页面；2、列表页面。（陈）

1. nlp：分词。（燕）

1. 找古诗对应的素材（名词+动词）：背景+人物：水、山、桥、房屋、月亮等，人物：老年中年青年男女。（邓）

1. 后端：1、根据名词找到素材，对素材进行上述2、3、4步骤处理，写个统一的脚本；（合）2、生成动画，将其添加适当位移缩放，合并成为最终的动画。（陈）

1. github维护，分支操作 （燕）

1. poster，ppt，视频制作 （合）

1. 文末彩蛋，中国风抖音。录凯哥读诗（合）

