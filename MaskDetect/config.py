# 分类损失函数的设置
# 注意这三个损失函数只能选其一使用
# 设置是否使用交叉熵分类损失函数
cross_entropy_loss = True
# 设置是否使用标签平滑分类损失函数
label_smoothing_loss = False
# 设置是否使用focal loss分类损失函数
focal_loss = False

# 网络结构参数的设置
# 二者可以同时开启
# 设置是否使用FPN机制
fpn = False
# 设置是否使用SE模块
se = False

# 数据增强策略
# 设置是否使用MixUp增强以及增强时使用的随机概率
is_mix_up = False
mix_up_prob = 0.2
# 设置是否使用Mosaic增强以及增强时使用的随机概率
is_mosiac = False
mosiac_prob = 0.2
# 设置是否使用小目标随机复制粘贴增强以及增强时使用的随机概率
is_random_paste = False
paste_prob = 0.2
paste_area_threshold = 0.01

# 设置是否使用soft-nms后处理-软阈值以及评分衰减方式
is_soft_nms = False
soft_threshold = 0.01
nms_method = "guassian"

# 设置前50轮每5轮存储一次检查点
# 默认后50轮每轮存储一次
freeze_save_period = 5
