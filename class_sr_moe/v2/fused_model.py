import torch
import torch.nn as nn
from model_sr import SRNet
from model_class import ClassNet


class AdaptiveSRSystem(nn.Module):
    def __init__(self, num_classes, sr_scale=2, pretrained_classifier_path=None, training_experts_only=False):
        super(AdaptiveSRSystem, self).__init__()

        self.num_classes = num_classes
        self.scale = sr_scale
        self.training_experts_only = training_experts_only

        # 初始化分类器
        if not training_experts_only:
            self.classifier = ClassNet(class_nums=num_classes)
            if pretrained_classifier_path:
                state_dict = torch.load(pretrained_classifier_path, map_location='cpu')
                self.classifier.load_state_dict(state_dict)

            # 冻结参数
            for param in self.classifier.parameters():
                param.requires_grad = False
            self.classifier.eval()
        else:
            self.classifier = None

        # 专家模块
        # self.sr_experts = nn.ModuleList([
        #     SRNet(scale_factor=sr_scale) for _ in range(num_classes)
        # ])
        self.sr_experts = nn.ModuleList([
            SRNet(scale_factor=sr_scale),
            SRNet(scale_factor=sr_scale, num_res_blocks=32), #为landscape增加更多残差块以提升性能
            SRNet(scale_factor=sr_scale)
        ])

    def forward(self, x, gt_label=None):
        if gt_label is not None:
            # 训练/验证使用GT标签
            selected_indices = gt_label
            class_logits = None
        else:
            # 推理必须分类
            if self.classifier is None:
                raise RuntimeError("Classifier not initialized! Cannot infer without gt_label.")

            with torch.no_grad():
                class_logits = self.classifier(x)
                selected_indices = torch.argmax(class_logits, dim=1)

        b, c, h, w = x.shape
        # 初始化全0的输出张量
        final_output = torch.zeros((b, c, h * self.scale, w * self.scale), device=x.device, dtype=x.dtype)

        # 获取当前Batch出现的所有类别
        unique_classes = torch.unique(selected_indices)

        for cls_idx in unique_classes:
            # 生成掩码，找出属于当前类别cls_idx的图片
            mask = (selected_indices == cls_idx)
            # 提取图片
            input_subset = x[mask]
            # 使用对应的专家网络
            expert_output = self.sr_experts[cls_idx](input_subset)
            # 将结果填回大张量
            final_output[mask] = expert_output

        return final_output, class_logits
