# 人工框选 YOLOX 位置检测

`train_detector.py` 只读取一致性快照导出的 `dataset.json` 和 NAS 原图，不写分类数据库、不复制原图、不替换 101 现有推荐模型。

筛选规则：只接收人工确认框、问题点非空、无图片状态、无分类草稿且源哈希及 EXIF 后尺寸一致的图片。订单及重复图分组由快照导出程序固化；训练程序再次检查 train/val/test 组互斥。所有类别保留，某组无样本的类别不宣称已验证。

首次实验为 YOLOX-s，COCO 初始化匹配的骨干、回归及目标存在分支，重新初始化问题类别预测层。FP32、512 输入、batch 2；冻结骨干，只训练检测头，使用官方 TrainTransform 的轻量水平翻转及 HSV，不用 mosaic/mixup。默认最多 50 轮或 450 秒。第 20 轮开始增加 YOLOX L1 损失。

部分图片的整图分类问题点尚无对应框；这些图片保留已确认正框，但在目标存在损失中屏蔽未匹配 anchor 的负样本，避免把未框问题当成正常背景。完整图保留原有正负 objectness；分类及位置回归损失不变。不修改第三方 YOLOX 源码。

开始前保存 `planned-config.json`；只按验证集类别匹配且 IoU ≥ 0.5 的 F2 选轮次，并在预先固定的 0.01/0.025/0.05/0.1/0.2/0.3 中选阈值。NMS 0.5，每张最多 10 个候选。测试集仅在选模完成后评分。

产物：`detector.pt`、`selection.json`、`metrics.json`、`training-history.json`、`validation-predictions.json`、`test-predictions.json`。测试报告区分“类别及位置均正确”的 TP/FN/FP，与忽略类别的“仅位置找回”。小样本结果仅为探索性，不与整图分类准确率混算。未训练类别头的初始化基线仅作实验自检，不能当作现有线上模型。

```powershell
python train_detector.py --dataset ./runs/example/dataset.json --run ./runs/example/detector
```

推理接口：`detector = load_detector(run_dir, device='cuda')`，`detector.predict_image(path_or_pil_or_rgb_array, threshold=None)`。返回 `problem`、`class_id`、`score`、像素 `xyxy` 和 EXIF 转正后归一化的 `xywh` / `normalized_xywh`。只能输出待人工审核候选；不得把预测写成人工确认框。默认阈值读取选模产物。

## 一次有限的迁移适配对照

第一轮验证集只找回 1/6 框后，增加 `--adapt-backbone`：从同一 COCO 权重重新开始，训练 neck、Darknet dark4/dark5 和 head，冻结 stem/dark2/dark3，学习率 1e-4。小 batch 下冻结全部 BN 运行统计，参数仍按上述层范围更新。使用 640 输入、batch 2、最多 50 轮或 240 秒。数据、分组、损失部分标注保护与阈值网格不变，输出新目录而不覆盖首轮。

使用 `--defer-test` 先完成验证选模且产出 `metrics.json`（test 为 null），报告验证效果后，再用相同 `--dataset/--run --evaluate-test-only` 评分一次测试集。该入口拒绝重复评分和数据 SHA 变化；不根据测试结果再改配置或阈值。加载器按 checkpoint 保存的 depth/width/source/input_size 构建模型。

## 2026-10-16 探索性实验结果

本轮符合规则的数据为 46 张图片、54 个已确认框、9 类问题。按订单及重复图片分组后，训练集 36 张 / 44 框，验证集 6 张 / 6 框，测试集 4 张 / 4 框。14 张图片仍有整图问题标签尚未对应到位置框，按上述部分标注规则处理。

| 实验 | 验证集找回 / 漏检 / 额外候选 | 测试集找回 / 漏检 / 额外候选 |
| --- | --- | --- |
| 首轮，仅训练检测头 | 1 / 5 / 1 | 0 / 4 / 3 |
| 第二轮，有限骨干适配 | 2 / 4 / 12 | 2 / 2 / 11 |

以上均要求问题类别一致且 IoU ≥ 0.5。第二轮仅依据验证集 F2 选择第 30 轮、阈值 0.05，再执行一次测试评分。测试找回的为其他破损和板件瑕疵各 1 框，漏掉的为崩边 2 框。测试集仅覆盖 3 类，其余 6 类没有测试正样本，不能据此评价识别效果。额外候选按现有人工框计算；部分图片的标注可能仍不完整，因此候选也需要人工复核。

第一轮测试结果已经被查看，第二轮沿用同一测试划分，因此属于探索性复测，不是新的盲测验收。第二轮轮次与阈值只由验证集选定，第二轮测试后没有继续调参。

4 个测试框不足以代表总体准确率或生产效果。本模型仍为实验候选生成器，不替换现有线上模型，不改变既有分类与人工框，也不把未审核候选纳入训练。后续位置建议由 `generate_detector_proposals.py` 生成，仅匹配图片已确认的问题类型，导入后仍须人工确认。
