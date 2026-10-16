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

## 人工纠错与局部正常区域辅助训练

`prepare_detector.py --feedback-json feedback-export.json` 可读取分类工具 `/api/annotation-feedback/export`。程序仍以同一 SQLite 快照的原分类和 NAS 源文件为准，重验问题点、图片状态、分类草稿、原始分类 JSON 指纹、源 SHA 和 EXIF 后尺寸；导出中的布尔标记不代替本地核验。人工确认的正框仍由正常流程进入 `records`。

已确认的工艺孔、正常结构等局部误报单独放在 `negative_feedback_records`，不生成整图正常标签，也不伪造正框。其区域内的非正 anchor 恢复加权负 objectness 监督；未框选的其余区域保持忽略，正 anchor 的损失不被正常区域覆盖。每个辅助样本每轮固定出现 3 次、区域权重 2，水平翻转和 resize 同步作用于图像及区域。与人工正框重叠的正常反馈排除。工艺孔形状本身不作为几何过滤条件，孔边真实崩缺仍可预测。

局部反馈只允许训练组使用；与当前或历史验证/测试存在同订单、同售后、相同 SHA 或保守近重复关系的反馈排除。数据集的 `included_feedback_ids` 仅记录实际参与训练的整数反馈 ID，包括已进入训练组的人工正框纠错。反馈不合格或属于留出组时不会标为已收录。

```powershell
python prepare_detector.py --run ./runs/new --snapshot ./runs/new/source/labels.sqlite3 --historical ./runs/previous/dataset.json --feedback-json ./feedback-export.json
python train_detector.py --dataset ./runs/new/dataset.json --run ./runs/new/detector --epochs 50 --seconds 330 --batch-size 2 --input-size 640 --adapt-backbone --defer-test --initialize-from ./runs/previous/detector
python train_detector.py --dataset ./runs/new/dataset.json --run ./runs/new/detector --compare-validation-to ./runs/previous/detector
```

`--initialize-from` 按问题名称映射旧类别输出通道，新增类别保留新初始化。比较命令只读同一新验证集，旧模型未知类别仍计额外候选，不因类别重排漏计。输出 `validation-comparison.json` 中 `baseline` / `candidate` 的 TP/FN/FP、F2、召回率，以及 `test_used: false`。自动循环使用 `--defer-test`，不反复查看测试集；验证不退步只是晋级必要条件，小样本验证通过也不能替代独立验收。

### 本轮有限实验结果

本轮 78 张人工确认图片、98 个正框、10 类问题；训练 55 张 / 71 框、验证 12 张 / 13 框、测试 11 张 / 14 框，另有 1 张 / 1 个明确正常局部 ROI。保留上一轮已知留出划分；本轮测试仍属于探索性复用。训练从上一轮位置检测权重迁移，FP32 640 输入，50 轮耗时约 206 秒。只按验证选第 35 轮和 0.3 阈值。

验证集找回 3、漏 10、额外候选 4；同一验证集旧模型各自按预设网格选阈值时找回 3、漏 10、额外候选 58。固定 0.05 对照中旧模型找回 2 / 额外候选 31，新模型找回 3 / 额外候选 11，说明变化来自实际权重训练。

但最终测试找回 **0 / 14**、额外候选 4，不满足使用要求。为定位原因，额外做了预先指定的固定 0.05 诊断，仍找回 0 / 14，因此并非单纯阈值过高。旧模型在同一新测试集按原 0.05 找回 2 / 14、额外候选 42；其当前验证选定的 0.01 找回 3 / 14、额外候选 68。上述诊断没有改变所选轮次或阈值，也没有继续调参训练。本轮新模型不应晋级为主候选模型。

已纠正的局部 ROI 上最大问题分数从约 0.395 降至 0.00000262，这是训练区域拟合检查，不能宣称已经识别所有正常孔。问题类别和位置在留出图上的泛化仍弱，继续依赖人工审核；已有分类、人工框和线上分类模型不受训练修改。
