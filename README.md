# STYOLOX：售后板件问题识别与人工补标

原图原位浏览、多问题点分类、错例复核、位置框选，以及本地模型训练代码。
当前目标是先找全问题，责任作为后续人工确认的业务关联。

2026-10-16 15:25 更新：已修复右侧分类确认不更新错例复核状态的问题，并按实际确认历史
恢复 90 张已复核状态（待复核 0）。2201 条人工分类、6 条草稿及线上原模型保留。
本次未新增位置框；下方 15:02 的训练条件检查属于此前快照。
保存流程说明见 [修复记录](docs/review-save-fix-20261016.md)。

## 当前状态（2026-10-16 15:02，中国时间）

101 的一致性备份包含 **2200 条人工分类、6 条分类草稿**。
`annotation_documents` 和 `annotation_history` 均为 **0 条**，所以尚无人工确认的位置框，
**本轮没有启动 YOLOX 缺陷检测训练，也没有可报告的检测找回数或漏检数**。
图片级标签不能当成局部位置框；未用自动生成的框冒充人工标注。

现有 YOLOX 环境已准备好，线上仍使用原来的 ConvNeXt 多标签问题建议模型。
本次只读取 101 在线备份并上传代码，没有更换服务、模型或写入人工分类。
汇总见 [检查报告](docs/yolox-readiness-20261016.md)。

## 目录

- `labeler/`：分类网页、错例复核、多框补标、独立补标存储、建议推理与部署脚本。
- `training/`：现有图片级分类训练、分组隔离、无关图片辅助模型与 ONNX 导出。
- `tools/audit_annotations.py`：只读位置标注检查，不训练、不连接线上服务。
- `tests/`：检查工具的筛选规则验证。
- `source-provenance.json`：导入源代码的提交及各文件 SHA-256。

**`training/` 当前是图片级分类代码，不是已完成的 YOLOX 位置检测训练器。**
第三方 YOLOX 源码、权重、原图、订单 JSON、标签数据库、业务凭据均不包含在本仓库。
两个原项目目录继续保留；本仓库是本次整合发布的代码快照。

## 补标与训练规则

1. 在分类工具筛选“错例复核”或“待补标”，选择问题点并拖框。
2. 同图多个可见问题分别框选，选择正确类别；需要修正原分类时先人工修正。
3. 点击“保存补标并下一张”，自动暂存不等于人工确认。
4. YOLOX 候选必须同时满足：有已确认问题点、没有分类草稿、补标已确认且结果为 `boxed`、
   原分类指纹一致、坐标有效、原图 SHA-256 和经 EXIF 纠正后的尺寸一致。
5. 选择了正常、无关或看不清图片状态的，一律不进入问题点训练；独立无关辅助遵循单独授权规则。
6. 候选具备后，仍须按订单、售后单与重复图片进行训练/验证/测试隔离，检查类别覆盖。
   工具不会仅凭图片数量宣布可以训练。

框采用 `exif_transposed_normalized_xywh`：经 EXIF 纠正后的原图归一化坐标，非浏览器缩放坐标。
本仓库的检查工具不导出训练数据，不把任意整图或模型候选框当作人工框。

## 本地使用

已有环境：`D:\视觉检测工作区\envs\yolox\Scripts\python.exe`，RTX 4060 Ti 8GB。
原环境验证过 float32，混合精度尚未通过真实训练验证。

```powershell
# 仅检查冻结的 SQLite 在线备份；不要把备份提交到仓库
python tools/audit_annotations.py --snapshot D:\private\labels.sqlite3 --root \\stdeer\home\本地分析图片\cases --output D:\private\audit.json
python -m unittest discover -s tests -v

# 分类服务独立运行：测试目录与正式数据库分开
python -m pip install -r labeler/requirements.txt
python labeler/app.py --help
```

历史训练用法见 [training/README.md](training/README.md)，分类工具用法见 [labeler/README.md](labeler/README.md)。
其中带时间的指标均为历史实验；部署脚本绑定既有 101 环境，不应在其他服务器直接执行。

后续训练参照 [YOLOX 官方自定义训练说明](https://yolox.readthedocs.io/en/latest/train_custom_data.html)，
使用预训练检测权重，在独立运行目录保存新权重。只按验证集选轮次和候选阈值，测试集最后评估。
报告实际问题总数、找回数、漏检数、误报数及框位置质量；未经明确部署指令不替换线上模型。
