# 公开源码发布包

先检查修改及敏感信息，提交并推送 GitHub，再执行：

```powershell
npm run deploy:check
npm run release:package
```

需要 Python 3.8 及以上、Node.js 和 Git。默认输出 `D:/项目开发/STYOLOX-release-artifacts/styolox-latest.zip` 与 `release.json`；也可通过 `npm run release:package -- --output <目录>` 指定仓库外目录。

打包只读取已发布的 Git 提交，不读取工作目录中的私有数据。包在临时目录通过格式、CRC、源码类型和基本凭据标记检查后才替换最新包。上一份包及版本记录归档到 `history/日期时间-唯一编号/`，历史文件名带同一日期时间。替换发生异常时恢复旧包。

这是公开源码安装包，不含模型、数据库、图片、登录状态和服务器配置；实际部署前还应核对目标服务及源码版本。打包成功不代表已经部署或业务验收成功。
