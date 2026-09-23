# 常见问题解答（FAQ）

路径与常见报错的速查。完整配置说明见 README[「配置」](../README.md#配置)，磁盘清理见[「磁盘占用与清理」](../README.md#磁盘占用与清理)。

## 配置文件在哪？

`~/.config/jev-jarvis/env`（env 格式，本项目只有这一种配置格式，没有 config.json）

```bash
cat ~/.config/jev-jarvis/env
```

⚠️ **这个文件里有你的 API key，把内容贴到 issue 或群里之前，先把 key 打码。**

## 日志在哪？

`~/Library/Logs/jev-jarvis.log`

```bash
tail -f ~/Library/Logs/jev-jarvis.log    # 实时滚动
tail -40 ~/Library/Logs/jev-jarvis.log   # 最近 40 行，贴 issue 用这个
```

日志刻意**不含消息正文与候选回复文字**，可以放心整段贴进 issue；反馈时说明当时在做什么（启动 / 首条消息 / 填入…）更好定位。

## 本地判断模型在哪？

`~/.cache/huggingface/hub/models--Mapika--decider-2b`

- 落盘实际占用 **约 3.8 GB**（实测）；查看占用、删除模型都在应用内：**模型设置 →「判断 · Jev」页**
- 删除后走本地判断会重新下载；不想下载可配置 `TYPESAFE_API_KEY` 走云端判断
- 若设置过 `HF_HUB_CACHE` 或 `HF_HOME` 环境变量，模型位置以环境变量为准

命令行提醒：`du -sh` 这个模型子目录会读出**偏小甚至接近 0** 的数字（HuggingFace Xet 缓存布局，实体 blob 存在模型目录之外），别用它判断「模型没下完」；以设置页显示的占用为准。

## 安装时提示「已损坏，无法打开，你应该将它移到废纸篓」？

浏览器下载的 zip 常见（Gatekeeper 隔离属性），右键打开也绕不过，**别删**——终端清掉隔离属性即可：

```bash
sudo xattr -r -d com.apple.quarantine /Applications/jev-jarvis.app
```

`.app` 改过名（如「jev-jarvis 2.app」）就把命令里的路径换成实际名字。装好后首次启动还需授予「屏幕录制」与「辅助功能」权限，详见 README[「只想用」](../README.md#只想用)一节。

## 启动后悬浮窗一片空白、没有任何提示？

这是**旧版本**的现象：新版本首次启动会弹出判断方式引导，模型下载/加载期间面板状态行有实时进度（如「下载判断模型 34% · 1.2/3.8 GB」），失败也有红字说明。遇到一片白先确认版本，**推荐更新到[最新版](https://github.com/jev-chat/jev-chat-jarvis-mac/releases/latest)**。新版本发布会在微信群与公众号通知，入口见 README 文末[「交流反馈」](../README.md#交流反馈)，建议关注以便第一时间收到更新。

## 还有问题？

先看 README[「已知限制」](../README.md#已知限制)与[置顶 issue](../../issues)；带上下文日志（打码后）开新 issue。数据收集与隐私说明：[PRIVACY.md](../PRIVACY.md)。
