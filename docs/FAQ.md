# 常见问题解答（FAQ）

路径与排查类问题的速查。完整配置说明见 README[「配置」](../README.md#配置)，磁盘清理见[「磁盘占用与清理」](../README.md#磁盘占用与清理)。

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

## 还有问题？

先看 README[「已知限制」](../README.md#已知限制)与[置顶 issue](../../issues)；带上下文日志（打码后）开新 issue。
