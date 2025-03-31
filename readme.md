# ⚠️ 项目状态: 实机测试中 ⚠️

## 📌 开发中
> * 文本 log

## 🚧 待完成
> * 添加快捷键

## ✅ 已完成功能
> * 全物品支持
> * 聚焦与最小化
> * 根据剩余时间触发 (声音提示 + 自动聚焦)
> * dxcam 截图，直接访问显存，彻底解决截图不稳定问题
> * 16:9 的1k, 2k, 4k 屏幕适配
> * 自动制造, 收集: 自动选择匹配物品, 自动开始制造
> * 自动购买材料 (不支持兑换物品)

## 🐦 已咕咕
> * 适配 720p HD：OCR 识别率不够，需要字形库
> * 适配普通窗口模式: 像素转换导致很多函数都要改，不如直接最小化

## ✔ 已修复
> * 截图不稳定: 概率截到纯黑 or 只有背景（win32）

## 🙃 已知问题
> * 后台时间过长可能需要重连网络，不清楚最大间隔时间

# Mar 20, 2025
使用虚拟环境! python 版本 3.9.21 (不用也行 /doge)

### Tesseract
[安装 Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)

选择 install for anyone using this computer

![alt text](.img/image.png)

使用默认安装位置 ("C:\Program Files\Tesseract-OCR")

### requirements
运行以安装所有需要的库:
`pip install -r requirements.txt`