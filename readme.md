# img2txt-qwen

## 项目介绍

img2txt-qwen 是一个通过`qwen-vl-ocr`模型对 PDF 文件进行 OCR 处理的程序，可处理含有文本和公式的 PDF 文件，并生成一个 txt 文件，常用于 RAG 等应用场景。相较于阿里云提供的 OCR 服务，该方案所提取内容的准确度更高，可提取公式，并且处理成本约为阿里云 OCR 服务的 12.5%，即 1RMB/100页（估测数据）。

## 环境要求

1. 需要将阿里云百炼的 API-KEY 保存到环境变量`DASHSCOPE_API_KEY`中。关于如何申请，请参阅[这篇文章](https://github.com/NatsUIJM/autoContents/blob/main/docs/如何申请云服务账户.md)1.1部分的前4条。如果有高校学生或教师身份，可参见1.3条来获取一些优惠；关于如何配置环境变量，请参阅[这篇文章](https://github.com/NatsUIJM/AICoder/blob/main/simplifiedDocs/环境变量配置方法.md)。

2. 需要简易的Python运行环境，请参阅[这篇文章](https://github.com/NatsUIJM/AICoder/blob/main/simplifiedDocs/Python%20运行环境配置方法.md)。

## 安装方法

1. 下载程序

```bash
git clone https://www.github.com/NatsUIJM/qwenOCRbyUIJM
```

2. 安装 ffmpeg

    - Windows

    ```powershell
    choco install ffmpeg
    ```

    - macOS

    ```bash
    brew install ffmpeg
    ```

3. 创建虚拟环境

```powershell
python -m venv .venv
```

4. 激活虚拟环境
    - Windows

    ```powershell
    .venv\Scripts\activate
    ```

    - macOS

    ```bash
    source .venv/bin/activate
    ```

5. 安装依赖

```bash
pip install -r requirements.txt
```

6. 运行程序

```bash
python process.py /path/to/your/file.pdf
```
