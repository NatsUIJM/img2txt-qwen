0. 预先准备：需要将阿里云百炼的 API-KEY 保存到环境变量`DASHSCOPE_API_KEY`中，关于如何申请，请参阅[这篇文章](https://github.com/NatsUIJM/autoContents/blob/main/docs/如何申请云服务账户.md)1.1部分的前4条。

1. 下载程序

```bash
git clone https://www.github.com/NatsUIJM/qwenOCRbyUIJM
```

2. 创建虚拟环境

```powershell
python -m venv .venv
```

3. 激活虚拟环境
    - Windows

    ```powershell
    .venv\Scripts\activate
    ```

    - macOS

    ```bash
    source .venv/bin/activate
    ```

4. 安装依赖

```bash
pip install -r requirements.txt
```

5. 运行程序

```bash
python process.py /path/to/your/file.pdf
```