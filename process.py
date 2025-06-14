import os
import base64
import time
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io
import math
from openai import OpenAI
import concurrent.futures
import threading
import queue
import shutil
import glob
import re
import argparse
import sys
import imghdr  # For image type detection

# 创建必要的目录
def create_directories(file_name):
    # 创建与文件同名的文件夹
    base_dir = file_name.stem  # 不带扩展名的文件名
    os.makedirs(base_dir, exist_ok=True)
    
    # 在该文件夹中创建pic和txt子文件夹
    pic_dir = os.path.join(base_dir, 'pic')
    txt_dir = os.path.join(base_dir, 'txt')
    os.makedirs(pic_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)
    
    return base_dir, pic_dir, txt_dir

# 读取本地文件，并编码为 BASE64 格式
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 高级速率限制器
class AdvancedRateLimiter:
    def __init__(self, qpm_limit=600, tpm_limit=2400000):
        self.qpm_limit = qpm_limit
        self.tpm_limit = tpm_limit
        self.query_count = 0
        self.token_count = 0
        self.last_reset = time.time()
        self.lock = threading.Lock()
        
        # 动态调整并发数
        self.concurrent_limit = min(20, qpm_limit // 10)  # 初始并发数，保守估计
        
        # 跟踪处理速度
        self.processed_in_window = 0
        self.window_start = time.time()
        
        # 用于控制请求发送的信号量
        self.semaphore = threading.Semaphore(self.concurrent_limit)
    
    def adjust_concurrency(self):
        """根据处理情况动态调整并发度"""
        current_time = time.time()
        window_duration = current_time - self.window_start
        
        # 至少经过5秒后再调整
        if window_duration >= 5 and self.processed_in_window > 0:
            # 计算当前QPM
            current_qpm = (self.processed_in_window / window_duration) * 60
            
            # 如果远低于限制，增加并发度
            if current_qpm < self.qpm_limit * 0.7:
                new_limit = min(50, self.concurrent_limit + 2)  # 增加，但不超过50
                
                # 如果并发度有变化，更新信号量
                if new_limit > self.concurrent_limit:
                    # 释放额外的信号量
                    for _ in range(new_limit - self.concurrent_limit):
                        self.semaphore.release()
                    self.concurrent_limit = new_limit
                    print(f"增加并发度到 {self.concurrent_limit}")
            
            # 如果接近限制，减少并发度
            elif current_qpm > self.qpm_limit * 0.9:
                new_limit = max(5, self.concurrent_limit - 3)  # 减少，但不低于5
                self.concurrent_limit = new_limit
                print(f"减少并发度到 {self.concurrent_limit}")
            
            # 重置窗口
            self.processed_in_window = 0
            self.window_start = current_time
    
    def wait_if_needed(self, estimated_tokens=4000):
        """获取执行许可，必要时等待"""
        # 获取信号量
        self.semaphore.acquire()
        
        try:
            with self.lock:
                current_time = time.time()
                elapsed = current_time - self.last_reset
                
                # 如果已经过了一分钟，重置计数器
                if elapsed >= 60:
                    self.query_count = 0
                    self.token_count = 0
                    self.last_reset = current_time
                    # 调整并发度
                    self.adjust_concurrency()
                
                # 检查是否超过了限制
                if self.query_count >= self.qpm_limit or self.token_count + estimated_tokens > self.tpm_limit:
                    # 计算需要等待的时间
                    wait_time = 60 - elapsed
                    if wait_time > 0:
                        print(f"达到速率限制，等待 {wait_time:.2f} 秒...")
                        time.sleep(wait_time)
                        # 重置计数器
                        self.query_count = 0
                        self.token_count = 0
                        self.last_reset = time.time()
                
                # 更新计数器
                self.query_count += 1
                self.token_count += estimated_tokens
                self.processed_in_window += 1
        finally:
            # 完成后释放信号量
            self.semaphore.release()

# 处理单个PDF页面：转换为图片后立即进行OCR
def process_page(args):
    pdf_path, page_num, max_dim, pic_dir, txt_dir, rate_limiter = args
    
    try:
        # 步骤1: 转换PDF页面为图片
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        pix = page.get_pixmap(dpi=300)
        
        # 将pixmap转换为PIL Image
        img = Image.open(io.BytesIO(pix.tobytes("jpeg")))
        
        # 计算调整大小后的尺寸
        orig_width, orig_height = img.size
        
        if orig_width >= orig_height:
            new_width = max_dim
            new_height = int(orig_height * (max_dim / orig_width))
        else:
            new_height = max_dim
            new_width = int(orig_width * (max_dim / orig_height))
        
        # 调整图片大小
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # 保存图片
        img_path = os.path.join(pic_dir, f"{page_num+1}.jpg")
        img.save(img_path, "JPEG", quality=95)
        
        doc.close()
        
        print(f"已转换第{page_num+1}页到图片")
        
        # 步骤2: 立即对生成的图片进行OCR处理
        process_image(img_path, page_num+1, txt_dir, rate_limiter)
        
        return page_num + 1
    except Exception as e:
        print(f"处理第{page_num+1}页时出错: {e}")
        return None

# 处理单个图像文件：调整大小后进行OCR
def process_image_file(args):
    image_path, target_num, max_dim, pic_dir, txt_dir, rate_limiter = args
    
    try:
        # 步骤1: 读取图像并调整大小
        img = Image.open(image_path)
        
        # 计算调整大小后的尺寸
        orig_width, orig_height = img.size
        
        if orig_width >= orig_height:
            new_width = max_dim
            new_height = int(orig_height * (max_dim / orig_width))
        else:
            new_height = max_dim
            new_width = int(orig_width * (max_dim / orig_height))
        
        # 调整图片大小
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # 保存图片
        img_path = os.path.join(pic_dir, f"{target_num}.jpg")
        img.save(img_path, "JPEG", quality=95)
        
        print(f"已调整图像大小并保存为 {img_path}")
        
        # 步骤2: 立即对图片进行OCR处理
        process_image(img_path, target_num, txt_dir, rate_limiter)
        
        return target_num
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")
        return None

# OCR处理单个图片
def process_image(image_path, page_num, txt_dir, rate_limiter, max_retries=5):
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # 等待，如果需要的话
            rate_limiter.wait_if_needed()
            
            # 编码图片
            base64_image = encode_image(image_path)
            
            # 初始化OpenAI客户端
            client = OpenAI(
                api_key=os.getenv('DASHSCOPE_API_KEY'),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            
            # 发送请求
            completion = client.chat.completions.create(
                model="qwen-vl-ocr-latest",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                "min_pixels": 28 * 28 * 4,
                                "max_pixels": 28 * 28 * 8192
                            },
                            {"type": "text", "text": """Read all the text in the image. If there are formulas, please use LaTeX to represent them.

"""},
                        ],
                    }
                ],
            )
            
            # 获取结果
            result = completion.choices[0].message.content
            
            # 保存结果
            output_path = os.path.join(txt_dir, f"{page_num}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result)
            
            print(f"已处理第{page_num}页图片并保存结果")
            return True
            
        except Exception as e:
            retry_count += 1
            print(f"处理第{page_num}页时出错 (尝试 {retry_count}/{max_retries}): {e}")
            if retry_count < max_retries:
                # 指数退避策略
                wait_time = 2 ** retry_count
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"第{page_num}页处理失败，已达到最大重试次数")
                # 保存错误信息
                output_path = os.path.join(txt_dir, f"{page_num}.txt")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(f"OCR处理失败: {e}")
                return False

# 获取PDF目录
def get_pdf_toc(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        toc = doc.get_toc()
        doc.close()
        return toc
    except Exception as e:
        print(f"获取PDF目录失败: {e}")
        return []

# 按目录分割文本文件
def split_text_by_toc(merged_file, txt_dir, base_dir, toc):
    if not toc or len(toc) <= 1:
        # 没有目录或目录项太少，直接返回
        return [merged_file]
    
    try:
        # 读取合并后的文本文件
        with open(merged_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 分割内容为各页面
        pages = re.split(r'\n\n--- 第(\d+)页 ---\n\n', content)
        pages = pages[2:]  # 跳过初始的空元素
        
        # 创建目录输出文件夹
        toc_dir = os.path.join(base_dir, 'toc')
        os.makedirs(toc_dir, exist_ok=True)
        
        # 用于存储每个目录项的起始页码
        toc_pages = []
        
        # 提取目录中的页码信息
        for entry in toc:
            level, title, page = entry
            if page >= 0:  # 确保是有效页码
                toc_pages.append((page, title))
        
        # 如果第一个目录项不是第一页，添加一个虚拟目录项
        if toc_pages and toc_pages[0][0] != 0:
            toc_pages.insert(0, (0, "前言"))
        
        # 如果最后一个目录项的页码不是最后一页，添加一个虚拟目录项
        last_page = len(pages) - 1
        if toc_pages and toc_pages[-1][0] < last_page:
            toc_pages.append((last_page, "结语"))
        
        # 按目录项生成文件
        output_files = []
        for i in range(len(toc_pages) - 1):
            start_page, title = toc_pages[i]
            end_page = toc_pages[i+1][0]
            
            # 清理标题作为文件名
            safe_title = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '', title)
            safe_title = safe_title[:100]  # 限制长度
            
            # 防止重复标题
            count = 1
            original_title = safe_title
            while os.path.exists(os.path.join(toc_dir, f"{safe_title}.txt")):
                safe_title = f"{original_title}_{count}"
                count += 1
            
            # 合并相关页面内容
            section_content = ""
            for p in range(start_page, end_page):
                if p < len(pages):
                    section_content += f"\n\n--- 第{p}页 ---\n\n{pages[p]}"
            
            # 保存文件
            output_path = os.path.join(toc_dir, f"{safe_title}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(section_content)
            
            output_files.append(output_path)
        
        print(f"已按目录分割为 {len(output_files)} 个文件，保存在: {toc_dir}")
        return output_files
    
    except Exception as e:
        print(f"按目录分割文件时出错: {e}")
        return [merged_file]

# 合并所有文本文件为一个完整文件
def merge_text_files(txt_dir, base_dir, file_name):
    # 获取所有txt文件
    txt_files = glob.glob(os.path.join(txt_dir, "*.txt"))
    
    # 提取文件名中的数字部分，用于排序
    def extract_number(file_path):
        file_name = os.path.basename(file_path)
        match = re.search(r'(\d+)\.txt', file_name)
        if match:
            return int(match.group(1))
        return 0
    
    # 按页码排序文件
    sorted_files = sorted(txt_files, key=extract_number)
    
    # 合并文件内容
    output_path = os.path.join(base_dir, f"{file_name.stem}_完整文本.txt")
    with open(output_path, "w", encoding="utf-8") as output_file:
        for i, file_path in enumerate(sorted_files, 1):
            with open(file_path, "r", encoding="utf-8") as input_file:
                output_file.write(f"\n\n--- 第{i}页 ---\n\n")
                output_file.write(input_file.read())
    
    print(f"已合并所有文本到: {output_path}")
    return output_path

# 检查文件类型
def check_file_type(file_path):
    # 获取文件扩展名
    ext = os.path.splitext(file_path)[1].lower()
    
    # 检查是否为PDF
    if ext == '.pdf':
        return 'pdf'
    
    # 检查是否为图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif']
    if ext in image_extensions:
        return 'image'
    
    # 如果扩展名不明确，尝试使用imghdr来检测
    if imghdr.what(file_path) is not None:
        return 'image'
    
    # 如果都不是则返回未知类型
    return 'unknown'

# 主函数
def main():
    # 创建解析器
    parser = argparse.ArgumentParser(description='文件OCR处理工具')
    parser.add_argument('file_path', type=str, help='要处理的文件路径(PDF或图像)')
    
    # 解析参数
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(1)
    
    # 检查文件是否存在
    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"错误: 找不到文件 '{file_path}'")
        sys.exit(1)
    
    # 检查文件类型
    file_type = check_file_type(file_path)
    
    if file_type == 'unknown':
        print(f"错误: 不支持的文件类型 '{file_path}'")
        print("支持的文件类型: PDF, JPG, JPEG, PNG, BMP, TIFF, GIF")
        sys.exit(1)
    
    print(f"开始处理{file_type}文件: {file_path}")
    
    # 创建必要的目录
    base_dir, pic_dir, txt_dir = create_directories(file_path)
    
    # 创建高级速率限制器
    rate_limiter = AdvancedRateLimiter()
    
    # 基于文件类型进行不同处理
    max_dim = 1500
    max_workers = 30  # 更高的并发数，由速率限制器控制实际并发
    
    if file_type == 'pdf':
        # 获取PDF总页数
        try:
            doc = fitz.open(file_path)
            total_pages = len(doc)
            doc.close()
        except Exception as e:
            print(f"打开PDF文件时出错: {e}")
            sys.exit(1)
        
        print(f"PDF共有{total_pages}页")
        
        # 准备参数
        args_list = [(file_path, i, max_dim, pic_dir, txt_dir, rate_limiter) for i in range(total_pages)]
        
        # 使用线程池并行处理：转换图片并立即进行OCR
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = [executor.submit(process_page, args) for args in args_list]
            
            # 等待所有任务完成
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
    
    elif file_type == 'image':
        # 处理单个图像文件
        args = (file_path, 1, max_dim, pic_dir, txt_dir, rate_limiter)
        process_image_file(args)
    
    # 合并所有文本文件
    merged_file = merge_text_files(txt_dir, base_dir, file_path)
    
    # 如果是PDF且存在目录，则按目录分割
    if file_type == 'pdf':
        toc = get_pdf_toc(file_path)
        if toc:
            split_text_by_toc(merged_file, txt_dir, base_dir, toc)
    
    if file_type == 'pdf':
        print(f"所有页面处理完成，完整文本文件保存在: {merged_file}")
    else:
        print(f"图像处理完成，文本结果保存在: {merged_file}")

if __name__ == "__main__":
    main()
