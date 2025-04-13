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

# 创建必要的目录
def create_directories(pdf_name):
    # 创建与PDF文件同名的文件夹
    base_dir = pdf_name.stem  # 不带扩展名的文件名
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
                model="qwen-vl-ocr",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                "min_pixels": 28 * 28 * 4,
                                "max_pixels": 28 * 28 * 1280
                            },
                            {"type": "text", "text": "Read all the text in the image."},
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

# 合并所有文本文件为一个完整文件
def merge_text_files(txt_dir, base_dir, pdf_name):
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
    output_path = os.path.join(base_dir, f"{pdf_name.stem}_完整文本.txt")
    with open(output_path, "w", encoding="utf-8") as output_file:
        for i, file_path in enumerate(sorted_files, 1):
            with open(file_path, "r", encoding="utf-8") as input_file:
                output_file.write(f"\n\n--- 第{i}页 ---\n\n")
                output_file.write(input_file.read())
    
    print(f"已合并所有文本到: {output_path}")
    return output_path

# 主函数
def main():
    # 创建解析器
    parser = argparse.ArgumentParser(description='PDF OCR处理工具')
    parser.add_argument('pdf_path', type=str, help='PDF文件路径')
    
    # 解析参数
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(1)
    
    # 检查文件是否存在
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"错误: 找不到PDF文件 '{pdf_path}'")
        sys.exit(1)
    
    print(f"开始处理PDF文件: {pdf_path}")
    
    # 创建必要的目录
    base_dir, pic_dir, txt_dir = create_directories(pdf_path)
    
    # 创建高级速率限制器
    rate_limiter = AdvancedRateLimiter()
    
    # 获取PDF总页数
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
    except Exception as e:
        print(f"打开PDF文件时出错: {e}")
        sys.exit(1)
    
    print(f"PDF共有{total_pages}页")
    
    # 准备参数
    max_dim = 1500
    args_list = [(pdf_path, i, max_dim, pic_dir, txt_dir, rate_limiter) for i in range(total_pages)]
    
    # 使用线程池并行处理：转换图片并立即进行OCR
    max_workers = 30  # 更高的并发数，由速率限制器控制实际并发
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = [executor.submit(process_page, args) for args in args_list]
        
        # 等待所有任务完成
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
    
    # 合并所有文本文件
    merged_file = merge_text_files(txt_dir, base_dir, pdf_path)
    
    print(f"所有页面处理完成，完整文本文件保存在: {merged_file}")

if __name__ == "__main__":
    main()