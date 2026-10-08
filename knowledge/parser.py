"""
文档解析器：
将不同格式的文件（如 PDF、DOCX、Excel、图片等）
统一转换为 Markdown 格式
"""
import logging
import os
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from langchain_community.document_loaders import PyPDFLoader


_docling_converter = None
_docling_lock = threading.Lock()


class BaseParser(ABC):
    """
    基础解析器类
    """

    def __init__(self,file_path:str):
        """
        初始化解析器，提取文件名和拓展名
        Args:
        file_path: 文件路径
        """
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.file_ext = os.path.splitext(file_path)[1]

    @abstractmethod
    def parse(self,params:dict | None = None) -> str:
        """
        解析后的 Markdown内容
        """
        pass

    def _validate_file(self) -> bool:
        """验证文件是否存在"""
        if not os.path.exists(self.file_path):
            return False
        if not os.path.isfile(self.file_path):
            return False
        return True

    def image_storage_params(self,params:Optional[dict] = None) -> tuple[str,str]:
        """
        设置图片云存档的格式
        """
        params = params or {}

        image_bucket = params.get("image_bucket") or "public"
        image_prefix = params.get("image_prefix")

        if image_prefix:
            normalized_prefix = str(image_prefix).strip("/")
            if normalized_prefix:
                return image_bucket, normalized_prefix

        return image_bucket, "unknown/kb-images"

class TextParser(BaseParser):
    """
    TXT文本解析器
    """
    def parse(self) -> str:
        if not self._validate_file():
            return f"错误：文件不存在或无法读取"
        try:
            with open(self.file_path,'r',encoding='utf-8') as f:
                content = f.read()

            lines = content.split('\n')
            markdown_lines =[]

            for line in lines:
                # 如果是空行，保留空行
                if line.strip() == "":
                    markdown_lines.append("\n")
                else:
                    markdown_lines.append(line)

            return '\n'.join(markdown_lines)

        except Exception as e:
            return f"文件解析错误：{e}"

class MarkdownParser(BaseParser):
    """
    Markdown 解析器
    """
    def parse(self) -> str:
        if not self._validate_file():
            return f"错误：文件不存在或无法读取"
        try:
            with open(self.file_path,'r',encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            return f"文件解析错误：{e}"



class ImageParser(BaseParser):
    """
    image 解析器
    """

    def __init__(self,image_path:str):
        self.image_path = image_path
        self._ocr_engine = None

    def _get_ocr_engine(self):
        """
        懒加载 PaddleOCR 引擎
        只在首次调用时初始化，避免导入错误导致整个模块不可用
        """
        if self._ocr_engine is None:
            try:
                from paddleocr import PaddleOCR
                self._ocr_engine = PaddleOCR(
                    use_angle_cls=True,   # 启用方向分类
                    lang="ch",            # 中英文混合
                    show_log=False        # 关闭 PaddleOCR 内部日志
                )
                logging.info("PaddleOCR 引擎初始化成功")
            except ImportError as e:
                logging.error(f"PaddleOCR 未安装，请运行: pip install paddleocr")
                raise ImportError("请安装 paddleocr: pip install paddleocr") from e
            except Exception as e:
                logging.error(f"PaddleOCR 初始化失败: {e}")
                raise
        return self._ocr_engine


    def parse(self,img_dir:str):
        """
        图片OCR识别
        """
        if not os.path.exists(self.image_path):
            logging.warning(f"图片文件不存在: {self.image_path}")
            return ""

        if not os.access(self.image_path, os.R_OK):
            logging.warning(f"图片文件不可读: {self.image_path}")
            return ""

        ocr = self._get_ocr_engine()

        try:
            result = ocr.ocr(self.image_path,cls=True)
            if result is None or len(result) == 0:
                logging.info("未识别到文字")
                return ""
            text_lines = []
            for line in result:
                # PaddleOCR 返回格式：
                # line = [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], ('文字', confidence)]
                if line and len(line) >= 2:
                    # 提取文字和置信度
                    text = line[1][0]  # 文字内容
                    confidence = line[1][1]  # 置信度

                    # 过滤低置信度结果（可选，根据实际情况调整阈值）
                    if confidence >= 0.5:
                        text_lines.append(text)

            if not text_lines:
                logging.info("未识别到有效文字（置信度过低）")
                return ""

            full_text = "\n".join(text_lines)
            logging.info(f"OCR 识别成功，共 {len(text_lines)} 行")
            return full_text
        except Exception as e:
            logging.error(f"OCR 识别错误: {e}")
            return ""


    # def _parse_data_url(data_uri: str) -> str:
    #     """
    #     工具：将base64编码的图片转换为Markdown格式的图片链接
    #     """


class DocxParser(BaseParser):
    """
    DOCX/xlsx/pptx 解析器
    """
    def _get_docling_converter(self):
        """
        获取Docling 文档转换器单例
        """
        global _docling_converter
        if _docling_converter is None:
            _docling_converter = DocumentConverter(
                format_options = {
                InputFormat.DOCSX: None,
                InputFormat.XLSX: None,
                InputFormat.PPTX: None,
                }
            )
        return _docling_converter

    def parse(self):
        """
        主方法：优先使用Docling，解析失败则使用python
        """
        if not self._validate_file():
            return f"错误：文件不存在或无法读取"

        try:
            return self.docx_parser_with_docling()
        except Exception as e:
            logging.warning(f"Docling 转换错误：{e}，降级使用 python 解析器")
            return self.docx_parser_with_python()

    def docx_parser_with_docling(self, params:dict | None = None) -> str:
        """
        使用 Docling 将 docx/xlsx/pptx 转换为 Markdown。
        """
        params = params or {}
        image_bucket, image_prefix = self.image_storage_params()

        with _docling_lock:
            converter = self._get_docling_converter()
            result = converter.convert(self.file_path)

        if result.status.name != "SUCCESS":
            raise ValueError(f"Docling 转换失败，状态码 {result.status.name}")

        doc = result.document
        # # 是否包含图片
        # if hasattr(doc, "pictures") and len(doc.pictures) > 0:
        #     replacements: list[str] = []
        #     for pic in doc.pictures:
        #         uri = str(pic.image.uri) if hasattr(pic, "image") and hasattr(pic.image, "uri") else ""
        #         if uri.startswith("data:"):

        return doc.export_to_markdown()

    def docx_parser_with_python(self):
        """
        DOCX 解析器
        使用python兜底
        """
        if not self._validate_file():
            return f"错误：文件不存在或无法读取"
        try:
            from docx import Document

            doc = Document(self.file_path)
            paragraphs = []

            for para in doc.paragraphs:
                text = para.text
                # 根据段落样式添加Markdown格式
                style = para.style.name.lower()

                if style in ["heading1", "heading 1"]:
                    paragraphs.append(f"# {text}")
                elif style in ["heading2", "heading 2"]:
                    paragraphs.append(f"## {text}")
                elif style in ["heading3", "heading 3"]:
                    paragraphs.append(f"### {text}")
                elif style in ["heading4", "heading 4"]:
                    paragraphs.append(f"#### {text}")
                else:
                    paragraphs.append(text)

            return '\n\n'.join(paragraphs)

        except ImportError:
            return f"依赖错误，需要安装 python-docx 库，使用 pip install python-docx 安装"
        except Exception as e:
            return f"文件解析错误：{e}"

class CsvParser(BaseParser):
    """
    CSV 解析器
    """
    def parse(self):
        """
        使用pandas 解析CSV 解析器
        """
        if not self._validate_file():
            return f"错误：文件不存在或无法读取"
        try:
            import pandas as pd
            dataframe = pd.read_csv(self.file_path)
            # tables: list[str] = []
            # for i in range(len(dataframe)):
            #     row_dataframe = dataframe.iloc[i]
            #     tables.append(row_dataframe.to_markdown(index=False))
            # return '\n\n'.join(tables)
            return dataframe.to_markdown(index=False)

        except ImportError:
            return f"依赖错误，需要安装 pandas 库，使用 pip install pandas 安装"
        except Exception as e:
            return f"文件解析错误：{e}"

class PdfParser(BaseParser):
    """
    PDF 解析器
    """
    def parse(self):
        """
        pdf 读取器
        """
        if not self._validate_file():
            return f"错误：文件不存在或无法读取"
        if isinstance(self.file_path, str):
            file_path = Path(self.file_path)

        assert file_path.suffix == '.pdf', f"文件格式错误，需要是PDF文件，当前文件格式为：{file_path.suffix}"

        loader = PyPDFLoader(file_path)
        dos = loader.load()
        text = "\n\n".join([d.page_content for d in dos])

        # 扫描版 / 图片型 PDF：文本层为空或近乎为空时，回退到 OCR 识别
        if text and text.strip():
            return text
        logging.info("PDF 文本层为空，回退到 OCR 解析")
        return self.pdf_ocr_parse()

    def pdf_ocr_parse(self, params: dict | None = None) -> str:
        """
        采用PaddleOCR解析PDF文件（适用于扫描版/图片型PDF）

        实现流程：
            1. 使用 pdf2image 将 PDF 每页转换为 PIL 图像
            2. 通过 PaddleOCR 对每页图像进行 OCR 识别
            3. 将每页识别结果按 Markdown 格式汇总输出

        Args:
            params: 可选参数，支持：
                - min_page: int   起始页（从1开始，默认1）
                - max_page: int   结束页（默认-1表示全部）
                - lang: str       OCR语言（默认"ch"）
                - confidence: float 置信度阈值（默认0.5）

        Returns:
            Markdown 格式的解析结果字符串
        """
        if not self._validate_file():
            return "错误：文件不存在或无法读取"

        params = params or {}
        min_page = params.get("min_page", 1)
        max_page = params.get("max_page", -1)
        lang = params.get("lang", "ch")
        confidence = params.get("confidence", 0.5)

        # 1. 懒加载 OCR 引擎
        if not hasattr(self, "_ocr_engine") or self._ocr_engine is None:
            try:
                from paddleocr import PaddleOCR
                self._ocr_engine = PaddleOCR(
                    use_angle_cls=True,
                    lang=lang,
                    show_log=False
                )
                logging.info("PaddleOCR 引擎初始化成功")
            except ImportError as e:
                logging.error("PaddleOCR 未安装，请运行: pip install paddleocr")
                raise ImportError("请安装 paddleocr: pip install paddleocr") from e
            except Exception as e:
                logging.error(f"PaddleOCR 初始化失败: {e}")
                raise

        # 2. 将 PDF 转换为图像
        try:
            from pdf2image import convert_from_path
        except ImportError as e:
            return (
                "依赖错误：需要安装 pdf2image 库。\n"
                "请执行: pip install pdf2image\n"
                "同时需要系统安装 Poppler (Windows 用户请安装 poppler 并添加到 PATH)"
            )

        try:
            images = convert_from_path(self.file_path, dpi=300)
        except Exception as e:
            logging.error(f"PDF 转图像失败: {e}")
            return f"PDF 转图像失败：{e}"

        total_pages = len(images)
        max_page = total_pages if max_page == -1 else min(max_page, total_pages)

        if min_page < 1 or min_page > total_pages:
            min_page = 1

        # 3. 逐页 OCR 识别
        markdown_pages = []
        ocr = self._ocr_engine

        for page_idx in range(min_page - 1, max_page):
            page_num = page_idx + 1
            logging.info(f"正在识别第 {page_num}/{total_pages} 页...")

            try:
                # PaddleOCR 支持直接传入 PIL.Image
                result = ocr.ocr(images[page_idx], cls=True)

                if result is None or len(result) == 0:
                    page_text = ""
                else:
                    text_lines = []
                    for line in result:
                        # line = [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], ('文字', confidence)]
                        if line and len(line) >= 2:
                            text = line[1][0]
                            conf = line[1][1]
                            if conf >= confidence:
                                text_lines.append(text)
                    page_text = "\n".join(text_lines)

                # 按页组织为 Markdown
                if page_text.strip():
                    markdown_pages.append(f"## 第 {page_num} 页\n\n{page_text}")
                else:
                    markdown_pages.append(f"## 第 {page_num} 页\n\n*（未识别到文字）*")

            except Exception as e:
                logging.error(f"第 {page_num} 页识别失败: {e}")
                markdown_pages.append(f"## 第 {page_num} 页\n\n*识别失败：{e}*")

        # 4. 汇总结果
        header = f"# {self.file_name}（OCR 识别结果，共 {total_pages} 页）\n\n"
        full_markdown = header + "\n\n".join(markdown_pages)

        logging.info(f"OCR PDF 解析完成，共处理 {max_page - min_page + 1} 页")
        return full_markdown


class ParserFactory:
    """
    解析器工具类：根据文件拓展名创建相应的解析器
    """
    _parser_map = {
        '.txt': TextParser,
        '.md': MarkdownParser,
        '.pdf': PdfParser,
        '.docx': DocxParser,
        '.xlsx': DocxParser,
        '.ppt': DocxParser,
        '.csv': CsvParser,
        '.jpg': ImageParser,
        '.jpeg': ImageParser,
        '.png': ImageParser,
    }

    @classmethod
    def get_parser(cls, file_path: str) -> BaseParser:
        """
        根据文件路径名获取相应的解析器
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        if ext in cls._parser_map:
            return cls._parser_map[ext](file_path)
        else:
            raise ValueError(f"不支持的文件格式：{ext}")


    @classmethod
    def is_supported(cls, file_path: str):
        """
        检查文件格式是否支持
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        return ext in cls._parser_map

    @classmethod
    def get_supported_extensions(cls):
        """
        获取所有支持的文件扩展名
        """
        return list(cls._parser_map.keys())
