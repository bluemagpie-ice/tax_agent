"""
文档解析器：
将不同格式的文件（如 PDF、DOCX、Excel、图片等）
统一转换为 Markdown 格式
"""
import logging
import os
import threading
import inspect
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
        super().__init__(image_path)
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
                if "use_textline_orientation" in inspect.signature(PaddleOCR).parameters:
                    self._ocr_engine = PaddleOCR(lang="ch", use_textline_orientation=True,
                                                 use_doc_orientation_classify=False, use_doc_unwarping=False,
                                                 enable_mkldnn=False, cpu_threads=4)
                else:
                    self._ocr_engine = PaddleOCR(
                        use_angle_cls=True,   # 启用方向分类
                        lang="ch",            # 中英文混合
                        show_log=False        # 关闭 PaddleOCR 内部日志
                    )
                logging.info("PaddleOCR 引擎初始化成功")
            except ImportError as e:
                raise RuntimeError("OCR 依赖错误：请安装 PaddleOCR 和 CPU版 PaddlePaddle") from e
            except Exception as e:
                logging.warning(f"PaddleOCR 初始化失败: {e}")
                raise RuntimeError(f"OCR 初始化错误：{type(e).__name__}，请检查模型和PaddlePaddle环境") from e
        return self._ocr_engine


    def parse(self,img_dir:str = ""):
        """
        图片OCR识别
        """
        if not os.path.exists(self.image_path):
            logging.warning(f"图片文件不存在: {self.image_path}")
            raise FileNotFoundError("图片文件不存在")

        if not os.access(self.image_path, os.R_OK):
            logging.warning(f"图片文件不可读: {self.image_path}")
            raise PermissionError("图片文件不可读")

        try:
            ocr = self._get_ocr_engine()
            if ocr is None:
                raise RuntimeError("OCR 引擎不可用，请检查运行环境")
            if hasattr(ocr, "predict"):
                text_lines = []
                for page in ocr.predict(self.image_path):
                    text_lines.extend(text for text, score in zip(page.get("rec_texts", []), page.get("rec_scores", []))
                                      if score >= 0.5)
                if not text_lines:
                    raise ValueError("OCR 未识别到有效文字，请上传更清晰的图片")
                return "\n".join(text_lines)
            result = ocr.ocr(self.image_path,cls=True)
            if result is None or len(result) == 0:
                logging.info("未识别到文字")
                raise ValueError("OCR 未识别到文字，请上传更清晰的图片")
            text_lines = []
            for line in (result[0] if len(result) == 1 and result[0] else result):
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
                raise ValueError("OCR 未识别到有效文字，请上传更清晰的图片")

            full_text = "\n".join(text_lines)
            logging.info(f"OCR 识别成功，共 {len(text_lines)} 行")
            return full_text
        except Exception as e:
            logging.error(f"OCR 识别错误: {e}")
            return f"OCR 错误：{type(e).__name__}；{e}"


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
                allowed_formats=[InputFormat.DOCX, InputFormat.XLSX, InputFormat.PPTX]
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
        file_path = Path(self.file_path)

        if file_path.suffix.lower() != '.pdf':
            return f"文件格式错误，需要是PDF文件，当前文件格式为：{file_path.suffix}"

        try:
            loader = PyPDFLoader(str(file_path))
            dos = loader.load()
        except ImportError:
            return "依赖错误：PDF 解析需要 pypdf，请按 README 安装 requirements.txt"
        except Exception as e:
            return f"文件解析错误：{type(e).__name__}，请检查 PDF 是否损坏或加密"
        text = "\n\n".join([d.page_content for d in dos])

        # 扫描版 / 图片型 PDF：文本层为空或近乎为空时，回退到 OCR 识别
        if text and text.strip():
            return "\n\n".join(f"## 第 {i + 1} 页\n\n{d.page_content}" for i, d in enumerate(dos))
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
                if "use_textline_orientation" in inspect.signature(PaddleOCR).parameters:
                    self._ocr_engine = PaddleOCR(lang=lang, use_textline_orientation=True,
                                                 use_doc_orientation_classify=False, use_doc_unwarping=False,
                                                 enable_mkldnn=False, cpu_threads=4)
                else:
                    self._ocr_engine = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
                logging.info("PaddleOCR 引擎初始化成功")
            except ImportError as e:
                logging.error("PaddleOCR 未安装，请运行: pip install paddleocr")
                return "OCR 依赖错误：请检查 PaddleOCR/PaddlePaddle 运行环境，或上传文本版 PDF"
            except Exception as e:
                logging.error(f"PaddleOCR 初始化失败: {e}")
                return "OCR 初始化错误：请检查模型文件和运行环境，或上传文本版 PDF"

        # 2. 将 PDF 转换为图像
        try:
            from pdf2image import convert_from_path, pdfinfo_from_path
        except ImportError as e:
            return (
                "依赖错误：需要安装 pdf2image 库。\n"
                "请执行: pip install pdf2image\n"
                "同时需要系统安装 Poppler (Windows 用户请安装 poppler 并添加到 PATH)"
            )

        try:
            total_pages = int(pdfinfo_from_path(self.file_path, timeout=30)["Pages"])
        except Exception as e:
            logging.error(f"PDF 转图像失败: {e}")
            raise RuntimeError("PDF 转图像失败，请检查文件及Poppler环境") from e

        max_page = total_pages if max_page == -1 else min(max_page, total_pages)
        if total_pages < 1:
            raise ValueError("PDF页码范围无效")

        if min_page < 1 or min_page > total_pages:
            min_page = 1
        if max_page < min_page:
            raise ValueError("PDF页码范围无效")

        # 3. 逐页 OCR 识别
        markdown_pages = []
        recognized_pages = 0
        ocr = self._ocr_engine

        for page_idx in range(min_page - 1, max_page):
            page_num = page_idx + 1
            logging.info(f"正在识别第 {page_num}/{total_pages} 页...")

            images = []
            try:
                images = convert_from_path(self.file_path, dpi=300, first_page=page_num,
                                           last_page=page_num, timeout=60)
                # PaddleOCR 支持直接传入 PIL.Image
                if hasattr(ocr, "predict"):
                    import numpy as np
                    result = ocr.predict(np.asarray(images[0]))
                else:
                    import numpy as np
                    result = ocr.ocr(np.asarray(images[0]), cls=True)

                if result is None or len(result) == 0:
                    page_text = ""
                else:
                    text_lines = []
                    lines = result if hasattr(ocr, "predict") else result[0]
                    for line in lines or []:
                        if isinstance(line, dict) or hasattr(line, "get"):
                            text_lines.extend(text for text, score in zip(line.get("rec_texts", []), line.get("rec_scores", []))
                                              if score >= confidence)
                            continue
                        # line = [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], ('文字', confidence)]
                        if line and len(line) >= 2:
                            text = line[1][0]
                            conf = line[1][1]
                            if conf >= confidence:
                                text_lines.append(text)
                    page_text = "\n".join(text_lines)

                # 按页组织为 Markdown
                if page_text.strip():
                    recognized_pages += 1
                    markdown_pages.append(f"## 第 {page_num} 页\n\n{page_text}")
                else:
                    markdown_pages.append(f"## 第 {page_num} 页\n\n*（未识别到文字）*")

            except Exception as e:
                logging.error(f"第 {page_num} 页识别失败: {e}")
                raise RuntimeError(f"第 {page_num} 页OCR识别失败，请检查图片清晰度与OCR环境") from e
            finally:
                for image in images:
                    image.close()

        # 4. 汇总结果
        if not recognized_pages:
            raise ValueError("扫描PDF未识别到有效文字，文档未入库")
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
        # '.ppt': DocxParser,    # 依赖 docling，暂无纯 Python 降级
        # '.pptx': DocxParser,   # 依赖 docling，暂无纯 Python 降级
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
