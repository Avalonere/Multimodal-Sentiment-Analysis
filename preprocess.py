import logging
from pathlib import Path
from typing import Optional

import chardet

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EncodingConverter:
    """
    编码转换器，用于将目录下的所有txt文件转换为指定编码格式
    """
    def __init__(self, source_dir: str, target_encoding: str = 'utf-8'):
        """
        初始化编码转换器

        Args:
            source_dir (str): 源文件目录
            target_encoding (str): 目标编码格式，默认UTF-8
        """
        self.source_dir = Path(source_dir)
        self.target_encoding = target_encoding
        # 定义常见的编码列表，按照尝试的优先顺序排列
        self.encoding_candidates = [
            'utf-8', 'gb18030', 'gb2312', 'gbk', 'big5',
            'euc-jp', 'shift-jis', 'euc-kr', 'latin1',
            'windows-1252', 'iso-8859-1'
        ]

    def read_with_fallback_encodings(self, file_path: Path) -> tuple[Optional[str], Optional[str]]:
        """
        使用多个编码尝试读取文件

        Args:
            file_path (Path): 文件路径

        Returns:
            tuple[Optional[str], Optional[str]]: (文件内容, 成功的编码)
        """
        # 首先尝试使用chardet检测
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                detected = chardet.detect(raw_data)
                if detected['confidence'] > 0.8:
                    detected_encoding = detected['encoding']
                    if detected_encoding:
                        try:
                            content = raw_data.decode(detected_encoding)
                            return content, detected_encoding
                        except Exception:
                            pass
        except Exception:
            pass

        # 如果chardet检测失败，尝试预定义的编码列表
        for encoding in self.encoding_candidates:
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                    # 使用errors='replace'来替换无法解码的字符
                    decoded_content = content.decode(encoding, errors='replace')
                    # 检查是否有太多替换字符（通常表示解码不正确）
                    if decoded_content.count('�') / len(decoded_content) < 0.1:
                        return decoded_content, encoding
            except Exception:
                continue

        return None, None

    def convert_file(self, file_path: Path) -> bool:
        """
        转换单个文件的编码

        Args:
            file_path (Path): 文件路径

        Returns:
            bool: 转换是否成功
        """
        try:
            # 使用改进的读取方法
            content, source_encoding = self.read_with_fallback_encodings(file_path)

            if not content or not source_encoding:
                logger.warning(f"无法成功读取文件 {file_path}，跳过此文件")
                return False

            # 使用新编码保存文件
            with open(file_path, 'w', encoding=self.target_encoding, errors='replace') as f:
                f.write(content)

            logger.info(f"成功处理并转换文件 {file_path}")
            return True

        except Exception as e:
            logger.error(f"转换文件 {file_path} 时出错: {str(e)}")
            return False

    def convert_directory(self) -> tuple[int, int]:
        """
        转换目录下所有txt文件的编码

        Returns:
            tuple[int, int]: (成功转换的文件数, 失败的文件数)
        """
        if not self.source_dir.exists():
            raise FileNotFoundError(f"目录 {self.source_dir} 不存在")

        success_count = 0
        failed_count = 0

        # 递归遍历所有txt文件
        for file_path in self.source_dir.rglob("*.txt"):
            if self.convert_file(file_path):
                success_count += 1
            else:
                failed_count += 1

        return success_count, failed_count


def main():
    converter = EncodingConverter("./data/data", "utf-8")
    try:
        success, failed = converter.convert_directory()
        logger.info(f"转换完成！成功: {success} 个文件, 失败: {failed} 个文件")
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")


if __name__ == "__main__":
    main()
