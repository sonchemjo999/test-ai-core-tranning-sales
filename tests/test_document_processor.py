"""
Tests for tools/document_processor.py — Phase 1 Ingestion Pipeline.

Run with:
    cd core-ai
    python -m pytest tests/test_document_processor.py -v
"""

import io
import pytest
from tools.document_processor import (
    process_document,
    detect_format,
    DocumentProcessingError,
    _parse_csv,
    _parse_txt,
)


class TestDetectFormat:
    """Test file format detection from filename and content_type."""

    def test_pdf_by_extension(self):
        assert detect_format("report.pdf") == "pdf"

    def test_docx_by_extension(self):
        assert detect_format("guide.docx") == "docx"

    def test_xlsx_by_extension(self):
        assert detect_format("prices.xlsx") == "xlsx"

    def test_csv_by_extension(self):
        assert detect_format("leads.csv") == "csv"

    def test_txt_by_extension(self):
        assert detect_format("notes.txt") == "txt"

    def test_by_content_type(self):
        assert detect_format("unknown", "application/pdf") == "pdf"

    def test_unsupported_raises(self):
        with pytest.raises(DocumentProcessingError, match="Unsupported"):
            detect_format("image.png")


class TestParseCSV:
    """Test CSV → Markdown table conversion."""

    def test_simple_csv(self):
        csv_data = "Tên,Giá,Ghi chú\nCRM Pro,500k,Best seller\nCRM Lite,200k,New"
        result = _parse_csv(csv_data.encode("utf-8"))
        assert "| Tên | Giá | Ghi chú |" in result
        assert "| --- | --- | --- |" in result
        assert "| CRM Pro | 500k | Best seller |" in result
        assert "| CRM Lite | 200k | New |" in result

    def test_csv_with_pipe_chars(self):
        csv_data = "Col A,Col B\nvalue|one,value|two"
        result = _parse_csv(csv_data.encode("utf-8"))
        assert "\\|" in result  # Pipes escaped

    def test_empty_csv_raises(self):
        with pytest.raises(DocumentProcessingError, match="empty"):
            _parse_csv(b"")


class TestParseTxt:
    """Test plain text passthrough."""

    def test_simple_text(self):
        text = "Quy trình bán hàng:\n1. Chào hỏi\n2. Tìm hiểu nhu cầu"
        result = _parse_txt(text.encode("utf-8"))
        assert "Quy trình bán hàng" in result
        assert "1. Chào hỏi" in result

    def test_empty_text_raises(self):
        with pytest.raises(DocumentProcessingError, match="empty"):
            _parse_txt(b"   ")


class TestProcessDocumentIntegration:
    """Integration test for the main process_document function."""

    def test_csv_end_to_end(self):
        csv_data = "Sản phẩm,Giá tháng,Giá năm\nPro,500k,5000k\nLite,200k,2000k"
        result = process_document(
            file_bytes=csv_data.encode("utf-8"),
            filename="bang_gia.csv",
        )
        assert result["format"] == "csv"
        assert result["document_id"]  # Not empty
        assert result["filename"] == "bang_gia.csv"
        assert "| Sản phẩm | Giá tháng | Giá năm |" in result["raw_markdown"]
        assert "| Pro | 500k | 5000k |" in result["raw_markdown"]
        assert result["token_count_estimate"] > 0

    def test_txt_end_to_end(self):
        text = "# Chính sách bảo hành\n\nBảo hành 12 tháng cho mọi sản phẩm."
        result = process_document(
            file_bytes=text.encode("utf-8"),
            filename="chinh_sach.txt",
        )
        assert result["format"] == "txt"
        assert "Chính sách bảo hành" in result["raw_markdown"]
        assert "Bảo hành 12 tháng" in result["raw_markdown"]

    def test_unsupported_format_raises(self):
        with pytest.raises(DocumentProcessingError, match="Unsupported"):
            process_document(
                file_bytes=b"fake",
                filename="image.png",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
