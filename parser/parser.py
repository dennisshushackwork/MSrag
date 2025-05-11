"""
Docling parser: This parser uses the open source docling parser.
See: https://github.com/DS4SD/docling
"""

# External imports:
import logging
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend

# Defining Logging:
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DoclingParser:
    """PDF Parser that uses Docling for conversion"""
    def __init__(self):
        # Defining the docling options:
        pipeline_options=PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True
        pipeline_options.ocr_options.lang = ["en"]
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=8, device=AcceleratorDevice.CUDA # Please change if Cuda is not available
        )
        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options, backend=DoclingParseV4DocumentBackend)
            }
        )

    def parse(self, path: Path) -> str:
        """Parses the document and returns the string representation of it."""
        logger.info("Parsing the uploaded PDF document!")
        conv_result = self.doc_converter.convert(path)
        doc = conv_result.document.export_to_markdown()
        # Save the document in the Postgres Database:
        return doc




