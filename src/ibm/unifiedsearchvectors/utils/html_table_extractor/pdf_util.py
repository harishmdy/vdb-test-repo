# *****************************************************************
#
# Copyright
#
# IBM Confidential
# IBM Zurich Research Laboratory - AIOPS Group
# All rights reserved.
#
# This software contains the valuable trade secrets of IBM or its
# * licensors.  The software is protected under international copyright
# * laws and treaties.  This software may only be used in accordance with
# * the terms of its accompanying license agreement.
#
# Authors :  Luc von Niederhäusern
#
# ******************************************************************

import re

import pdfplumber


def transform_camelot_to_fitz_coords(camelot_bbox, page_height):
    """
    Transforms the pdf coordinates provided by camelot to fitz (PyMuPDF2) format.

    Args:
    - camelot_bbox: PDF table bbox coordinates in camelot format (x0,y0,x1,y1).
    - page_height: Page height given by the fitz page object.

    Returns:
    - The transformed coordinates in fitz format.
    """
    fitz_bbox = (
        camelot_bbox[0],
        page_height - camelot_bbox[3],
        camelot_bbox[2],
        page_height - camelot_bbox[1],
    )

    return fitz_bbox


def extract_sentences_with_table_ref(docs: pdfplumber.pdf.PDF, page_num, table_caption):
    """
    Extracts sentences containing a specific table caption from pages i-1, i, and i+1.

    Args:
    - docs: A fitz.Document object representing the PDF.
    - page_num: The page number (0-indexed) where the table was found.
    - table_caption: The caption of the table to search for (e.g., "Table 1-1 bla bla").

    Returns:
    - A list of sentences containing the table caption from the specified pages.
    """
    sentences_with_ref = []
    # Regular expression to split text into sentences.
    match = re.search(r"Table \d+(-\d*)?", table_caption)
    table_id = match.group()
    # Pages to search: i-1, i, i+1
    pages_to_search = [page_num - 1, page_num, page_num + 1]

    for pg_num in pages_to_search:
        if 0 <= pg_num < len(docs.pages):
            page = docs.pages[pg_num]
            #            text = page.get_text("text")
            text = page.extract_text()
            # Split the page text into sentences
            sentences = re.split(r'\. |\n(?=[A-Z])', text)
            # Check each sentence for the table caption
            for sentence in sentences:
                if table_caption not in sentence and (
                    (table_id in sentence) or (table_id.replace(' ', '') in sentence)
                ):
                    short_form = table_id + '.'
                    sentence = sentence.rstrip()
                    word_count = len(sentence.split())
                    if word_count > 5 and sentence != short_form:
                        sentence = sentence.replace('\n', ' ')
                        sentences_with_ref.append(sentence)

    return sentences_with_ref
