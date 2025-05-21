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

import glob
import io
import logging as logger
import os
import random
import re
import sys

import pandas as pd
import pdfplumber
import pdfplumber.page
from bs4 import BeautifulSoup, NavigableString

# sys.path.insert(0, "../src")
# from pdf_util import transform_camelot_to_fitz_coords, extract_sentences_with_table_ref
from ibm.unifiedsearchvectors.utils.html_table_extractor.html_util import (
    find_description,
    find_text_after_table,
    modify_links,
    read_pre_tables,
)
from ibm.unifiedsearchvectors.utils.html_table_extractor.pdf_util import (
    extract_sentences_with_table_ref,
)
from langchain.text_splitter import (
    SentenceTransformersTokenTextSplitter,
    SpacyTextSplitter,
)
from tqdm import tqdm

log = logger.getLogger("ibmsearch")


class TableExtractor:
    def __init__(self, config=None):
        """
        Initialize the TableExtractor.
        :param config: The config defined in config.yaml.
        """
        self.html_paths = config["paths"]["html_data"]
        self.pdf_paths = config["paths"]["pdf_data"]
        self.output_pdf = config["paths"]["output_folder"] + "pdf_tables/"
        self.output_html = config["paths"]["output_folder"] + "html_tables/"
        self.pdf_captions = config["pdf"]["captions"]
        self.pdf_metadata = config["pdf"]["metadata"]
        self.html_tags = config["html"]["tags"]
        self.html_metadata = config["html"]["metadata"]
        self.error_file = config["paths"]["error_file"]

    def extract_html_tables(self, html_content: str) -> list[dict]:
        '''
        Extracts all tables from text in the form of HTML

            Parameters:
                html_content (str): singular document text in HTML format
            Returns:
                tables_list (list[dict]): list of dicts holding table info
        '''

        # Initialize metadata dictionary
        meta_dict = {}

        # Parse the HTML
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract the title of the page
        try:
            page_title = soup.title.string.lstrip() if soup.title else ""
        except:
            # This might fail for corrupted or empty pages
            page_title = ""

        try:

            # meta_dict["Title"] = page_title

            # Remove leading whitespaces
            clean_text = re.sub(r'^\s+', '', soup.get_text(), flags=re.MULTILINE)
            clean_text = clean_text.replace(page_title, '')
            text_at_beginning = '\n'.join(clean_text.split('\n')[0:3])
            text_at_beginning = text_at_beginning.replace('\n', ' ')
            # Limit the "page description" to 300 characters
            text_at_beginning = text_at_beginning[:300]

            tables_list = []
            # Process each table
            for j, table in enumerate(soup.find_all(self.html_tags)):
                try:
                    description = ""
                    is_compatibility = False
                    # The pre tag is not unique to tables
                    if '<pre class="pre">' in str(table) and '----------' in str(table):
                        df = read_pre_tables(str(table))
                    elif '<pre' in str(table):
                        continue
                    else:
                        # Process images within the table
                        for img in table.find_all('img'):
                            if '0.1094.gif' in img['src']:
                                img.replace_with('Yes')
                                is_compatibility = True
                            elif '0.11DE.gif' in img['src']:
                                img.replace_with('No')
                                is_compatibility = True

                        # Find the description
                        description = find_description(table)
                        description = description.replace('\n', ' ')
                        if "(show details)" in description:
                            # These are sub-elements in tables not containing useful information
                            continue
                        else:
                            modify_links(table)

                        try:
                            # This can fail for corrupted or empty tables
                            df = pd.read_html(io.StringIO(str(table)))[0]
                        except Exception as e:
                            log.debug(f"Skipped table {j}, exception: {e}")
                            # print(table)
                            continue

                        # If there was no header extracted, use the first row as a header
                        if all(isinstance(col, int) for col in df.columns):
                            df = pd.read_html(io.StringIO(str(table)), header=0)[0]
                except Exception as e:
                    log.error(f"table try catch, exception: {e}")
                # Extract caption (if present) or use default text
                caption_tag = table.find('caption')
                caption = (
                    caption_tag.get_text()
                    if caption_tag and len(caption_tag.get_text()) < 500
                    else ""
                )
                caption = caption.replace('\n', ' ')

                # meta_dict["Page Context"] = text_at_beginning
                meta_dict["Description"] = description
                meta_dict["Caption"] = caption
                # meta_str = ""
                # for key, value in meta_dict.items():
                #     if key in self.html_metadata:
                #         meta_str += f"{key}: {value}\n"

                table_str = ""
                for _, row in df.iterrows():
                    row_str = ""
                    for col in df.columns:
                        row_str += f"{col}: {row[col]}, "
                    formatted_str = row_str[:-2]
                    formatted_str += "\n"
                    table_str += formatted_str
                if df.apply(lambda x: x.astype(str).str.contains('\*').any()).any():
                    # Find text after the table
                    additional_text = find_text_after_table(table)
                    table_str += additional_text
                meta_dict["Table"] = table_str
                if table_str.strip() != "":
                    tables_list.append(meta_dict.copy())

            return tables_list
        except Exception as e:
            log.error(f"Outer try catch, page title: {page_title}")
            log.error(e)
            # if table_str:
            #     self.write_table_files(meta_str, table_str, file_path, j)

    def get_page_tables_map(self, filename):
        all_tables = dict()

        with pdfplumber.open(filename) as pdf:
            i = 0
            for page in pdf.pages:
                i += 1
                # page = pdf.pages[824]
                try:
                    table_dfs = []
                    tables = page.extract_tables()
                    for table in tables:
                        print(i)
                        # print(table)
                        # break;
                        columns = table[0]  # [0:1]
                        print(columns)
                        table_data = table[1:]
                        print(table_data)
                        df = pd.DataFrame(table_data, columns=columns)
                        print(f'size: {df.size}')
                        if df.size > 2:
                            table_dfs.append(df)
                    # add the collection of dfs to the page dictionary
                    if len(table_dfs) > 0:
                        all_tables[i] = table_dfs
                    # findTables = page.find_tables()
                except Exception as e:
                    print(e)

        # size 1 or 2.
        all_tables

    def extract_pdf_tables(
        self,
        pdf_file_name,
    ):
        """
        Extracts tables from PDF files and saves them as TXT files.
        Note: This PDF functionality is computationally expensive, parallelize if possible.
        (You can parallelize by passing subsets of all pdf files as lists to this function)
        """
        # if not isinstance(self.pdf_paths, list):
        #    pdf_paths = glob.glob(self.pdf_paths + "**/*.pdf", recursive=True)
        # else:
        #    pdf_paths = self.pdf_paths

        # for path in tqdm(pdf_paths):
        tables_dict = (
            dict()
        )  # a dictionary of tables with the page as the key and the array of tables in the page as the object

        # all_tables = dict()
        all_tables = []

        with pdfplumber.open(pdf_file_name) as doc:
            i = 1
            for page in doc.pages:
                try:
                    tables = page.find_tables()
                    for table in tables:
                        table_extract = table.extract()
                        table_dict = dict()
                        columns = table_extract[0]  # [0:1]
                        table_data = table_extract[1:]
                        df = pd.DataFrame(table_data, columns=columns)
                        if (df.size > 2) and (len(df.columns.to_list()) > 1):
                            table_dict['page'] = i
                            table_dict['df'] = df
                            table_dict['bbox'] = table.bbox
                            all_tables.append(table_dict)

                except Exception as e:
                    log.exception(e)
                i += 1

            # doc = fitz.open(pdf_file_name)
            skip_tables = []  # Buffer for corrupted or continued tables
            for i, table in enumerate(all_tables):
                # Skip tables that we are not detecting with at least 95% accuracy
                if i not in skip_tables:
                    #                if table.parsing_report['accuracy'] > 95.0 and i not in skip_tables:
                    meta_dict = {}  # Dictionary to store the relevant metadata
                    metadata = doc.metadata
                    meta_dict["File Path"] = pdf_file_name
                    meta_dict["Title"] = metadata.get('title', '')
                    meta_dict["Keywords"] = metadata.get('keywords', '')
                    old_header_text = ""

                    look_for_cont = True
                    # First, look for a caption to check if its a valid table.
                    table_nonhuman_page_num = table['page'] - 1
                    page: pdfplumber.page.Page = doc.pages[table_nonhuman_page_num]
                    bbox = table['bbox']

                    try:
                        text = page.within_bbox(
                            (
                                bbox[0],
                                max(
                                    bbox[1] - 30, page.bbox[1]
                                ),  # make sure we don't go out of bounds of the page
                                bbox[2],
                                bbox[1] + 30,
                            )
                        ).extract_text()

                        for pattern in self.pdf_captions:
                            caption = re.search(pattern, text)
                            if "Table \d+\." in pattern:
                                # If format: Table 1. [...], then we don't need to look for continuations,
                                # as they are already have their own label
                                look_for_cont = False
                            if caption:
                                break

                        if not caption:
                            # If we can't find a caption, we skip this "table".
                            continue

                        # Create the pandas DataFrame.
                        df = table['df']

                        # If no header detected, we use the first row as the header
                        if all(isinstance(col, int) for col in df.columns):
                            df.columns = df.iloc[0]
                            df = df.drop(df.index[0])
                            df = df.reset_index(drop=True)

                        # If the caption is part of the df, of if header is mostly empty
                        # we remove it and re-assign the header

                        # make sure there are no None in the columns
                        df.columns = df.columns.to_series().fillna('')
                        num_empty_cols = float(sum(x == '' for x in df.columns[1:]))
                        if (num_empty_cols / len(df.columns[1:])) > 0.50:
                            old_header_text = " ".join(df.columns).strip()

                        if (len(old_header_text) > 1) and (
                            any(caption.group() in column for column in df.columns)
                        ):

                            old_columns = df.columns
                            df.columns = df.iloc[0]
                            df = df.drop(df.index[0])
                            df = df.reset_index(drop=True)
                            log.info(f"demoting header {old_columns} to {df.columns}")

                        # Check whether or not table continues on next page.
                        if (
                            bbox[1] < 110
                            and look_for_cont
                            and ((i + 1) < len(all_tables))
                        ):
                            # If the current table goes until the bottom of the page,
                            # check if it continues on the next page. To do this, we
                            # check that it has (1) no caption, (2) the same column headers.
                            next_table = all_tables[i + 1]
                            # (1) Check that there's no caption.
                            next_bbox = next_table['bbox']
                            next_page = doc.pages[table['page']]

                            next_text = next_page.within_bbox(
                                (
                                    next_bbox[0],
                                    max(
                                        next_bbox[1] - 30, next_page.bbox[1]
                                    ),  # make sure we don't go out of bounds of the page
                                    next_bbox[2],
                                    next_bbox[1],
                                )
                            ).extract_text()

                            # This is currently only needed for captions of format Table 1-1 [...] or Table 1 [...] (i.e. RedBooks)
                            next_caption = re.search(
                                r"Table \d+(-\d*)?\s+[A-Z][^\n]*", next_text
                            )
                            if not next_caption:
                                # (2) Check the potential continuation has the same column headers
                                next_df = next_table['df']
                                if all(isinstance(col, int) for col in next_df.columns):
                                    next_df.columns = next_df.iloc[0]
                                    next_df = next_df.drop(next_df.index[0])
                                    next_df = next_df.reset_index(drop=True)
                                try:
                                    # This might fail in rare cases
                                    if (next_df.columns == df.columns).all():
                                        df = pd.concat([df, next_df]).reset_index(
                                            drop=True
                                        )
                                        skip_tables.append(i + 1)
                                except:
                                    pass

                        # Handle compatibility tables
                        x_count = (df == 'X').sum().sum()
                        if x_count >= 2:
                            df.replace('X', "Yes", inplace=True)
                            df.replace('', "No", inplace=True)

                        # Get describing sentences
                        descr_sntcs = extract_sentences_with_table_ref(
                            docs=doc,
                            page_num=table_nonhuman_page_num,
                            # page_num=table.parsing_report['page'] - 1,
                            table_caption=caption.group(),
                        )

                        # Create the table content string
                        # start with the old_header text if there is one otherwise it's emptystring
                        table_str = old_header_text
                        for _, row in df.iterrows():
                            row_str = ""
                            for j, col in enumerate(df.columns):
                                row_str += f"{col}: {row.iloc[j]}, "
                            formatted_str = row_str[:-2]
                            formatted_str += "\n"
                            table_str += formatted_str

                        # Create the metadata string
                        meta_dict["Page"] = table['page']
                        meta_dict["Descriptive Sentences"] = '\n'.join(descr_sntcs)
                        meta_dict["Caption"] = caption.group()
                        meta_dict["Table Content"] = table_str  # added this
                        meta_dict["Table Number"] = i  # added this
                        meta_dict["bbox"] = (
                            bbox  # added this so later we can grab only text that is NOT in the table aread
                        )

                        # is there already a table for this page?  else create a new one
                        if meta_dict["Page"] in tables_dict:
                            current_table_list = tables_dict[meta_dict["Page"]]
                        else:
                            current_table_list = []
                        current_table_list.append(meta_dict)
                        tables_dict[meta_dict["Page"]] = current_table_list

                    except Exception as e:
                        log.exception(e)
                        log.error(f"error on page {page}")
                        # skip the table and keep going
                        continue

            return tables_dict
