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
from bs4 import NavigableString
import pandas as pd

def find_text_after_table(table):
    """
    Extracts text after the table. Only used to find footnotes.
    Args:
    - table: The BeautifulSoup table object.
    Returns:
    - A string containing footnotes.
    """
    next_sibling = table.find_next_sibling()
    if next_sibling and isinstance(next_sibling, NavigableString):
        return next_sibling.strip()
    elif next_sibling and next_sibling.get_text(strip=True):
        return next_sibling.get_text(strip=True)
    return ""

def find_description(table):
    """
    Finds a description, i.e. text immediately above the table.
    Args:
    - element: The BeautifulSoup table object.
    Returns:
    - A string (possibly) containing a description of the table.
    """
    # Check the immediate preceding sibling
    prev_sibling = table.find_previous_sibling()
    if prev_sibling:
        # Extract text from the previous sibling (e.g. from <b> tag inside <p>)
        text = ' '.join(prev_sibling.stripped_strings)
        if text:
            return text
    return ""

def modify_links(table):
    """
    Transforms clickable links to format "description: link" (e.g. "IBM WatsonX: https://www.ibm.com/watsonx").
    This happens in place.
    Args:
    - table: The BeautifulSoup table object.
    Returns:
      None
    """
    links = table.find_all('a')
    for link in links:
        # Replace the link text with its href attribute
        descriptive_text = link.text  
        href = link.get('href')
        link.string = f"{descriptive_text}: {href}"

def process_row(line, col_ids):
    """
    Transforms clickable links to format "description: link" (e.g. "IBM WatsonX: https://www.ibm.com/watsonx").
    This happens in place.
    Args:
    - table: The BeautifulSoup table object.
    Returns:
      None
    """
    # Fct to decide whether a given line starts a new row or continues the last one
    parts = [line[i:j].strip() for i,j in zip(col_ids, [*col_ids[1:], None])]

    return parts, sum(bool(s) for s in parts) >= 2, sum(bool(s) for s in parts) == 0

def read_pre_tables(text):
    """
    Reads tables specifies by the <pre class="pre"> tag, i.e. pure string tables.
    This is currently specific to tables as displayed on https://www.ibm.com/docs/en/announcements/system-storage-ts3500-tape-library-model-d23.
    
    Use this as a template to extend the functionality to other delimiters, etc..
    Args:
    - text: The string making up the table.
    Returns:
    - df: A Pandas DataFrame representation of the input table.
    """
    # Split the text into lines
    lines = text.strip().split('\n')
    lines = [re.sub(r'<.*?pre.*?>', '', l) for l in lines]
    split_pattern = re.compile(r'\s*-+\s*')
    matching_indices = [i for i, line in enumerate(lines) if split_pattern.match(line)]
    split_id = matching_indices[0]

    # Identify the header (assuming the first line is headers)
    header_rows = lines[:split_id]

    # Infer the number of columns from the dashed columns
    num_columns = len(lines[split_id].split())
    
    # Split the header cols and create the header row
    # 1. Find the indices marking the start of columns
    col_ids = [match.start() for match in re.finditer(r"-{2,}", lines[split_id])]
    headers = [[] for _ in range(num_columns)]
    for r in header_rows:
        for i in range(num_columns):
            if i == num_columns-1:
                headers[i].append(r[col_ids[i]:].strip())
            else:
                headers[i].append(r[col_ids[i]:col_ids[i+1]].strip())
    headers = [' '.join(sublist) for sublist in headers]
    headers = [header.strip() for header in headers]

    data_rows = lines[(split_id+1):]  # Skipping the line of dashes
    # Prepare data for DataFrame
    data = []
    buffer_rows = []  # Temporary storage for incomplete lines
    for line in data_rows:
        parts, is_complete_line, is_empty_line = process_row(line, col_ids)
        if is_empty_line:
            continue  # Skip empty lines

        if is_complete_line:
            if buffer_rows:
                # TODO: Potentially add decision here whether to back- or fwdtrack, for now we backtrack only.
                # If there's buffered rows, backtrack to include those
                buffer_rows.append(parts)
                full_row = [' '.join(sublist) for sublist in zip(*buffer_rows)]
                full_row = [p.strip() for p in full_row]
                data.append(full_row)
                buffer_rows = []
            else:
                # Directly add complete lines to structured data
                parts = [p.strip() for p in parts]
                data.append(parts)
        else:
            # Buffer incomplete lines
            buffer_rows.append(parts)

    # Create DataFrame
    df = pd.DataFrame(data, columns=headers)

    return df