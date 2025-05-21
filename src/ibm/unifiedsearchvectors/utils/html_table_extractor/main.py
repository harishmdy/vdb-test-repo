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
#%%
import os
import subprocess
import sys

sys.path.insert(0, "../src")
def install_requirements():
    subprocess.run(["pip", "install", "-r","../src/ibm/unifiedsearchvectors/utils/html_table_extractor/requirements.txt"])
install_requirements()

print(os.getcwd())
import yaml
from utils.html_table_extractor.TableExtractor import TableExtractor


from ibm.unifiedsearchvectors.utils.html_table_extractor.TableExtractor import (
    TableExtractor,
)


def main():
    with open('../src/ibm/unifiedsearchvectors/utils/html_table_extractor/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    extractor = TableExtractor(config)
    # extractor.ensure_output_dir_exists()
    f = open("utils/html_table_extractor/example.txt", "r")
    tables_list = extractor.extract_html_tables(f.read())
    print(len(tables_list))
    print(tables_list[1])
    print(tables_list[2])

    # extractor.extract_pdf_tables()

# if __name__ == "__main__":
#     main()
main()
#%%