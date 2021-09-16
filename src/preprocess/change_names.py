#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""

import os
fnames = os.listdir('mzXML')
os.makedirs('positive_mzXML', exist_ok=True)
os.makedirs('negative_mzXML', exist_ok=True)

for filename in fnames:
    splits = filename.split('_')

    if splits[2] in ['pos', 'Pos']:
        splits[2] = 'Pos'
    elif splits[2] in ['neg', 'Neg']:
        splits[2] = 'Neg'

    first_split = "_".join([''.join([splits[0], splits[1]]), splits[2], splits[3]])
    new_name = "-".join([first_split, splits[4]])
    if splits[2] == "Pos":
        os.rename(src=f"mzXML/{filename}", dst=f"positive_mzXML/{new_name}")
    elif splits[2] == "Neg":
        os.rename(src=f"mzXML/{filename}", dst=f"negative_mzXML/{new_name}")
    else:
        print(splits)
        exit('Should be pos or neg...')
