# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 08:37:43 2021

@author: Aju
"""
#lower case
import numpy as np
def to_lower(word_arr):
    word_arr=np.append([],word_arr)
    word_arr=[word.lower() for word in word_arr]
    return word_arr


"""
w="ArrT"
print(to_lower(w))
"""
    