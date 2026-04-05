#!/bin/python3

class Template:
    def __init__(self):
        return

    def get(self, \
        tensor_parallel_size: int, \
        num_gpu: int, \
        gpu_product: str, \
        completions: int, \
        line_array: str, \
        filename: str):

        gpu=gpu_product.split("-")[1] 
        
        return f"""
              """
