
def print_tensor_shape(inputs):
    if isinstance(inputs,list) or isinstance(inputs,tuple):
        for input in inputs:
            print(input.shape)
    else:
        print(inputs)