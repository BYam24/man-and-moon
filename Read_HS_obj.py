import numpy as np


def reformat_obj(input_file, output_file):
    f_in = open(input_file, "r")
    f_out = open(output_file, 'w')
    for line in f_in:
        line_list = line.split()
        if line_list[0] == 'f':
            f_0 = line_list[1].split('/')
            f_1 = line_list[2].split('/')
            f_2 = line_list[3].split('/')
            out_line = 'f ' + f_0[0] + ' ' + f_1[0] + ' ' + f_2[0] + '\n'
            f_out.write(out_line)
        else:
            f_out.write(line)
    return
