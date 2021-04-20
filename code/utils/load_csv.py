''''
 * @Descripttion: csv to json code
 * @version: v1.0.0
 * @Author: yangbinchao
 * @Date: 2021-04-18 12:38:54
 '''

 #!/usr/bin/env python
# coding=utf-8

import json
import csv
import codecs


def read_csv_label(csv_input_name = '../../data/sample/sampleSubmission.csv'):

    #csv_input_name = '../../data/sample/sampleSubmission.csv'
    json_output_name = '../../data/sample/all_labels.json' 
    label_dict = {}
    label_json = []

    with open(csv_input_name, 'r') as myFile:
        lines=csv.reader(myFile)
        for line in lines:
            label_dict[line[0]] = line[1]
            
    return label_dict


if __name__ == "__main__":
    read_csv_label()