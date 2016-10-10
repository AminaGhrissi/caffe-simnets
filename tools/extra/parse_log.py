#!/usr/bin/env python

"""
Parse training log with added functionality to parse debug information into 
a csv file. Only relevant if debug_info flag set to 'true'

Competitor to parse_log.sh
"""

import os
import re
import extract_seconds
import argparse
import csv
from collections import OrderedDict



FLOAT_RE = 'nan|[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?'
re_float_exp = re.compile('(^'+FLOAT_RE+'$)')


def get_display_interval(line_iterable):
    """
    Returns the display interval set for the run
    """
    re_display_interval = re.compile('display: (\d+)')
    for line in line_iterable:
        line = line.strip()
        display_interval_match = re_display_interval.search(line)
        if display_interval_match:
            return float(display_interval_match.group(1))
    return -1
    
def is_debug_mode(line_iterable):
    """
    return True if debug_info flag is on, False if not
    """
    re_debug_flag = re.compile('debug_info: (true|false)')
    for line in line_iterable:
        line = line.strip()
        debug_flag_match = re_debug_flag.search(line)
        if debug_flag_match:
            return (debug_flag_match.group(1) == 'true')
         

def getMaxParamCount(line_iterable):
    """
    Get the maximum number of parameters in use by any layer, to determine 
    the number of columns needed.
    """
    re_train_loss = re.compile('Iteration \d+, loss = ([\.\d]+)')
    re_forward_param = re.compile('\[Forward\] Layer (\S+), param blob (\d+) data: ('+FLOAT_RE+')')
    b_reached_forward_phase = False
    max_param_num = -1
    for line in line_iterable:
        line = line.strip()
        forward_param_match = re_forward_param.search(line)
        if forward_param_match:
            b_reached_forward_phase = True
            param_num = int(forward_param_match.group(2))
            if param_num > max_param_num:
                max_param_num = param_num
        re_iter_match = re_train_loss.search(line)
        if re_iter_match:
            if b_reached_forward_phase:
                break
        
    return max_param_num
    
    
def get_line_type(line):
    """Return either 'test' or 'train' depending on line type
    """

    line_type = None
    if line.find('Train') != -1:
        line_type = 'train'
    elif line.find('Test') != -1:
        line_type = 'test'
    return line_type



def extended_float(num_str):
    """
    casting strings representing floating point numbers of the types:
    1. Xe-0n where X is a decimal representation and n is an integer
    2.Regular decimal numbers (i.e 8.101)
    3. nan
    """
    match_num = re_float_exp.search(num_str)
    if not match_num:
        return 'N\A'  # parsing error
    if num_str == 'nan':
        return float('NaN')
    return float(num_str)


def parse_log(path_to_log):
    """Parse log file
    Returns (train_dict_list, train_dict_names, test_dict_list, 
    test_dict_names, debug_info_names, debug_info_dict_list)
    
    If the debug info wasn't enabled for the run, debug_info_dict_list is
    empty

    train_dict_list, test_dict_list and debug_info_dict_list are lists of 
    dicts that define the table rows

    train_dict_names, test_dict_names and  debug_info_names are ordered 
    tuples of the column names for the two dict_lists
       
    """

    re_iteration = re.compile('Iteration (\d+)')

    re_train_loss = re.compile('Iteration (\d+), loss = ('+FLOAT_RE+')')
    
    regex_train_output = re.compile('Train net output #(\d+): (\S+) = ([\.\deE+-]+)')
    regex_test_output = re.compile('Test net output #(\d+): (\S+) = ([\.\deE+-]+)')
    regex_learning_rate = re.compile('lr = ([-+]?[0-9]*\.?[0-9]+([eE]?[-+]?[0-9]+)?)')
    # For parsing debug info
    re_forward_data = re.compile('\[Forward\] Layer (\S+), top blob (\S+) data: ('+FLOAT_RE+')')
    re_backward_diff = re.compile('\[Backward\] Layer (\S+), bottom blob (\S+) diff: ('+FLOAT_RE+')')
    re_backward_param_diff = re.compile('\[Backward\] Layer (\S+), param blob (\d+) diff: ('+FLOAT_RE+')')
    re_forward_param_data = re.compile('\[Forward\] Layer (\S+), param blob (\d+) data: ('+FLOAT_RE+')')
    was_in_backward = False

    # Pick out lines of interest
    iteration = -1
    fb_iteration = -1 # iter # used for timing forward\backward
    debug_flag = False
    max_param_count = -1
    learning_rate = float('NaN')
    train_dict_list = []
    test_dict_list = []
    debug_info_dict_list = []
    debug_layer_dict = {}
    train_row = None
    test_row = None
    
    train_dict_names = ('NumIters', 'Seconds', 'TrainingLoss', 'LearningRate')
    test_dict_names = ('NumIters', 'Seconds', 'TestAccuracy', 'TestLoss')
    debug_info_dict_list = []
    debug_info_names_list = ['NumIters', 'LayerName', 'Activation', 'BackPropBottomDiff']
    debug_info_names = tuple(debug_info_names_list)
    

    logfile_year = extract_seconds.get_log_created_year(path_to_log)
    with open(path_to_log) as f:
        display_interval = get_display_interval(f)        
        debug_flag = is_debug_mode(f)
        if debug_flag:
            max_param_count = getMaxParamCount(f)
            additional_header_list = []
            backward_param_headers = ['BackPropDiff'+'_param'+str(i) for i in range(max_param_count+1)]
            additional_header_list += backward_param_headers
            for i in range(max_param_count+1):
                additional_header_list.append('param'+str(i)+'_Data')
                additional_header_list.append('param'+str(i)+'_Change')

            # adding new headers for each of the parameters            
            debug_info_names_list += additional_header_list  
            debug_info_names = tuple(debug_info_names_list)             
            f.seek(0) # return to head of file
            
        start_time = extract_seconds.get_start_time(f, logfile_year)

        for line in f:
            iteration_match = re_iteration.search(line)
            if iteration_match:
                iteration = float(iteration_match.group(1))

            
            if iteration == -1:
                # Only look for other stuff if we've found the first iteration
                continue
            # Try to extract date and time from line, assuming there exists one in
            # the expected format
            try:
            	time = extract_seconds.extract_datetime_from_line(line,
                                                              logfile_year)
            except:
            	continue
            seconds = (time - start_time).total_seconds()
            train_loss_match = re_train_loss.search(line)
            if train_loss_match:
                fb_iteration = float(train_loss_match.group(1))
            learning_rate_match = regex_learning_rate.search(line)
            if learning_rate_match:
                learning_rate = float(learning_rate_match.group(1))


            train_dict_list, train_row = parse_line_for_net_output(
                regex_train_output, train_row, train_dict_list,
                line, iteration, seconds, learning_rate
            )
            test_dict_list, test_row = parse_line_for_net_output(
                regex_test_output, test_row, test_dict_list,
                line, iteration, seconds, learning_rate
            )
            
            fix_initial_nan_learning_rate(train_dict_list)
            fix_initial_nan_learning_rate(test_dict_list)

            # Only extract debug information if debug_info is true
            if not debug_flag:
                continue

            forward_match = re_forward_data.search(line)
            if forward_match:
                # If was_in_update flag was on, we are starting a new forward
                # pass so we will save last iteration info and 
                # initialize the iteration specific variables
                if was_in_backward:                    
                    debug_info_dict_list += debug_layer_dict.values()
                    debug_layer_dict = {}
                    was_in_backward = False                
                layer_name = forward_match.group(1)
                activation_val = extended_float(forward_match.group(3))
                if not debug_layer_dict.has_key(layer_name):
                    debug_layer_dict[layer_name] = dict.fromkeys(debug_info_names)
                    debug_layer_dict[layer_name]['LayerName'] = layer_name
                    debug_layer_dict[layer_name]['NumIters'] = \
                    (fb_iteration != -1) * (fb_iteration + display_interval)
                debug_layer_dict[layer_name]['Activation'] = activation_val

            forward_param_data_match = re_forward_param_data.search(line)
            if forward_param_data_match:
                
                layer_name = forward_param_data_match.group(1)
                param_num = forward_param_data_match.group(2)
                param_header = 'param'+param_num
                param_data = extended_float(forward_param_data_match.group(3))
                debug_layer_dict[layer_name][param_header+'_Data'] = param_data
                
            backward_match = re_backward_diff.search(line)
            if backward_match:
                layer_name = backward_match.group(1)
                back_prop_val = extended_float(backward_match.group(3))
                if not debug_layer_dict.has_key(layer_name):
                    debug_layer_dict[layer_name] = dict.fromkeys(debug_info_names)
                debug_layer_dict[layer_name]['BackPropBottomDiff'] = back_prop_val
            
            backward_param_match = re_backward_param_diff.search(line)
            if backward_param_match:
                was_in_backward = True
                layer_name = backward_param_match.group(1)
                param_num = backward_param_match.group(2)
                param_header = '_param'+param_num
                back_prop_param_val = extended_float(backward_param_match.group(3))
                if not debug_layer_dict.has_key(layer_name):
                    debug_layer_dict[layer_name] = dict.fromkeys(debug_info_names)
                debug_layer_dict[layer_name]['BackPropDiff'+param_header] = back_prop_param_val
                                

    
    # add last iteration information if it exists
    if debug_flag and debug_layer_dict:
        debug_info_dict_list += debug_layer_dict.values()

    return train_dict_list, train_dict_names, test_dict_list, test_dict_names, \
    debug_info_dict_list, debug_info_names

def parse_line_for_net_output(regex_obj, row, row_dict_list,
                              line, iteration, seconds, learning_rate):
    """Parse a single line for training or test output

    Returns a a tuple with (row_dict_list, row)
    row: may be either a new row or an augmented version of the current row
    row_dict_list: may be either the current row_dict_list or an augmented
    version of the current row_dict_list
    """

    output_match = regex_obj.search(line)
    if output_match:
        if not row or row['NumIters'] != iteration:
            # Push the last row and start a new one
            if row:
                # If we're on a new iteration, push the last row
                # This will probably only happen for the first row; otherwise
                # the full row checking logic below will push and clear full
                # rows
                row_dict_list.append(row)

            row = OrderedDict([
                ('NumIters', iteration),
                ('Seconds', seconds),
                ('LearningRate', learning_rate)
            ])

        # output_num is not used; may be used in the future
        # output_num = output_match.group(1)
        output_name = output_match.group(2)
        output_val = output_match.group(3)
        # Added to handle issue of not saving all net test outputs in iteration
        # 0
        if row.has_key(output_name):
            row_dict_list.append(row)
            row = OrderedDict([
                ('NumIters', iteration),
                ('Seconds', seconds),
                ('LearningRate', learning_rate)
            ])

        row[output_name] = float(output_val)
        
            

    if row and len(row_dict_list) >= 1 and len(row) == len(row_dict_list[0]):
        # The row is full, based on the fact that it has the same number of
        # columns as the first row; append it to the list
        row_dict_list.append(row)
        row = None

    return row_dict_list, row


def fix_initial_nan_learning_rate(dict_list):
    """Correct initial value of learning rate

    Learning rate is normally not printed until after the initial test and
    training step, which means the initial testing and training rows have
    LearningRate = NaN. Fix this by copying over the LearningRate from the
    second row, if it exists.
    """
    # TODO after making the change to solve bug of not saving all iteration 0
    # test outputs, the learning rate may not necessarily be in the index 1 
    # of the dict_list
    if len(dict_list) > 1:
        dict_list[0]['LearningRate'] = dict_list[1]['LearningRate']

def save_csv_files(logfile_path, output_dir, train_dict_list,
                   test_dict_list, debug_info_dict_list, 
                   delimiter=',', verbose=False):
    """Save CSV files to output_dir

    If the input log file is, e.g., caffe.INFO, the names will be
    caffe.INFO.train and caffe.INFO.test
    If saving debug information to csv, the resulting filename will be
    caffe.INFO_debug.csv
    """
    """Save CSV files to output_dir

    If the input log file is, e.g., caffe.INFO, the names will be
    caffe.INFO.train and caffe.INFO.test
    """

    log_basename = os.path.basename(logfile_path)
    train_filename = os.path.join(output_dir, log_basename + '.train')
    write_csv(train_filename, train_dict_list, delimiter, verbose)

    test_filename = os.path.join(output_dir, log_basename + '.test')
    write_csv(test_filename, test_dict_list, delimiter, verbose)
    
    if debug_info_dict_list:  
        print "Writing debug info csv file..."
        debug_filename = os.path.join(output_dir, log_basename + '_debug.csv')

        write_csv(debug_filename, debug_info_dict_list, delimiter,
                  verbose)

def write_csv(output_filename, dict_list, delimiter, verbose=False):
    """Write a CSV file
    """

    if not dict_list:
        if verbose:
            print('Not writing %s; no lines to write' % output_filename)
        return

    dialect = csv.excel
    dialect.delimiter = delimiter

    with open(output_filename, 'w') as f:
        dict_writer = csv.DictWriter(f, fieldnames=dict_list[0].keys(),
                                     dialect=dialect)
        dict_writer.writeheader()
        dict_writer.writerows(dict_list)
    if verbose:
        print 'Wrote %s' % output_filename


def parse_args():
    description = ('Parse a Caffe training log into two CSV files '
                   'containing training and testing information')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('logfile_path',
                        help='Path to log file')

    parser.add_argument('output_dir',
                        help='Directory in which to place output CSV files')

    parser.add_argument('--verbose',
                        action='store_true',
                        help='Print some extra info (e.g., output filenames)')

    parser.add_argument('--delimiter',
                        default=',',
                        help=('Column delimiter in output files '
                              '(default: \'%(default)s\')'))

    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    train_dict_list, train_dict_names, test_dict_list, test_dict_names, debug_info_dict_list, debug_info_names = parse_log(args.logfile_path)
    
    save_csv_files(args.logfile_path, args.output_dir, train_dict_list,
                   test_dict_list, debug_info_dict_list, delimiter=args.delimiter)
   
  
    print "Done!"

if __name__ == '__main__':
    main()
