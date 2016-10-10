#!/usr/bin/env python

"""
Script for graphically displaying run information contained in the Caffe log:
Will generate up to 4 plots (2-4 only available if debug mode enabled in Caffe):
1. Training\Test performance
2. Activations for all layers (taken from the Forward stage)
3. Parameter mean data value (taken from the Update stage)
4. Parameter mean data change (taken from the Update stage)

Plots are interactive, so you can zoom, pan, hide layers, etc.
"""

import os
import argparse
from parse_log import parse_log
from parse_log import getMaxParamCount
import display_results 
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    description = ('Parse a Caffe log file to create interactive graphs'
                   ' displaying training and test performance, and net '
                   'parameters and hyperparameters over time (if debug mode'
                   ' enabled)')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('logfile_path',
                        help='Path to log file')
                        # TODO Temporary bugfix, remove this
    parser.add_argument('--acc_layer', default='accuracy',
                        help='Name of accuracy layer to display test\\training'
                        ' performance for. If not supplied, default of '
                        '\'accuracy\' will be used' )
                        
    args = parser.parse_args()
    return args
""" 
Graph test\train accuracies. Currently doesn't support multiple accuracy
(such as top-1, top-5) since Caffe's parse_log.py doesn't support this    
"""
def get_test_accs_and_iters(test_dict_list, acc_layer_name):
    test_iters_list = [ item.get('NumIters') for item in test_dict_list]
    test_accuracy_name = ''
    num_test_outputs = 0
    # count number of different outputs (less pythonic code but faster)
    for i in range(len(test_iters_list)):
        if test_iters_list[i] == test_iters_list[0]:
            num_test_outputs += 1
        else:
            break
    # TODO fix code to support multiple loss layers. THis currently isn't in
    # use
#    for key in test_dict_list[0].keys():
#        if key not in ['NumIters', 'Seconds', 'LearningRate', train_loss_name]:
#            test_accuracy_name = key
    #######
            
    test_accs_lists = [np.array([ item.get(acc_layer_name) for item in test_dict_list[i::num_test_outputs]]) for i in range(num_test_outputs)]
    max_len_idx = [len(test_acc_list) for test_acc_list in test_accs_lists].index(max([len(test_acc_list) for test_acc_list in test_accs_lists]))
    max_iters_arr = np.array(test_iters_list[max_len_idx::num_test_outputs])
    return test_accs_lists , max_iters_arr

def get_train_loss_and_iters(train_dict_list, train_loss_name):
    train_iters_arr = np.array([ item.get('NumIters') for item in train_dict_list])
    train_loss_arr = np.array( [ item.get(train_loss_name) for item in train_dict_list])
    return train_iters_arr , train_loss_arr
 
    
def interactive_display_test_train(test_dict_list,train_dict_list,
                                   acc_layer_name):  
    test_accs_lists = []
    max_iters_list = []
    display_pairs_list = []
    train_display_pairs_list = []
    train_loss_name = ''
    test_output_num = 0

    xlabel = 'NumIters'
    title = 'Training and Test Performance'

    if train_dict_list:
        
        train_loss_name = train_dict_list[0].keys()[-1]
        train_iters_arr , train_loss_arr = get_train_loss_and_iters(train_dict_list, train_loss_name)
        train_loss_names = [key for key in train_dict_list[0].keys() if key not in ('NumIters', 'Seconds', 'LearningRate')]
        for loss_name in train_loss_names:
            train_iters_arr , train_loss_arr = get_train_loss_and_iters(train_dict_list, loss_name)
            train_display_pairs_list.append(( train_iters_arr[:len(train_loss_arr)],(train_loss_arr, loss_name)))

    if test_dict_list:
        test_accs_lists , max_iters_list = get_test_accs_and_iters(test_dict_list, acc_layer_name)
    for test_acc_list in test_accs_lists:
        display_pairs_list.append((max_iters_list[:len(test_acc_list)],
                                                  (test_acc_list, 'Test_#'+str(test_output_num)+' accuracy')))

        test_output_num +=1
   
    # Display Train\Test performance
    display_results.interactive_plot(display_pairs_list, xlabel, title, ylim = [0,1.05])
    # Dislay train losses
    display_results.interactive_plot(train_display_pairs_list, xlabel, title)


"""
Return list of all layer names net is composed of.
"""    
def get_layer_names(debug_info_dict_list):
    layer_names = []
    for row in debug_info_dict_list:
        if layer_names.count(row.get('LayerName')) > 0 :
            break
        else:
            layer_names.append(row.get('LayerName'))
    return layer_names


def main():
    args = parse_args() 
    train_dict_list, train_dict_names, test_dict_list, test_dict_names, debug_info_dict_list, debug_info_names = parse_log(os.path.realpath(args.logfile_path))
    interactive_display_test_train(test_dict_list,train_dict_list, args.acc_layer)
    
    if debug_info_dict_list: # only show the rest of the graphs if debug info exists
        with open(os.path.realpath(args.logfile_path)) as f:
            max_param_count = getMaxParamCount(f)
        # Get the layer names used in this net
        layer_name_list = get_layer_names(debug_info_dict_list)
        
        # Show activation, parameter data and backpropagated gradients per 
        # layer and per parameter. 
        # Activations
        layer_list = [(layer,'Activation') for layer in layer_name_list]
        display_results.interactive_plot_layers(layer_list,
                                                debug_info_dict_list,
                                                'Layer Mean Abs Activations')
        # Back-propagated gradients per layer
        layer_list = [(layer,'BackPropBottomDiff') for layer in layer_name_list]
        display_results.interactive_plot_layers(layer_list,
                                                debug_info_dict_list,
                                                'Back-propagated Gradients per Layer')
        # Layer parameter data values
        layer_list = [(layer,'param'+str(i)+'_Data') for layer in layer_name_list for i in range(max_param_count+1)]        
        display_results.interactive_plot_layers(layer_list,
                                                debug_info_dict_list,
                                                'Layer Mean Abs Data Values')
        # Gradients per layer
        layer_list = [(layer,'BackPropDiff_param'+str(i)) for layer in layer_name_list for i in range(max_param_count+1)]        
        display_results.interactive_plot_layers(layer_list,
                                                debug_info_dict_list,
                                                'Gradient per Parameter')
    plt.show()
    
if __name__ == '__main__':
    main()
    

    
