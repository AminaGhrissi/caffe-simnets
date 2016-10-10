#!/usr/bin/env python

"""
Script for graphically displaying results of debug information csv file.

"""


import argparse
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import CheckButtons

def parse_args():
    description = ('Display the debug information csv file according to selected headers')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('csvfile_path',
                        help='Path to csv file created by parse_log.py')

    parser.add_argument('layer_header_pairs',
                        help='Comma-separated pairs of the form'
                        ' #layer_name,column_to_plot# where column_to_plot is'
                        ' the header of the desired column you wish to display')


    args = parser.parse_args()
    return args

"""
Gets a list of pairs of the form (layer name,header of column to plot),
 numpy arrays of the data loaded from a csv (one copy as strings and one copy of 
 floats) and creates a plot of the desired information as a function iterations.
"""
def plot_layers_info(layer_info_type_list,data,data_float):
    headers_list = data[0].tolist()
    iters = np.unique(data_float[1:,headers_list.index('NumIters')])
    handles = []    
    plt.xlabel('NumIters')
    for (layer_name,column_to_plot) in layer_info_type_list:
        layer_rows = data_float[data[:,headers_list.index('LayerName')] == layer_name]
        y = layer_rows[:,headers_list.index(column_to_plot)]
        plt.plot(iters[:len(y)],y )
        handles.append(layer_name+'.'+column_to_plot)
            
    plt.legend(handles)    
    plt.show()
    
def plot_layers_info_from_dict_list(layer_info_type_list,debug_info_dict_list):
    iters_list = np.unique(np.array([ item.get('NumIters') for item in debug_info_dict_list]))
    handles = []    
    plt.xlabel('NumIters')
    for (layer_name,column_to_plot) in layer_info_type_list:
        row_list = [item.get(column_to_plot)  for item in debug_info_dict_list if layer_name==item.get('LayerName')]
        if row_list.count(None) > 0:
            continue
        y = np.array(row_list)
        plt.plot(iters_list[:len(y)],y )
        handles.append(layer_name+'.'+column_to_plot)
    
    plt.legend(handles)   


def interactive_plot( plot_pairs_list, x_label, plot_title = "untitled", ylim=[]):
    
    fig, ax = plt.subplots()
    
    #iters_list = np.unique(np.array([ item.get('NumIters') for item in debug_info_dict_list]))
    handles = []  
    lines = []
    plt.xlabel(x_label)
    for x,y_pair in plot_pairs_list:
        y, y_label = y_pair

        if len(y) == 0:
            continue
        line, = ax.plot(x[:len(y)], y , lw=2)
        handles.append(y_label)
        lines.append(line)
        if ylim:
            ax.set_ylim(ylim)

    plt.title(plot_title)    
    plt.subplots_adjust(left=0.4)
    rax = plt.axes([0.05, 0.4, 0.3, 0.5])
    check = CheckButtons(rax, tuple(handles), tuple([True for i in range(len(handles))]))
    
    [rec.set_facecolor(lines[i].get_color()) for i, rec in enumerate(check.rectangles)]
    def func(label):
        
        for i in range(len(handles)):
            if label == handles[i]:
                lines[i].set_visible(not lines[i].get_visible())
                
        plt.draw()
    check.on_clicked(func)
    rax._cbutton = check
   
def interactive_plot_layers(layer_info_type_list,debug_info_dict_list,
                            plot_title = "untitled"):
    
    fig, ax = plt.subplots()
    
    iters_list = np.unique(np.array([ item.get('NumIters') for item in debug_info_dict_list]))
    handles = []  
    lines = []
    plt.xlabel('NumIters')
    for (layer_name,column_to_plot) in layer_info_type_list:
        row_list = [item.get(column_to_plot)  for item in debug_info_dict_list if (layer_name==item.get('LayerName') and item.get(column_to_plot) != None and not np.isnan(item.get(column_to_plot)))]
        
        if len(row_list) == 0:
            continue
        y = np.array(row_list)
        line, = ax.plot(iters_list[:len(y)], y , lw=2)
        handles.append(layer_name+'.'+column_to_plot)
        lines.append(line)

    plt.title(plot_title)    
    plt.subplots_adjust(left=0.4)
    rax = plt.axes([0.05, 0.4, 0.3, 0.5])
    check = CheckButtons(rax, tuple(handles), tuple([True for i in range(len(handles))]))
    
    [rec.set_facecolor(lines[i].get_color()) for i, rec in enumerate(check.rectangles)]
    def func(label):
        
        for i in range(len(handles)):
            if label == handles[i]:
                lines[i].set_visible(not lines[i].get_visible())
                
        plt.draw()
    check.on_clicked(func)
    rax._cbutton = check

def main():
    args = parse_args()
    matches_list = re.findall('(?:#(\S+?),(\S+?)#)+',args.layer_header_pairs)
    
    
    data = np.genfromtxt(args.csvfile_path , delimiter=',' , dtype = None)
    data_float = np.genfromtxt(args.csvfile_path , delimiter=',')
    
    plot_layers_info(matches_list,data,data_float)

    
    


if __name__ == '__main__':
    main()
    
        
        
        
        
