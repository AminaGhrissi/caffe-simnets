#!/usr/bin/env python
# Inspired by Xavier Bouthillier's YAML generator
from __future__ import unicode_literals
import argparse, json, os, os.path, itertools
import sys, tempfile,  subprocess, random, glob
import shutil, filecmp, re, math, csv, uuid, datetime, signal
import numpy as np
import permutes
import curtsies
from curtsies import CursorAwareWindow, Input, events, fmtstr, FSArray, fsarray
from curtsies.fmtfuncs import *
import platform

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
LOCAL_CAFFE_EXEC = os.path.abspath(os.path.join(SCRIPT_DIR, '../../build/tools/caffe'))

def error(error_message):
    print error_message
    print """Try `python hyper_train.py --help` for more information"""
    sys.exit(2)

def subprocess_lines_iterator(p, output='stdout'):
    while True:
        if output == 'stdout':
            line = p.stdout.readline()
        elif output == 'stderr':
            line = p.stderr.readline()
        else:
            raise Error('Bad output value: %s' % str(output))

        c = p.poll()
        if line == '':
            if c != None:
                break
            else:
                continue
        yield line

def iterate_and_write(it, f):
    while True:
        item = it.next()
        f.write(item)
        try:
            f.flush()
        except IOError as e:
            sys.stderr.write('\nIO Error while flushing the log file: {0} (Error number {1})\n'.format(e.errno, e.strerror))
        yield item

def reprint_line(s):
    sys.stdout.write("\r\x1b[K" + s)
def ensure_precision(value, precision):
    if type(value) is bool:
        return str(value)
    try:
        num = float(value)
        if num.is_integer():
            if num < 1000:
                return str(int(num))
            if num < 1000000:
                return ('%.1f' % (num/1000)) + 'k'
            return ('%.1f' % (num/1000000)) + 'm'
        return ('%%.%de' % precision) % num
    except (ValueError, TypeError):
        return str(value)
def flatten(data):
    newd = dict()
    if type(data) is dict:
        for key, value in data.iteritems():
            f_v = flatten(value)
            for k, v in f_v.iteritems():
                newd[key + k] = v
    elif type(data) is list:
        for i, value in enumerate(data):
            f_v = flatten(value)
            for k, v in f_v.iteritems():
                newd['%d_%s' % (i, k)] = v
    elif type(data) is tuple and len(data) == 2 and isinstance(data[0], (str, unicode)):
        f_v = flatten(data[1])
        for k, v in f_v.iteritems():
            newd[data[0] + k] = v
    else:
        newd[''] = str(data)
    return newd

def semi_unique():
    return str(uuid.uuid1().int>>96)

# A single hyper parameter which parse and validate the parameter dict, and supports sampling.
class HyperParameter():
    def __init__(self, param, search_mode):
        try:
            self._name = param['name']
            self._type = param['type']
            if 'data_type' in param:
                self._data_type = param['data_type']
            else:
                self._data_type = None
            if self._type == 'preset':
                self._values = param['values']
                self._sample_count = len(self._values)
            elif self._type == 'sample':
                self._sample_type = param['sample_type']
                if self._sample_type == 'choice':
                    self._values = param['values']
                    if 'sample_count' in param:
                        self._sample_count = param['sample_count']
                    else:
                        self._sample_count = len(self._values)
                else:
                    if 'sample_count' in param:
                        self._sample_count = int(param['sample_count'])
                    elif search_mode == 'random':
                        self._sample_count = 1
                    else:
                        error('Invalid hyper parameter %s: missing sample count.' % self.name)

                    self._min = float(param['min'])
                    self._max = float(param['max'])
                    if (self._min > self._max):
                        error('Invalid hyper parameter %s: min > max.' % self.name)
                                  
            elif self._type == 'radius_permute':
                self._sample_count = 1
                self._height = int(param['height'])
                self._width = int(param['width'])
                self._max_radius = int(param['max_R'])
                self._values = [permutes.radius_permute( (self._height, self._width), self._max_radius)]
            
            elif self._type == 'cyclic_shift_permute':
                self._sample_count = 1
                self._height = int(param['height'])
                self._width = int(param['width'])
                self._s_x = int(param['s_x'])
                self._s_y = int(param['s_y'])   
                self._values = [permutes.cyclic_shift_permute( (self._height, self._width), self._s_x, self._s_y)]
                
            else:
                error('Invalid hyper parameter %s: unknown type %s.' % (self.name, self._type))
        except ValueError:
            json_str = json.dumps(param, separators=(',',':'))
            error('Invalid hyperparameter\n%s' % (json_str))

    def name(self):
        return self._name

    def values_iter(self):
        it = None
        if (self._type == 'preset' or self._type == 'radius_permute' 
            or self._type == 'cyclic_shift_permute'):
            it = self._preset_iter()
        elif self._type == 'sample':
            if self._sample_type == 'linspace':
                it = self._linspace_iter()
            elif self._sample_type == 'logspace':
                it = self._logspace_iter()
            elif self._sample_type == 'uniform':
                it = self._uniform_iter()
            elif self._sample_type == 'log_uniform':
                it = self._log_uniform_iter()
            elif self._sample_type == 'choice':
                it = self._choice_iter()
            else:
                error('Invalid hyper parameter %s: unknown sample type %s.' % (self.name, self._sample_type))
            
        return itertools.imap(lambda x: self._cast_type(x), it)

    def values(self):
        it = self.values_iter()
        return [it.next() for i in xrange(self._sample_count)]

    def _cast_type(self, x):
        if self._data_type is None:
            return x
        elif self._data_type == 'int':
            return int(x)
        elif self._data_type == 'float':
            return float(x)
        elif self._data_type == 'string':
            return str(x)

    def _preset_iter(self):
        return itertools.cycle(self._values)
    def _linspace_iter(self):
        return itertools.cycle(np.linspace(self._min, self._max, self._sample_count))
    def _logspace_iter(self):
        if np.sign(self._min) != np.sign(self._max):
            error('log uniform requires min and max to have the same sign')
        exp_min, exp_max =  sorted([np.log10(np.abs(self._min)), np.log10(np.abs(self._max))])
        return itertools.cycle(np.sign(self._min) * np.logspace(exp_min, exp_max, self._sample_count))
    def _uniform_iter(self):
        while True:
            yield np.random.uniform(self._min, self._max)
    def _log_uniform_iter(self):
        if np.sign(self._min) != np.sign(self._max):
            error('log uniform requires min and max to have the same sign')
        sign = np.sign(self._min)
        exp_min, exp_max =  sorted([np.log10(np.abs(self._min)), np.log10(np.abs(self._max))])
        while True:
            u = np.random.uniform(exp_min, exp_max)
            yield sign * (10 ** u)
    def _choice_iter(self):
        while True:
            yield random.choice(self._values)

class HyperParameters():
    def __init__(self, params_list, search_type):
        self.search_type = search_type
        if not isinstance(params_list[0], list):
            params_list = [params_list]
        if not isinstance(search_type, list):
            search_type = [search_type] * len(params_list)
        for t in search_type:
            if t not in ['grid', 'random']:
                error('Invalid search type!')
        self.search_type = search_type
        self.params = [[HyperParameter(p, self.search_type[i]) for p in ps] for i, ps in enumerate(params_list)]
    def _search_iter(self, params, search_type):
        if search_type == 'grid':
            return self._grid_search_iter(params)
        elif search_type == 'random':
            return self._random_search_iter(params)
    def _grid_search_iter(self, params):
        return itertools.product(*[itertools.product([p.name()], p.values()) for p in params])
    def _random_search_iter(self, params):
        return itertools.izip(*[itertools.izip(itertools.repeat(p.name()), p.values_iter()) for p in params])
    def search_iter(self):
        iters = [enumerate(self._search_iter(p, self.search_type[i]), 1) for i, p in enumerate(self.params)]
        def combine_enumeration(x):
            res = zip(*x)
            res[1] = list(itertools.chain.from_iterable(res[1]))
            return res
        if len(iters) > 1:
            return itertools.imap(combine_enumeration, itertools.product(*iters))
        else:
            return  iters[0]
# Helper class to calculate moving average of accuracies
class MovingAverageWindow():
    def __init__(self, n):
        self._size = n
        self._window = []
    def insert(self, value):
        if len(self._window) >= self._size :
            del(self._window[0])
        self._window.append(value)
    def get_mean(self):
        return np.mean(self._window)

class TrainPlan():
    SUCCESS, SKIPPED, CAFFE_TERMINATED, EXISTS, RETRY, USER_ASK_EXIT = range(6)
    SIGINT, NEWLINE = range(2)
    def __init__(self,file_name):
        self._train_plan_filename = file_name
        with open(file_name, 'r') as f:
            def filter_comments(line):
                in_string = False
                string_char = None
                for i in range(len(line)):
                    if in_string:
                        if line[i] == string_char:
                            in_string = False
                    else:
                        if line[i] == '#':
                            return line[:i]
                        if line[i] in ['"', "'"]:
                            string_char = line[i]
                            in_string = True
                return line
            filtered_file = '\n'.join([filter_comments(line) for line in f])
            data = json.loads(filtered_file)
            try:
                self._user_ask_exit = False
                self._name = data['name']
                self._params = HyperParameters(data['hyper_params'], data['search_type'])
                try:
                    self._advanced_rendering_engine = data['rendering_engine']
                except KeyError:
                    self._advanced_rendering_engine = False

                if self._advanced_rendering_engine:
                    from jinja2 import Environment
                    self._templateGenerator = Environment()
                    self._templateGenerator.filters['bool'] = lambda x: str(x).lower()

                try:
                    self._max_data_points = data['max_data_points']
                except KeyError:
                    if data['search_type'] == 'random':
                        error('Invalid train plan! When using random search you must set max_data_points.')
                    self._max_data_points = None
                
                try:
                    self._weights_file = data['weights_file']
                except KeyError:
                    self._weights_file = None
                
                self._mva_window_size = 10 #default size
                try:
                    if data['moving_avg_window_size'] > 0:
                        self._mva_window_size = int(data['moving_avg_window_size'])
                    else:
                        error('moving_avg_window_size must be positive integer. Using default value of '
                            '%d' %  self._mva_window_size)
                except KeyError:
                    pass # Use the default value

                self._termination_rules = dict()
                self._termination_rules['nan'] = True
                self._termination_rules['delta'] = -float('inf')
                try:
                    t = data['termination_rules']
                    for key, value in t.iteritems():
                        self._termination_rules[key] = value
                except KeyError:
                    pass

                self._runs_history = []
            except:
                error('Invalid train plan!\n')
    def name(self):
        return self._name

    def execute(self, template_net, gpu_id=None):
        directory = os.path.abspath(self.name())
        base_name = os.path.normpath(self.name()).replace('/', '_')
        try:
            os.makedirs(directory)
        except OSError:
            if not os.path.isdir(directory):
                raise

        script_file = os.path.join(directory, 'train.py')
        if not os.path.isfile(script_file):
            with open(script_file, 'wb') as f:
                f.write(exec_script)
            try:
                subprocess.call('chmod +x %s' % script_file, shell=True)
            except:
                pass

        target_train_plan = os.path.join(directory, 'train_plan.json')
        if not os.path.isfile(target_train_plan):
            shutil.copyfile(self._train_plan_filename, target_train_plan)
        elif not filecmp.cmp(self._train_plan_filename, target_train_plan):
            error('Train plan present in target directory is different than the input train plan.')

        template_net_file = os.path.join(directory, 'net.prototmp')
        if not os.path.isfile(template_net_file):
            with open(template_net_file, 'wb') as f:
                f.write(template_net)
        else:
            with open(template_net_file, 'rb') as f:
                other_template = f.read()
            if other_template != template_net:
                error('The net template in target directory is different than the input template.')

        unique_csv_file = '%s/%s_unique%s.csv' % (directory, base_name, semi_unique())
        all_keys = set()
        update_csv_interval = random.randint(3,7)
        iterations_since_update = 0
        def extra_bytes(x):
            pass
        self._last_reprint = None
        self._window_context = CursorAwareWindow(hide_cursor=False, keep_last_line=True, extra_bytes_callback=extra_bytes)
        self._inside_window_context = False
        self._input_generator = Input(sigint_event=True)
        computer_name = platform.node()
        with self._input_generator:
            for indices, params in itertools.islice(self._params.search_iter(), self._max_data_points):
                should_break = False
                try:
                    indices = list(indices)
                except:
                    indices = [indices]
                    params = list(params)
                indices_str = '_'.join(map(str,indices))
                params.extend(list(itertools.imap(lambda x: ('run_idx_%d' % x[0], x[1]), enumerate(indices,1))))
                while True:
                    params_dict = dict(params)
                    base_filename = '%s/%s_%s' % (directory, base_name, indices_str)
                    params_dict['name'] = base_filename
                    if not params_dict.has_key('caffe_random_seed'):
                        params_dict['caffe_random_seed'] = random.randint(0, 0xffffffff)
                    if self._advanced_rendering_engine:
                        template = self._templateGenerator.from_string(template_net)
                        net = template.render(params_dict)
                    else:
                        net = template_net % params_dict
                    weights_file = self._weights_file
                    if weights_file is not None:
                        weights_file_glob = self._templateGenerator.from_string(self._weights_file).render(params_dict)
                        weights_files = glob.glob('%s/%s' % (directory, weights_file_glob))
                        weights_files.sort(key=os.path.getmtime)
                        if len(weights_files) == 0:
                            self._print('Cannot find weights file. Skipped run!')
                            break
                        weights_file = '/'.join(weights_files[-1].split('/')[1:])
                    description = ('i: %s | ' % indices_str) + ' | '.join(sorted(map(lambda x: '{}: {}'.format(x[0], ensure_precision(x[1], 4)), params)))
                    solve_state, run_params = self._solve(params_dict['name'], description, net, weights_file, gpu_id)
                    if solve_state == TrainPlan.EXISTS:
                        break
                    for key, value in params_dict.iteritems():
                        run_params['hyper__' + key] = value
                    run_params['run_id'] = indices_str if len(indices) > 1 else indices[0]
                    run_params['computer_name'] = computer_name
                    self._runs_history.append(flatten(run_params))
                    all_keys.update(self._runs_history[-1].keys())
                    with open(unique_csv_file , 'w') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=all_keys)
                        writer.writeheader()
                        writer.writerows(self._runs_history)
                    iterations_since_update += 1
                    if iterations_since_update > update_csv_interval:
                        self._update_main_csv(directory, base_name)
                        iterations_since_update = 0
                    if solve_state == TrainPlan.CAFFE_TERMINATED:
                        ans = ''
                        while ans.lower() not in ['y', 'n']:
                            ans = self._raw_input('Previous run has stopped unexpectedly. Continue? [y/n]')
                        if ans.lower() == 'n':
                            should_break = True
                            break
                    elif solve_state == TrainPlan.SKIPPED:
                        self._print('Skipped run!')
                        break
                    elif solve_state == TrainPlan.RETRY:
                        self._print('Retry run!')
                        logfile = '{}.log'.format(params_dict['name'])
                        os.remove(logfile)
                        continue
                    elif solve_state == TrainPlan.USER_ASK_EXIT:
                        should_break = True
                        break
                    break
                if should_break:
                    break
        self._input_generator = None
        if self._inside_window_context:
            self._inside_window_context.__exit__()
            self._inside_window_context = False
        self._last_reprint = None
        self._window_context = None
        self._update_main_csv(directory, base_name)
    def _update_main_csv(self, directory, base_name):
        regex = re.compile('.*_unique[0-9]+\.csv$')
        all_keys = set()
        rows = []
        for filename in filter(lambda x: regex.match(x) is not None, os.listdir(directory)):
            try:
                with open('%s/%s' % (directory, filename), 'rb') as incsv:
                    reader = csv.DictReader(incsv)
                    for row in reader:
                        rows.append(row)
                    all_keys.update(reader.fieldnames)
            except:
                pass
        all_keys = sorted(all_keys)
        with open('%s/%s.csv' % (directory, base_name), 'wb') as outcsv:
            writer = csv.DictWriter(outcsv, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(rows)
    def _string_to_fsarray(self, msg):
        if not self._inside_window_context:
            raise RuntimeError('Calling _string_to_fsarray outside of window context')
        rows, columns = self._window_context.get_term_hw()
        msg = fmtstr(msg)
        arr = FSArray(0, columns)
        i = 0
        lines = msg.split('\n') if '\n' in str(msg) else [msg]
        for line in lines:
            for j in xrange(len(line)):
                c = line[j]
                if i >= rows * columns:
                    return arr
                else:
                    arr[i // arr.width, i % arr.width] = [c]
                i += 1
            i = ((i // columns) + 1) * columns
        if len(arr) == 0:
            return fsarray([''])
        return arr
    def _print(self, s, add_new_line=True, bypass_curtsies=True):
        if self._last_reprint is not None:
            if self._inside_window_context:
                self._window_context.__exit__(None, None, None)
                self._inside_window_context = False
            self._last_reprint = None
        new_string = s
        if add_new_line:
            new_string += '\n'
        if bypass_curtsies:
            sys.stdout.write(str(new_string))
        else:
            with self._window_context:
                self._inside_window_context = True
                fsarr = self._string_to_fsarray(new_string)
                (totwidth, height), width = self._actual_size(fsarr)
                self._window_context.render_to_terminal(fsarr, cursor_pos=(max(height-1,0), max(width-1,0)))
                self._inside_window_context = False
    def _reprint(self, s, add_new_line=True):
        new_reprint = s
        if add_new_line:
            new_reprint += '\n'
        if new_reprint == self._last_reprint:
            return
        else:
            self._last_reprint = new_reprint
        if not self._inside_window_context:
            self._window_context.__enter__()
            self._inside_window_context = True
        fsarr = self._string_to_fsarray(self._last_reprint)
        (_, height), width = self._actual_size(fsarr)
        self._window_context.render_to_terminal(fsarr, cursor_pos=(max(height-1,0), max(width-1,0)))
    def _actual_size(self, fsarr):
        if not self._inside_window_context:
            raise RuntimeError('Calling _actual_size outside of window context')
        width = fsarr.width
        height = fsarr.height
        last_width = fsarr[max(height - 1, 0)].width
        return ((width, height), last_width)
    def _raw_input(self, s):
        self._get_all_input()
        base_str = s
        self._reprint(base_str, add_new_line=False)
        ans = ''
        last = ''
        while last != '<Ctrl-j>':
            last = self._input_generator.next()
            if type(last) != unicode:
                continue
            if last == '<BACKSPACE>':
                ans = ans[:-1]
            elif last == '<SPACE>':
                ans += ' '
            elif last != '<Ctrl-j>':
                ans += last
            self._reprint(base_str + ' ' + ans, add_new_line=False)
        self._print(' ')
        self._get_all_input()
        return ans
    def _special_user_input(self):
        inputs = self._get_all_input()
        res = []
        for inp in inputs:
            if type(inp) == events.SigIntEvent:
                res.append(TrainPlan.SIGINT)
                break
        for inp in inputs:
            if type(inp) == unicode and inp == '<Ctrl-j>':
                res.append(TrainPlan.NEWLINE)
                break
        return res
    def _get_all_input(self):
        inputs = []
        while True:
            res = self._input_generator.send(0)
            if res is not None:
                inputs.append(res)
            else:
                break
        return inputs
    def _solve(self, name, description, net, weights_file, gpu=None):
        logfile = '{}.log'.format(name)

        # Check if file already exists and if so don't solve this case
        os.utime(os.path.dirname(logfile), None) # touch the directory to refresh the filelisting's cache
        if os.path.isfile(logfile):
            return (TrainPlan.EXISTS, None)
        with open(logfile, 'a'):
            pass
        self._print('===================')
        self._print(description)

        # try:

        prototxt = name + '.prototxt'
        with open(prototxt, 'wb') as f:
            f.write(net)
        with open(logfile, 'wb') as f:
            self._loss_history = []
            cmd = "%s train -solver=%s" % (LOCAL_CAFFE_EXEC, prototxt)
            if weights_file is not None:
                cmd += " -weights=%s" % (os.path.join(os.path.dirname(prototxt), weights_file))
            if gpu is not None:
                cmd += " -gpu=%s" % (str(gpu))
            
            handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            p = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
            signal.signal(signal.SIGINT, handler)
            it = iterate_and_write(subprocess_lines_iterator(p, output='stderr'), f)
            logParser = LogParser(it, self._mva_window_size)
            params = logParser.parse_and_react(lambda params: self._react_to_log_parser(params, p))
            self._print(' ')
            code = p.poll()
            if code is None:
                p.terminate()
                if not p.poll():
                    p.kill()
                if logParser.last_code() != LogParser.STOP:
                    error('Caffe subprocess has not terminated as expected.')
            if logParser.last_code() == LogParser.STOP:
                return (logParser.extra_reaction_info(), params)
            if code is not None and code != 0:
                return (TrainPlan.CAFFE_TERMINATED, params)
            return (TrainPlan.SUCCESS, params)
# except:
#     error('Error while calling caffe executable.')

    def _react_to_log_parser(self, params, process):
        ret_value = (LogParser.CONTINUE, None)
        line_data = []
        iteration = None
        if params['iterations']:
            iteration = int(params['iterations'])
            line_data.append(('iters', iteration))

        if params['last_minibatch_loss']:
            if self._termination_rules['nan'] and math.isnan(params['last_minibatch_loss']):
                ret_value = (LogParser.STOP, TrainPlan.SKIPPED)
            loss = float(params['last_minibatch_loss'])
            line_data.append(('loss', loss))
            if iteration is not None and (len(self._loss_history) == 0 or self._loss_history[-1][0] < iteration):
                self._loss_history.append((iteration, loss))

        if params['last_lr']:
            line_data.append(('lr', params['last_lr']))

        for i, test_res in enumerate(params['test_results']):
            def format_percentage(x):
                try:
                    float_value = float(x)
                    return '%.2f%%' % (100 * float_value) if float_value < 1.0 else '100%'
                except ValueError:
                    return x
            loss = next(itertools.ifilter(lambda x: x[0] == 'loss', test_res), None)
            if loss:
                if self._termination_rules['nan'] and 'nan' in loss[1]:
                        ret_value = (LogParser.STOP, TrainPlan.SKIPPED)
            accuracy = next(itertools.ifilter(lambda x: x[0] == 'accuracy', test_res), None)
            if accuracy:
                formatted_value = format_percentage(accuracy[1])
                best_accuracy = next(itertools.ifilter(lambda x: x[0] == 'best_accuracy', test_res), None)
                moving_avg = next(itertools.ifilter(lambda x: x[0] == 'moving_avg', test_res), None)
                if moving_avg:
                    avg_formatted_value = format_percentage(moving_avg[1])
                else:
                    avg_formatted_value = ''
                
                if best_accuracy:
                    best_formatted_value = format_percentage(best_accuracy[1])
                    line_data.append(('acc %d' % i, '%s (b: %s, a: %s)' % (formatted_value, best_formatted_value, avg_formatted_value)))
                else:
                    line_data.append(('acc %d' % i, '%s (a: %s)' % (formatted_value, avg_formatted_value)))
        n = len(self._loss_history)
        if n > 10:
            mid = int(round(n/2.0))
            current_average = np.median([x for _, x in self._loss_history[mid:]])
            previous_average = np.median([x for _, x in self._loss_history[:mid]])
            iter_gap = self._loss_history[-1][0] - self._loss_history[-mid-1][0]
            if iter_gap > 0:
                delta = (current_average - previous_average) / (abs(previous_average)*iter_gap)
                line_data.append(('rel_avg_delta', delta))
                if delta > -self._termination_rules['delta']:
                    ret_value = (LogParser.STOP, TrainPlan.SKIPPED)
        self._reprint(bold(fmtstr(' | ').join([underline(key) + ':' + (' %s' % (ensure_precision(value, 2),)) for key, value in line_data])))
        user_input = self._special_user_input()
        if TrainPlan.NEWLINE in user_input:
            self._print(' ', add_new_line=False)
        if TrainPlan.SIGINT in user_input:
            self._print(' ')
            ans = ''
            while ans.lower() not in ['s', 'c', 'e', 'r']:
                ans = self._raw_input('Do you want to (s)kip this run, (c)ontinue running, (r)etry or (e)xit? ')
            if ans.lower() == 's':
                return (LogParser.STOP, TrainPlan.SKIPPED)
            elif ans.lower() == 'e':
                process.terminate()
                self._user_ask_exit = True
                return (LogParser.STOP, TrainPlan.USER_ASK_EXIT)
            elif ans.lower() == 'r':
                return (LogParser.STOP, TrainPlan.RETRY)

        return ret_value

class LogParser():
    CONTINUE, STOP = range(2)

    def __init__(self, lines_iter, mva_window_size):
        self._lines_iter = lines_iter
        self._last_code = LogParser.CONTINUE
        self._extra_reaction_info = None
        self._accuracies_dict = {}
        self._moving_avg_window_size = mva_window_size

    def log_summary(self):
        return self.parse_and_react()

    def last_code(self):
        return self._last_code
    def extra_reaction_info(self):
        return self._extra_reaction_info

    def parse_and_react(self, react_fun=lambda params: LogParser.CONTINUE):
        params = dict()
        # The last loss of the mini batch backword step
        params['last_minibatch_loss'] = None
        # The current learning rate
        params['last_lr'] = None
        # Was the optimization of the network completed
        params['optimization_done'] = False
        # Find the start and end time of the optimization
        params['start_time'] = None
        params['end_time'] = None
        params['total_time'] = 0
        # The current interation number
        params['iterations'] = 0
        # The list of the last test results
        params['test_results'] = []

        # Set to true when the next line might contain test results
        test_result_on_next_line = False
        # Sets the index of the test result that might be reported in the next lines
        test_index_on_next_line = -1
        
        # Helpful regexes
        iterationsRegex = re.compile('Iteration ([0-9]*)')
        testRegex = re.compile('Testing net \\(#([0-9]*)\\)')
        testOutputRegex = re.compile('Test net output #([0-9]*): (\w*) = (.*)')
        minibatchLossRegex = re.compile('Iteration [0-9]*, loss = ([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?|nan)')
        lrRegex = re.compile('Iteration [0-9]*, lr = ([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?|nan)')
        prefetch_error = 'Data layer prefetch queue empty'
        for line in self._lines_iter:
            match = iterationsRegex.search(line)
            if match:
                params['iterations'] = match.group(1)
                self._last_code, self._extra_reaction_info = react_fun(params)
                if self.last_code() is LogParser.STOP:
                    break
            
            parts = line.split(' ')
            if len(parts) >= 2:
                try:
                    current_year = datetime.date.today().year
                    params['end_time'] = datetime.datetime.strptime('%d %s %s' %(current_year,parts[0], parts[1]), '%Y I%m%d %H:%M:%S.%f')
                    if params['start_time'] is None:
                        params['start_time'] = params['end_time']
                    params['total_time'] = (params['end_time'] - params['start_time']).seconds
                    self._last_code, self._extra_reaction_info = react_fun(params)
                    if self.last_code() is LogParser.STOP:
                        break
                except ValueError:
                    pass

            # Check that optimization was completed and wasn't stopped / is in progress
            if 'Optimization Done' in line:
                params['optimization_done'] = True
                self._last_code, self._extra_reaction_info = react_fun(params)
                if self.last_code() is LogParser.STOP:
                    break
                continue

            # Detect minibatch backword loss report
            match = minibatchLossRegex.search(line)
            if match:
                params['last_minibatch_loss'] = float(match.group(1))
                self._last_code, self._extra_reaction_info = react_fun(params)
                if self.last_code() is LogParser.STOP:
                    break
                continue

            # Detect learning rate report
            match = lrRegex.search(line)
            if match:
                params['last_lr'] = float(match.group(1))
                self._last_code, self._extra_reaction_info = react_fun(params)
                if self.last_code() is LogParser.STOP:
                    break
                continue

            # Detect the start of a test report
            match = testRegex.search(line)
            if match:
                test_result_on_next_line = True
                test_index_on_next_line = int(match.group(1))
                continue
            if prefetch_error in line:
                continue
            # Parse test results
            if test_result_on_next_line:
                match = testOutputRegex.search(line)
                if match:
                    test_results = params['test_results']
                    while len(test_results) <= test_index_on_next_line:
                        test_results.append([])
                    test_outputs = test_results[test_index_on_next_line]
                    output_index = int(match.group(1))
                    while len(test_outputs) <= output_index:
                        test_outputs.append(None)
                    output_key = match.group(2)
                    output_value = match.group(3)
                    test_outputs[output_index] = (output_key, output_value)
                    test_results[test_index_on_next_line] = test_outputs
                    params['test_results'] = test_results
                    self._last_code, self._extra_reaction_info = react_fun(params)
                    if self.last_code() is LogParser.STOP:
                        break
                    continue
                else:
                    test_result_on_next_line = False
                    test_index_on_next_line = -1
                    test_results = params['test_results']

                    for test_idx in range(len(test_results)):
                        best_accuracy_idx = -1
                        accuracy_idx = -1
                        moving_avg_idx = -1
                        for i, pair in enumerate(test_results[test_idx]):
                            if pair[0] == 'best_accuracy':
                                best_accuracy_idx = i
                            if pair[0] == 'accuracy':
                                accuracy_idx = i
                            if pair[0] == 'moving_avg':
                                moving_avg_idx = i
                        if accuracy_idx >= 0:
                            try:
                                accuracy = float(test_results[test_idx][accuracy_idx][1])
                                if best_accuracy_idx == -1:
                                    test_results[test_idx].append(('best_accuracy', accuracy))
                                else:
                                    prev_accuracy = test_results[test_idx][best_accuracy_idx][1]
                                    new_accuracy = max(prev_accuracy, accuracy)
                                    test_results[test_idx][best_accuracy_idx] = ('best_accuracy', new_accuracy)

                                if self._accuracies_dict.has_key(test_idx):
                                    
                                    self._accuracies_dict[test_idx].insert(accuracy)
                                    if moving_avg_idx == -1:
                                        test_results[test_idx].append(('moving_avg', self._accuracies_dict[test_idx].get_mean()))
                                    else:
                                        test_results[test_idx][moving_avg_idx] = ('moving_avg', self._accuracies_dict[test_idx].get_mean())
                                else:
                                    self._accuracies_dict[test_idx] = MovingAverageWindow(self._moving_avg_window_size)
                            except ValueError:
                                pass
                    params['test_results'] = test_results
            
            self._last_code, self._extra_reaction_info = react_fun(params)
            if self.last_code() is LogParser.STOP:
                break
        return params


def main(argv):
    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "template_net",
        help="A template network architecture protobuf file, with %%(param_name)t as place holder for param_name"
    )
    parser.add_argument(
        "train_plan",
        help="A plan how to train the network: name of the plan, the sampling to use for hyperparameters,  etc."
    )
    parser.add_argument(
        "--gpu",
        help="The device id to use when running on the GPU",
        default=None
    )
    args = parser.parse_args()

    if not os.path.isfile(args.template_net):
        print 'Cannot find template file'
        exit()
    if not os.path.isfile(args.train_plan):
        print 'Cannot find train plan file'
        exit()

    template_net = None
    with open(args.template_net, 'r') as f:
        template_net = f.read()
    
    train_plan = TrainPlan(args.train_plan)
    train_plan.execute(template_net, args.gpu)

if __name__ == '__main__':
    main(sys.argv)

