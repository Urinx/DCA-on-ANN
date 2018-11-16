#!/usr/bin/env python
# coding: utf-8

# ====================================
import sys
import random
import os
import shutil
import time
import itertools
import datetime
import pickle as Pickle
# ====================================


def timeit(fn):
    def wrapper(*args):
        n = 1
        startTime = time.time()
        for i in range(n): result = fn(*args)
        totalTime = (time.time()-startTime) / n
        print('[*] Test function {0}() {1} times, average: {2} s'.format(fn.__name__, n, totalTime))
        return result
    return wrapper


GLOBAL_START_TIME = time.time()
GLOBAL_TEMPORARY_TIME = GLOBAL_START_TIME

def time_cost():
    global GLOBAL_TEMPORARY_TIME
    now = time.time()
    totalTime = now - GLOBAL_TEMPORARY_TIME
    GLOBAL_TEMPORARY_TIME = now
    t = datetime.timedelta(seconds=totalTime)
    return t


def size_formating(s_byte):
    """
    @brief      format the bytes into a string, eg. 'xx MB' etc.
    @param      s_byte, int of bytes
    @return     'xx KB' or 'xx MB' etc.
    """
    if s_byte > 1024:
        s_kb = s_byte >> 10
        if s_kb > 1024:
            s_mb = s_kb >> 10
            if s_mb > 1024:
                s_gb = s_mb >> 10
                return '%d GB' % s_gb
            else:
                return '%d MB' % s_mb
        else:
            return '%d KB' % s_kb
    else:
        return '%d B' % s_byte


def get_size_of(obj):
    """
    @brief      get the space size of the object in cache
    """
    s_byte = sys.getsizeof(obj)
    return size_formating(s_byte)


def get_file_size_of(file_name):
    s_byte = os.path.getsize(file_name)
    return size_formating(s_byte)


def echo(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def fasta_read(fasta_file):
    """
    @brief      read fasta format file
    @param      fasta_file  The fasta file path
    @return     {
                    count: 0,
                    head: [],
                    ac: [],
                    seq: [],
                }
    """
    head, ac, seq = [], [], []
    n = 0
    with open(fasta_file) as f:
        for line in f:
            l = line.strip()
            if l and l[0] == '>':
                n += 1
                head.append(l)
                ac.append(l.split(' ')[0][1:])
                seq.append('')
            else: seq[n-1] += l
    return {'count': n, 'head': head, 'ac': ac, 'seq': seq}


def random_select_seqs(input_file, output_file, n):
    """
    @brief      random select seqs from huge seqs into another file
    @param      input_file   The input file
    @param      output_file  The output file
    @param      n            Number of selected seqs
    """
    r = fasta_read(input_file)
    if r['count'] <= n:
        shutil.copyfile(input_file, output_file)
    else:
        with open(output_file, 'w') as f:
            for i in random.sample(range(r['count']), n):
                f.write('%s\n' % r['head'][i])
                f.write('%s\n' % r['seq'][i])


def file_line_number(file_name):
    """
    @brief      count the lines number of file
    @param      file_name   The input file
    @return     lines number of file
    @link       http://stackoverflow.com/questions/845058/how-to-get-line-count-cheaply-in-python
    """
    f = open(file_name)
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.read # loop optimization
    buf = read_f(buf_size)
    while buf:
        lines += buf.count('\n')
        buf = read_f(buf_size)
    f.close()
    return lines


def random_select_seqs_in_large_file(input_file, output_file, n):
    """
    @brief      random select seqs from huge seqs into another file
                (input_file > 2G)
    @param      input_file   The input file
    @param      output_file  The output file
    @param      n            Number of selected seqs
    """
    fi = open(input_file, 'r')
    fo = open(output_file, 'w')

    line_offset = []
    offset = 0
    for line in fi:
        line_offset.append(offset)
        offset += len(line)
    fi.seek(0)

    seqn = len(line_offset) / 2
    if seqn <= n:
        shutil.copyfile(input_file, output_file)
    else:
        for i in random.sample(range(seqn), n):
            fi.seek(line_offset[2*i])
            fo.write(fi.readline())
            fo.write(fi.readline())

    fi.close()
    fo.close()


def T2U_in_rfam(rfam_file):
    """
    @brief      change DNA seqs to RNA seqs by just
                replace T to U.
    @param      rfam_file   The input file
    """
    tmp_file = 'tmp'
    tmp_f = open(tmp_file, 'w+')
    with open(rfam_file, 'r') as f:
        for line in f:
            l = line.strip()
            if l[0] != '>':
                l = l.replace('T', 'U')
            tmp_f.write('%s\n' % l)
    tmp_f.close()
    os.remove(rfam_file)
    os.rename(tmp_file, rfam_file)


def half_loop_through_fij(n, q):
    for (A, B) in itertools.product(range(q), range(q)):
        for (i, j) in itertools.combinations(range(n), 2):
            yield (i, j, A, B)


def pickle_dump(el, file):
    with open(file, 'w') as f:
        Pickle.dump(el, f)


def pickle_load(file):
    with open(file, 'r') as f:
        return Pickle.load(f)

