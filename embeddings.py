# Copyright (C) 2016-2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from cupy_utils import *

import numpy as np
import io


def read(file, threshold=0, vocabulary=None, dtype='float'):
    header = file.readline().split(' ')
    count = int(header[0]) if threshold <= 0 else min(threshold, int(header[0]))
    dim = int(header[1])
    words = []
    matrix = np.empty((count, dim), dtype=dtype) if vocabulary is None else []
    # matrix = np.empty((count, length_normalize_dimensionwise), dtype=dtype) if vocabulary is None else []
    # matrix = []
    # print(file)
    # with io.open(file, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
    for i in range(count):
        # for i, line in enumerate(f):
        line = file.readline().split(' ', 1)
        if len(line) > 1:
            word, vec = line
            # print(len(vec))
            if vocabulary is None:
                words.append(word)
                temp = np.fromstring(vec, sep=' ', dtype=dtype)
                if len(temp) == dim:
                    matrix[i] = np.fromstring(vec, sep=' ', dtype=dtype)
                # print(len(temp))
                # matrix.append(np.fromstring(vec, sep=' ', dtype=dtype))
            elif vocabulary is not None and word in vocabulary:
                words.append(word)
                matrix.append(np.fromstring(vec, sep=' ', dtype=dtype))
    matrix = np.array(matrix)
    return (words, matrix) if vocabulary is None else (words, np.array(matrix, dtype=dtype))

def write(words, matrix, file):
    m = asnumpy(matrix)
    print('%d %d' % m.shape, file=file)
    for i in range(len(words)):
        print(words[i] + ' ' + ' '.join(['%.6g' % x for x in m[i]]), file=file)


def length_normalize(matrix):
    xp = get_array_module(matrix)
    # print(matrix.shape)
    norms = xp.sqrt(xp.sum(matrix**2, axis=1))
    norms[norms == 0] = 1
    matrix /= norms[:, xp.newaxis]


def mean_center(matrix):
    xp = get_array_module(matrix)
    avg = xp.mean(matrix, axis=0)
    matrix -= avg


def length_normalize_dimensionwise(matrix):
    xp = get_array_module(matrix)
    norms = xp.sqrt(xp.sum(matrix**2, axis=0))
    norms[norms == 0] = 1
    matrix /= norms


def mean_center_embeddingwise(matrix):
    xp = get_array_module(matrix)
    avg = xp.mean(matrix, axis=1)
    matrix -= avg[:, xp.newaxis]


def normalize(matrix, actions):
    for action in actions:
        if action == 'unit':
            length_normalize(matrix)
        elif action == 'center':
            mean_center(matrix)
        elif action == 'unitdim':
            length_normalize_dimensionwise(matrix)
        elif action == 'centeremb':
            mean_center_embeddingwise(matrix)
