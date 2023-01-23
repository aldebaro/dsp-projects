import pandas as pd
import itertools
import matplotlib.pyplot as plt

from collections import Counter
from lib_dolphin.htk_helpers import *
from scipy.stats import chisquare


class Strings(namedtuple('Strings', 'l1 l2 label file start_l1 stop_l1 start_l2 stop_l2')):
            
    def ngrams_l1(self, n):
        if len(self.l1) >= n:
            strg  = [(" ".join(self.l1[i-n:i]), self.start_l1[i-n], self.stop_l1[i], self.file) for i in range(n, len(self.l1))]
            return strg
        return []
    
    def ngrams_l2(self, n):
        if len(self.l2) >= n:
            strg  = [(" ".join(self.l2[i-n:i]), self.start_l2[i-n], self.stop_l2[i], self.file) for i in range(n, len(self.l2))]
            return strg
        return []

    
def parse_l2(sequencing_file, T, FFT_STEP, perc=0.01, do_compress=False):
    annotations = parse_mlf(sequencing_file)
    win         = T // 2 * FFT_STEP
    likelihoods = []
    labels      = []
    regions     = [] 
    for f, x in annotations.items():
        if len(x) > 1 and do_compress:
            x = compress(x)
        for start, stop, c, ll in x:
            start *= win
            stop  *= win
            likelihoods.append(ll)
            labels.append(c)
            regions.append((f, start, stop))
    likelihoods = sorted(likelihoods)
    th = int(len(likelihoods) * perc)
    starts = []
    stops  = []
    files  = []
    lab    = []
    for i in range(0, len(regions)):
        if likelihoods[i] < th and labels[i] != 'sil':
            file, start, stop = regions[i]
            lab.append(labels[i])
            starts.append(start)
            stops.append(stop)
            files.append(file)
    return starts, stops, files, lab
    
    
def parse_l1(sequencing_file, T, FFT_STEP):
    annotations = pd.read_csv(sequencing_file)
    win         = FFT_STEP
    annotations['start'] = annotations['start'].apply(lambda x: x * win)
    annotations['stop']  = annotations['stop'].apply(lambda x: x * win)
    annotations['filenames']  = annotations['filenames'].apply(lambda x: x.replace('.wav', ''))

    starts = list(annotations['start'])
    stops  = list(annotations['stop'])
    files  = list(annotations['filenames'])
    label  = list(annotations['label'])
    return starts, stops, files, label


def by_file(files, symbols, starts, stops):
    f = {}
    cur_f = files[0]
    symbols = [str(s) for s in symbols]
    cur_symb   = [symbols[0]]
    cur_starts = [starts[0]]
    cur_stops  = [stops[0]]
    for i in range(1, len(files)):
        if files[i] != cur_f:
            f[cur_f] = (cur_symb, cur_starts, cur_stops)
            cur_f    = files[i]
            cur_symb   = [symbols[i]]
            cur_starts = [starts[i]]
            cur_stops  = [stops[i]] 
        else:
            cur_symb.append(symbols[i])
            cur_starts.append(starts[i])
            cur_stops.append(stops[i])
    return f


def stats_key(counters, th=3):
    total = Counter([]) 
    for c in counters.values():
        total += c
        
    key_dict = {}
    cur = 0
    for i, k in enumerate(list(total.keys())):
        if total[k] >= th and k not in key_dict:
            key_dict[k]= cur
            cur += 1
    n = max(key_dict.values()) + 1

    labels = ["" for i in range(0, n)]
    for k, i in key_dict.items():
        labels[i] = k
    return key_dict, labels, n


def vectorize(keys, counters, n, normalize=True):
    vectors = {}
    for label, counter in counters.items():
        v = np.zeros(n)
        for k, c in counter.items():    
            if k in keys:
                v[keys[k]] += c
        if normalize:
            v = v / np.sum(v)
        vectors[label] = v    
    return vectors


def highest_difference(vx, vy, kx, ky, labels, k = 25):
    if k is None:
        k == len(labels)
    diff =  vx - vy
    diff = [(labels[i], diff[i]) for i in range(len(diff))]
    diff.sort(key = lambda x: -np.abs(x[1]))
    diff = [(diff[i][0], kx if diff[i][1] > 0 else ky) for i in range(0, k)]
    return diff
    
    
def ngram_statistics(l1, l2, label_f, output, T, FFT_STEP, N = 10):
    l1_starts, l1_stops, l1_files, l1_labs = parse_l1(l1, T, FFT_STEP)
    l2_starts, l2_stops, l2_files, l2_labs = parse_l2(l2, T, FFT_STEP)
    l1_byfile = by_file(l1_files, l1_labs, l1_starts, l1_stops)
    l2_byfile = by_file(l2_files, l2_labs, l2_starts, l2_stops)
    stats = {}
    for key in l1_byfile.keys():
        label = label_f(key)
        if label not in stats:
            stats[label] = []
        if key in l1_byfile and key in l2_byfile:
            stats[label].append(
                Strings(l1_byfile[key][0], l2_byfile[key][0], label, key, l1_byfile[key][1], l1_byfile[key][2],  l2_byfile[key][1], l2_byfile[key][2]))

    counters = { key: Counter([]) for key in stats.keys()}
    for i in range(1, N):
        for l in stats.keys():
            l1 = list(itertools.chain(*[[g[0] for g in strg.ngrams_l1(i)] for strg in stats[l]]))
            l2 = list(itertools.chain(*[[g[0] for g in strg.ngrams_l2(i)] for strg in stats[l]]))
            counters[l] += Counter(l1) + Counter(l2)
    keys, labels, n = stats_key(counters)
    vectorized = vectorize(keys, counters, n)
    
    diffs = []
    for kx, vx in vectorized.items():
        for ky, vy in vectorized.items():
            if kx != ky:
                p = chisquare(vx, vy).pvalue
                if p < 0.001:
                    print(f"\t Significant {kx} / {ky}: {p} p")
                    diff = highest_difference(vx, vy, kx, ky, labels)
                    for d in diff:
                        diffs.append(d)
                        
                    img = f'{output}/feature_count_{kx}_{ky}.png'
                    print(f"... writing image {img}")
                    plt.figure(figsize=(90, 10))
                    plt.bar(np.arange(n), vx, label=kx, alpha=0.5)
                    plt.bar(np.arange(n), vy, label=ky,  alpha=0.5)
                    plt.legend()
                    plt.title(f'Comparative Statistics {kx} {ky} {p}')
                    plt.xticks(np.arange(n), labels, rotation=90)
                    plt.savefig(img)
                    plt.close()
                    
                    
    by_pattern = {}
    for pattern, label in diffs:
        for strg in stats[label]:
            for i in range(1, N):
                for ngram, start, stop, file in strg.ngrams_l1(i):
                    if ngram == pattern:
                        if ngram not in by_pattern:
                            by_pattern[ngram] = []
                        by_pattern[ngram].append((start, stop, file, label))
                for ngram, start, stop, file in strg.ngrams_l2(i):
                    if ngram == pattern:
                        if ngram not in by_pattern:
                            by_pattern[ngram] = []
                        by_pattern[ngram].append((start, stop, file, label))
    return by_pattern