from itertools import groupby
import numpy as np

morse = {'.-': 'a',     '-...': 'b',   '-.-.': 'c',
         '-..': 'd',    '.': 'e',      '..-.': 'f',
         '--.': 'g',    '....': 'h',   '..': 'i',
         '.---': 'j',   '-.-': 'k',    '.-..': 'l',
         '--': 'm',     '-.': 'n',     '---': 'o',
         '.--.': 'p',   '--.-': 'q',   '.-.': 'r',
         '...': 's',    '-': 't',      '..-': 'u',
         '...-': 'v',   '.--': 'w',    '-..-': 'x',
         '-.--': 'y',   '--..': 'z',   ' ': ' '}

def regroup(blocks):
    """Merge adjacent blocks of same type."""
    blocks_wide = [[a]*b for a, b in blocks]
    blocks_flat = [item for sublist in blocks_wide for item in sublist]
    return [(k, sum(1 for i in g)) for k, g in groupby(blocks_flat)]

def decode(samples):
    """Decode the message."""
    # primitive thresholding and splitting into blocks of same value
    # we will use 0.5 as a threshold since the problem guarantees that the avg 
    # of all dots/dashes is 1 and avg of all silences is 0
    samples = [1 if s > 0.5 else 0 for s in samples]
    blocks = [(k, sum(1 for i in g)) for k, g in groupby(samples)]
    # hacky way to deal with blocks shorter than 5 (noise)
    blocks = [(-1, b) if b < 5 else (a, b) for a, b in blocks]
    blocks = regroup(blocks)
    for i, block in enumerate(blocks):
        if block[0] == -1:
            if i > 0 and i < len(blocks) - 1:
                half = block[1] // 2
                blocks[i-1] = (blocks[i-1][0], blocks[i-1][1] + block[1] - half)
                blocks[i+1] = (blocks[i+1][0], blocks[i+1][1] + half)
            elif i > 0:
                blocks[i-1] = (blocks[i-1][0], blocks[i-1][1] + block[1])
            elif i < len(blocks) - 1:
                blocks[i+1] = (blocks[i+1][0], blocks[i+1][1] + block[1])
            else:
                break
            blocks[i] = (-1, 0)
    blocks = regroup(blocks)
    # find relevant block lengths
    vals = sorted([b for (a, b) in blocks])
    diffs = np.diff(vals)
    ind = np.argpartition(diffs, -2)[-2:]
    ups = (vals[ind[0]], vals[ind[1]])
    # map block length to block type
    blocks = [(a, 1) if b <= ups[0] else (a, 2) if b <= ups[1] else (a, 3) for (a, b) in blocks]
    # map block type to appropriate character
    dec = {(1, 1) : '.', (1, 2) : '-', (0, 1) : '', (0, 2) : '/', (0, 3) : '/ /'}
    arr = ''.join([dec[tup] for tup in blocks]).split('/')
    # extract letters
    decoded = [morse[a] for a in arr]
    return ''.join(decoded)

if __name__ == "__main__":
    in_file = raw_input()
    fh = open(in_file, 'r')
    samples = [float(s) for s in fh.readlines()]
    print(decode(samples))