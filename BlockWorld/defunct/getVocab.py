import sys
import gzip
import json
import codecs
from nltk.tokenize import TreebankWordTokenizer

Text = []
Vocab = {}
longest = 0

def onehot(utr):
  unk = [0]*len(Vocab)
  unk[0] = 1
  v = []
  for i in range(longest):
    if i >= len(utr):
      v.extend(unk)
    else:
      t = [0]*len(Vocab)
      t[Vocab[utr[i]]] = 1
      v.extend(t)
  return v

for line in gzip.open(sys.argv[1],'r'):
  j = json.loads(line)
  text = TreebankWordTokenizer().tokenize(j["text"])
  Text.append(text)
  if len(text) > longest:
    longest = len(text)
  for word in text:
    if word not in Vocab:
      Vocab[word] = 0
    Vocab[word] += 1

V = codecs.open("Vocab.txt", "w", "utf-8")
for word in Vocab:
  V.write("%-15s  %-2d\n" % (word, Vocab[word]))
V.close()
