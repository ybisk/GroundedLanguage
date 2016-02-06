import os,sys,json

for section in ["Train","Dev","Test"]:
  Mask = {}
  for line in open("PartialSequences/%s/Mask.txt" % section,'r'):
    line = line.split()
    Mask[line[0]] = [int(v) for v in line[1:]]
  for name in Mask:
    input = open("SimpleActions/MTurk/%s/%s" % (section, name))
    out = open("PartialSequences/%s/%s" % (section,name), 'w')
    j = json.load(open("SimpleActions/MTurk/%s/%s" % (section,name)))
    A = []
    for i in range(len(Mask[name])):
      A.append(j["block_states"][Mask[name][i]])
    j["block_states"] = A
    j["type"] = "partial"
    out.write(json.dumps(j))
