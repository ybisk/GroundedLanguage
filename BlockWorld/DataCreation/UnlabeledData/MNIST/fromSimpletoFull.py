import os,sys,json

for section in ["Train","Dev","Test"]:
  for root,dirs,files in os.walk("SimpleActions/MTurk/%s/" % section):
    for name in files:
      out = open("FullSequences/%s/%s" % (section,name), 'w')
      j = json.load(open(os.path.join(root,name)))
      j["block_states"] = [j["block_states"][0],j["block_states"][len(j["block_states"])-1]]
      out.write(json.dumps(j))
