import os,sys,json

for section in ["Train","Dev","Test"]:
  for root,dirs,files in os.walk("SimpleActions/MTurk/%s/" % section):
    for name in files:
      out = open("FullSequences/%s/full_%s" % (section,name), 'w')
      j = json.load(open(os.path.join(root,name)))
      j["block_states"] = [j["block_states"][0],j["block_states"][len(j["block_states"])-1]]
      j["type"] = "full"
      out.write(json.dumps(j))
