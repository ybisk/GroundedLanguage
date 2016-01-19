import os
import sys
import json
import gzip

digits = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
logos = [ \
    'adidas', 'bmw', 'burger king', 'coca cola', 'esso', \
    'heineken', 'hp', 'mcdonalds', 'mercedes benz', 'nvidia', \
    'pepsi', 'shell', 'sri', 'starbucks', 'stella artois', \
    'target', 'texaco', 'toyota', 'twitter', 'ups']

## Read in the world configurations ##
worlds = {}
for root,folders,files in os.walk(sys.argv[1],'r'):
  for name in files:
    justname = name.split(".json")[0].lower()
    worlds[justname] = json.load(open(os.path.join(root, name)))

for name in worlds:
  print name, worlds[name]["block_meta"]["decoration"]

out = gzip.open("data.json.gz", 'w')
## Read in the annotations JSONs ##
for line in gzip.open(sys.argv[2],'r'):
  j = json.loads(line)
  world = worlds[j["file"]]
  print j["file"], world["block_meta"]["decoration"]
  approved = []
  tasks = {}
  index = 0
  for job in j["HIT"]["submitted"]:
    if "valid" not in job or job["valid"] == "yes":
      approved.append(job["name"])
  for idxlist in j["task"]["idxlist"]:
    tasks[index] = idxlist
    index += 1
  for user in j["HIT"]["notes"]:
    if user in approved:
      for stage in j["HIT"]["notes"][user]:
        pair = tasks[int(stage)]
        for phrase in j["HIT"]["notes"][user][stage]:
          start = world["block_states"][pair[0]]
          end = world["block_states"][pair[1]]
          decor = logos if world["block_meta"]["decoration"] == "logo" else digits
          out.write(json.dumps({"current": start["block_state"], "next": end["block_state"],
                                "utterance": phrase, "decoration": decor}) + "\n")
out.close()