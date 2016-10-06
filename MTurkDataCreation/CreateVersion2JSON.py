import os,sys,json,gzip
import requests,time

directory = "devset"
directory = "trainset"
directory = "testset"

#from simple
#organize by file name
#get locations from state.json
#those correspond to steps from task.json and screencapid from state.json
#hit.json has text linked to action
#annotated actions are in the hit.json


#if not os.path.exists(directory):
#    os.makedirs(directory)

def smash_the_server(cmd):
  r = None
  while r == None:
    try:
      r = requests.get(cmd)
    except:
      r = None
      time.sleep(1)
  return r.text


filenames = {}
for root,folders,files in os.walk("RawTurkData/Version2/", 'r'):
  J = {}
  for name in files:
    if "txt" in name:
      print os.path.join(root, name)
      for line in open(os.path.join(root, name),'r'):
        line = line.split()
        if len(line) > 0:
          if line[0] == "State:":
            filename = line[3]
          elif line[0] == "HIT:":
            cmd = "http://cwc-isi.org/api/hit/%s" % line[1]
            hit = json.loads(smash_the_server(cmd))
          elif "EXAMPLE" in line[0]:
            task = line[0].split("&")[0].split("=")[1]
            cmd = "http://cwc-isi.org/api/task/%s" % task
            task = json.loads(smash_the_server(cmd))
            cmd = "http://cwc-isi.org/api/state/%s" % task["stateid"]
            state = json.loads(smash_the_server(cmd))
            if filename not in filenames:
              filenames[filename] = {"simple":[],"partial":[],"full":[]}
            filenames[filename]["simple"].append({"hit":hit, "task":task, "state":state})

print len(filenames)
for file in filenames:
  print file, len(filenames[file]["simple"])

o = gzip.open("version2" + ".json.gz", 'w')
o.write(json.dumps(filenames) + "\n")
o.close()
