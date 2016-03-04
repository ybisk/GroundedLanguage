import os,sys,commands,json,gzip

directory = "trainset"
#directory = "devset"
#directory = "testset"

#from simple
#organize by file name
#get locations from state.json
#those correspond to steps from task.json and screencapid from state.json
#hit.json has text linked to action
#annotated actions are in the hit.json


#if not os.path.exists(directory):
#    os.makedirs(directory)

filenames = {}
for root,folders,files in os.walk("RawTurkData/MNIST/SimpleActions/" + directory, 'r'):
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
            a,b = commands.getstatusoutput("curl http://cwc-isi.org/api/hit/%s -o tmp.json" % line[1])
            hit = json.load(open("tmp.json"))
          elif "EXAMPLE" in line[0]:
            task = line[0].split("&")[0].split("=")[1]
            a,b = commands.getstatusoutput("curl http://cwc-isi.org/api/task/%s -o tmp.json" % task)
            task = json.load(open("tmp.json"))
            a,b = commands.getstatusoutput("curl http://cwc-isi.org/api/state/%s -o tmp.json" % task["stateid"])
            state = json.load(open("tmp.json"))
            if filename not in filenames:
              filenames[filename] = {"simple":[],"partial":[],"full":[]}
            filenames[filename]["simple"].append({"hit":hit, "task":task, "state":state})

for root,folders,files in os.walk("RawTurkData/MNIST/PartialSequences/" + directory, 'r'):
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
            a,b = commands.getstatusoutput("curl http://cwc-isi.org/api/hit/%s -o tmp.json" % line[1])
            hit = json.load(open("tmp.json"))
          elif "EXAMPLE" in line[0]:
            task = line[0].split("&")[0].split("=")[1]
            a,b = commands.getstatusoutput("curl http://cwc-isi.org/api/task/%s -o tmp.json" % task)
            task = json.load(open("tmp.json"))
            a,b = commands.getstatusoutput("curl http://cwc-isi.org/api/state/%s -o tmp.json" % task["stateid"])
            state = json.load(open("tmp.json"))
            filenames[filename]["partial"].append({"hit":hit, "task":task, "state":state})

for root,folders,files in os.walk("RawTurkData/MNIST/FullSequences/" + directory, 'r'):
  J = {}
  for name in files:
    if "txt" in name:
      print os.path.join(root, name)
      for line in open(os.path.join(root, name),'r'):
        line = line.split()
        if len(line) > 0:
          if line[0] == "State:":
            filename = line[3].replace("full_","")
          elif line[0] == "HIT:":
            a,b = commands.getstatusoutput("curl http://cwc-isi.org/api/hit/%s -o tmp.json" % line[1])
            hit = json.load(open("tmp.json"))
          elif "EXAMPLE" in line[0]:
            task = line[0].split("&")[0].split("=")[1]
            a,b = commands.getstatusoutput("curl http://cwc-isi.org/api/task/%s -o tmp.json" % task)
            task = json.load(open("tmp.json"))
            a,b = commands.getstatusoutput("curl http://cwc-isi.org/api/state/%s -o tmp.json" % task["stateid"])
            state = json.load(open("tmp.json"))
            filenames[filename]["full"].append({"hit":hit, "task":task, "state":state})

print len(filenames)
for file in filenames:
  print file, len(filenames[file]["simple"]), len(filenames[file]["partial"]), len(filenames[file]["full"])

o = gzip.open(directory + ".json.gz", 'w')
o.write(json.dumps(filenames) + "\n")
o.close()
a,b = commands.getstatusoutput("rm tmp.json")
