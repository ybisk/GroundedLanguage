import os,sys,commands,json,gzip

o = gzip.open("out.json.gz", 'w')
for root,folders,files in os.walk(sys.argv[1],'r'):
  J = {}
  for name in files:
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
          J["file"] = filename
          J["HIT"] = hit
          J["task"] = task
          J["state"] = state
          o.write(json.dumps(J) + "\n")
  a,b = commands.getstatusoutput("rm tmp.json")
