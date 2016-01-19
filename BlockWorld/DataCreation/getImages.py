import sys,json,gzip,commands
import requests

for line in gzip.open(sys.argv[1],'r'):
  j = json.loads(line)
  for idx in range(len(j["state"]["block_states"])):
    dir = sys.argv[2] + "/" + j["file"]
    commands.getstatusoutput("mkdir %s" % dir)
    cmd = "https://cwc-isi.org/api/screencap/%s" % (j["state"]["block_states"][idx]["screencapid"])
    r = requests.get(cmd)
    print cmd
    fh = open("%s/%d.png" % (dir,idx), "wb")
    fh.write(r.text.split(",",1)[1].decode('base64'))
    fh.close()