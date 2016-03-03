import os,sys,commands,json,gzip
import requests

directory = "devset"
directory = "testset"
directory = "trainset"

#from simple
#organize by file name
#get locations from state.json
#those correspond to steps from task.json and screencapid from state.json
#hit.json has text linked to action
#annotated actions are in the hit.json

def convert(query, states):
  for i in range(len(states)):
    state = states[i]
    match = True
    for j in range(len(state["block_state"])):
      match = match and state["block_state"][j]["position"]["x"] == query["block_state"][j]["position"]["x"] \
                    and state["block_state"][j]["position"]["y"] == query["block_state"][j]["position"]["y"] \
                    and state["block_state"][j]["position"]["z"] == query["block_state"][j]["position"]["z"]
    if match:
      return i
  print "FAIL"
  sys.exit()

if not os.path.exists(directory):
  os.makedirs(directory)

j = json.load(gzip.open(directory + ".json.gz"))
images = []
JSON = {}
for file in j:
  ## Read information about the Simple state
  states = j[file]["simple"][0]["state"]["block_states"]
  for state in states:
    del state["created"]
  notes = {}
  for job in j[file]["simple"]:
    # Extract users who did the task correctly
    valid = []
    for user in job["hit"]["submitted"]:
      if "valid" not in user:
        print "missing", job["hit"]["jid"]
      elif user["valid"] != "no":
        valid.append(user["name"])
    transitions = job["task"]["idxlist"]
    for user in job["hit"]["notes"]:
      if user in valid:
        for i in range(len(job["hit"]["notes"][user])):
          if str(transitions[i]) not in notes:
            notes[str(transitions[i])] = []
          for note in job["hit"]["notes"][user][str(i)]:
            notes[str(transitions[i])].append({"user": user, "note": note})

  ## Find the partial transitions to add to notes
  for job in j[file]["partial"]:
    # Extract users who did the task correctly
    valid = []
    for user in job["hit"]["submitted"]:
      if user["valid"] != "no":
        valid.append(user["name"])

    # convert partial sequence IDs to true IDs
    transitions = []
    for f,t in job["task"]["idxlist"]:
      start = convert(job["state"]["block_states"][f], states)
      end   = convert(job["state"]["block_states"][t], states)
      transitions.append([start,end])
    for user in job["hit"]["notes"]:
      if user in valid:
        for i in range(len(job["hit"]["notes"][user])):
          if str(transitions[i]) not in notes:
            notes[str(transitions[i])] = []
          for note in job["hit"]["notes"][user][str(i)]:
            notes[str(transitions[i])].append({"user": user, "note": note})

  ## Full (jump from first to last)
  for job in j[file]["full"]:
    # Extract users who did the task correctly
    valid = []
    for user in job["hit"]["submitted"]:
      if user["valid"] != "no":
        valid.append(user["name"])
    transition = "[0, %d]" % (len(states) - 1)
    for user in job["hit"]["notes"]:
      if user in valid:
        for i in range(len(job["hit"]["notes"][user])):
          if transition not in notes:
            notes[transition] = []
          for note in job["hit"]["notes"][user][str(i)]:
            notes[transition].append({"user": user, "note": note})

  JSON[file] = {}
  JSON[file]["states"] = states
  JSON[file]["annotations"] = notes
  JSON[file]["decoration"] = j[file]["simple"][0]["state"]["block_meta"]["decoration"]
  JSON[file]["block_properties"] = j[file]["simple"][0]["state"]["block_meta"]["blocks"][0]

# Get Images
for file in JSON:
  print "downloading images for ", file
  for idx in range(len(JSON[file]["states"])):
    # Get image and create normal name
    screencapid = JSON[file]["states"][idx]["screencapid"]
    cmd = "https://cwc-isi.org/api/screencap/%s" % (screencapid)
    r = requests.get(cmd)
    imagename = "%s_%s.png" % (file, str(idx) if idx > 9 else ('0' + str(idx)))
    print imagename
    fh = open("%s/%s" % (directory, imagename), "wb")
    fh.write(r.text.split(",",1)[1].decode('base64'))
    fh.close()
    del JSON[file]["states"][idx]["screencapid"]
    JSON[file]["states"][idx]["image"] = imagename

json.dump(JSON, gzip.open(directory + "/" + directory + ".json.gz", 'w'))
