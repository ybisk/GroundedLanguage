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


class transObj():
  def __init__(self, tran = [0,1], typ = "A0"):
    self.type = typ
    self.start = tran[0]
    self.finish = tran[1]
    self.users = []
    self.notes = []

  def add(self, u, n):
    self.users.append(u)
    self.notes.append(n)

  def __eq__(self, other):
    return self.start == other.start and self.finish == other.finish and self.type == other.type

  def jsonable(self):
    return self.__dict__

class Task:
  def __init__(self, name):
    self.filename = name
    self.side_length = 0.1524
    self.shape_params = ["blue", "green", "cyan", "magenta", "yellow", "red"]

  def add_states(self, given):
    """
    Convert to 3D array
    """
    self.states = []
    self.images = []
    for state in given:
      current = []
      for position in state["block_state"]:
        current.append([position["position"]["x"], position["position"]["y"], position["position"]["z"]])
      self.states.append(current)
      self.images.append(state["screencapid"])

  def add_decoration(self, decor):
    self.decoration = decor

  def add_notes(self, notes):
    self.notes = notes

  def jsonable(self):
    return self.__dict__

# http://stackoverflow.com/questions/5160077/encoding-nested-python-object-in-json
def ComplexHandler(Obj):
  if hasattr(Obj, 'jsonable'):
    return Obj.jsonable()
  else:
    raise TypeError, 'Object of type %s with value of %s is not JSON serializable' % (type(Obj), repr(Obj))


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
JSON = []
for file in j:
  ## Read information about the Simple state
  states = j[file]["simple"][0]["state"]["block_states"]
  for state in states:
    del state["created"]
  notes = []
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
          trans = transObj(transitions[i], "A0")
          if trans not in notes:
            notes.append(trans)
          idx = notes.index(trans)
          for note in job["hit"]["notes"][user][str(i)]:
            notes[idx].add(user, note)

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
          trans = transObj(transitions[i], "A1")
          if trans not in notes:
            notes.append(trans)
          idx = notes.index(trans)
          for note in job["hit"]["notes"][user][str(i)]:
            notes[idx].add(user, note)

  ## Full (jump from first to last)
  for job in j[file]["full"]:
    # Extract users who did the task correctly
    valid = []
    for user in job["hit"]["submitted"]:
      if user["valid"] != "no":
        valid.append(user["name"])
    transitions = []
    for f,t in job["task"]["idxlist"]:
      start = convert(job["state"]["block_states"][f], states)
      end   = convert(job["state"]["block_states"][t], states)
      transitions.append([start,end])
    for user in job["hit"]["notes"]:
      if user in valid:
        for i in range(len(job["hit"]["notes"][user])):
          trans = transObj(transitions[i], "A2")
          if trans not in notes:
            notes.append(trans)
          idx = notes.index(trans)
          for note in job["hit"]["notes"][user][str(i)]:
            notes[idx].add(user, note)

  task = Task(file)
  task.add_states(states)
  task.add_notes(notes)
  task.add_decoration(j[file]["simple"][0]["state"]["block_meta"]["decoration"])
  #task.block_properties = j[file]["simple"][0]["state"]["block_meta"]["blocks"][0]
  JSON.append(task)

# Get Images
for file in JSON:
  print "downloading images for ", file.filename
  for idx in range(len(file.states)):
    # Get image and create normal name
    screencapid = file.images[idx]
    cmd = "https://cwc-isi.org/api/screencap/%s" % (screencapid)
    r = requests.get(cmd)
    imagename = "%s_%s.png" % (file.filename, str(idx) if idx > 9 else ('0' + str(idx)))
    print imagename
    fh = open("%s/%s" % (directory, imagename), "wb")
    fh.write(r.text.split(",",1)[1].decode('base64'))
    fh.close()
    file.images[idx] = imagename


out = gzip.open(directory + "/" + directory + ".json.gz", 'w')
for file in JSON:
  out.write(json.dumps(file, default=ComplexHandler)+ "\n")
out.close()
