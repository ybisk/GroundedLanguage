python DataCreation/getTurkAnnotationJSON.py trainset/digits/
mv out.json.gz trainset/digits.json.gz

python DataCreation/AddSceneToTurk.py ../digitstates/MTurk/Train trainset/digits.json.gz
mv data.json.gz trainset/digits.worlds.json.gz
