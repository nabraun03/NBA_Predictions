
scripts = ['apirequests.py', 'preprocessing.py', 'currentprofiles.py', 'predictions.py']


for script in scripts:
    print(script)
    with open(script, "r") as file:
        exec(file.read())

