# print the requirements.txt for the pip 
# command
# #%%
import subprocess

with open("requirements.txt", "w") as f:
    subprocess.run(["pip", "freeze"], text=True, stdout=f)
