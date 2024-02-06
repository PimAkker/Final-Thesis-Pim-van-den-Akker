import subprocess
import time

def run_blender():
    while True:
        print("Starting Blender...")
        process = subprocess.Popen([r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe", r"C:\Users\pimde\OneDrive\thesis\Blender\random room test 3 .blend"])
        exit_code = process.wait()
        if exit_code == 0:
            print("Blender exited normally.")
            break
        else:
            print("Blender crashed. Restarting in 5 seconds...")
            time.sleep(5)

run_blender()