import os
import subprocess

def main():
	with open('server_installs.txt', 'r+') as f:
		for line in f:
			subprocess.call(line)
if __name__ == '__main__':
	main()