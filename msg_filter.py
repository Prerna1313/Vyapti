import sys
msg = sys.stdin.read()
msg = '\n'.join([line for line in msg.split('\n') if 'Co-Authored-By: Claude' not in line])
sys.stdout.write(msg)
