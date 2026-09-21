import sys
msg = sys.stdin.read()
msg = '
'.join([line for line in msg.split('
') if 'Co-Authored-By: Claude' not in line])
sys.stdout.write(msg)
