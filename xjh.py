import re
import sys
input = sys.stdin.readline().strip()
find = re.fullmatch("[ab]|([ab])[ab]*\\1", input)
print(find)