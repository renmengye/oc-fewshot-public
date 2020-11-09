import numpy as np

with open('all.txt', 'r') as f:
  all_env = f.readlines()

all_env = map(lambda x: x.strip('\n'), sorted(all_env))
print(all_env)
rnd = np.random.RandomState(10)
rnd.shuffle(all_env)

NTRAIN = 60
NVAL = 10
NTEST = 20

train = all_env[:NTRAIN]
val = all_env[NTRAIN:NTRAIN + NVAL]
test = all_env[NTRAIN + NVAL:]

with open('train.txt', 'w') as f:
  for line in train:
    f.write(line + '\n')
with open('val.txt', 'w') as f:
  for line in val:
    f.write(line + '\n')
with open('test.txt', 'w') as f:
  for line in test:
    f.write(line + '\n')
