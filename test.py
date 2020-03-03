matches = [1, 1, 1, 0, 0]

if True in matches:
    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
    print(matchedIdxs)

matches = [0, 0, 0]
if True not in matches:
    print(121)