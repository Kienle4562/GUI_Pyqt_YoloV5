from shlex import join

lpll = ['95B1-54664', '65M1-03032']
lpll = '_'.join(map(str, lpll))
pathDestination = ['1s', '2s']
pathDestination = lpll.replace("-","") +"_"+ pathDestination[1]
print(pathDestination)