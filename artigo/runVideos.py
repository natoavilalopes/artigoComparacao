import os
import detectOpencvLogs
import time


videos = 'videos'
logs = 'logs'



nome = "relatorio"
 
txt = f'logs/{nome}.txt'
with open(txt, 'w') as f:
	f.write('Relatório geral\n')

file_object = open(txt, 'a')    













accEsta = 0
accNaoEsta = 0

t1 = time.time()

lista = []
for fn in os.listdir(logs):
    g = os.path.join(logs, fn)
    lista.append(g[5:-8])



for filename in os.listdir(videos):
    f = os.path.join(videos, filename)
    #print(f)

    if f[10:-4] not in lista:
        accNaoEsta += 1
        if os.path.isfile(f):
            detectOpencvLogsVideos.run(f)
            #print(f)
    else:
        accEsta += 1
        
        
 
t2 = time.time()

file_object.write("Tempo total: " + str(t2 - t1))

print("Tempo total: " + str(t2 - t1))       
