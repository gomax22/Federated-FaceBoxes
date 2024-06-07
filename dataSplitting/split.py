import os
import random
import sys
import math
import zipfile
import shutil
import time

class SplitData:

    def __init__(self, annotationsPath,trainSetPath, nClients):
        self.annotations = []
        self.trainSet = []
        self.nClients = nClients
        self.populate_annotations(annotationsPath)
        self.populate_trainSet(trainSetPath)
    
    def populate_annotations(self, directory):
        for filename in os.listdir(directory):
            self.annotations.append(filename)
        random.shuffle(self.annotations)


    def populate_trainSet(self,directory):
        for subdir in os.listdir(directory):
            subdir_path = os.path.join(directory, subdir)
            if os.path.isdir(subdir_path):
                # Per ogni file all'interno della subdirectory
                for filename in os.listdir(subdir_path):
                    self.trainSet.append(tuple((subdir,filename)))
                random.shuffle(self.trainSet)

        
#QUESTA SI PUò VELOCIZZARE con i threads.         
    def split(self,dataDirectory):
        #########PER TESTARE ,ALTRIMENTI TOGLIERE IL /100 #######################
        dataChunk = math.ceil( (len(self.trainSet)/100) / self.nClients)
        print("dataChunk=", dataChunk)
        for index in range(0, self.nClients):
            #creo zip vuoto
            zf = zipfile.ZipFile(f"WIDER_train{index+1}.zip", "w")
            startIndex = index*dataChunk
            endIndex = min(startIndex + dataChunk-1, len(self.trainSet))
            file_name = f"img_list.txt"
            with open(file_name, 'a') as file:
                # Ogni client crea la sua porzione di annotazioni 
                for imgName in self.trainSet[startIndex:endIndex]:
                    annotation = self.findAnnotations(imgName[1])
                    file.write(f"{imgName[0]+'/'+imgName[1]} {annotation}\n")
                    #Creazione directory root
                    try:
                        os.makedirs(f"WIDER_train/images/{imgName[0]}",exist_ok=False)
                        os.makedirs("annotations/",exist_ok=False)
                    except FileExistsError:
                        pass
                    # Copio il file assegnato al thread nella sua directory che andrò a zippare.
                    shutil.copy2(f"{dataDirectory}/{imgName[0]}/{imgName[1]}", f"WIDER_train/images/{imgName[0]}")
                    # Copio l'annotation assegnata.
                    shutil.copy2(f"data/annotations/{annotation}", "annotations/")
            #Copio il file img_list del thread nella sua directory WIDER_train
            shutil.copy2(file_name, "WIDER_train/")
            # Aggiungo al zip la cartella WIDER_train
            for dirname, subdirs, files in os.walk(f"WIDER_train"):
                zf.write(dirname)
                for filename in files:
                    zf.write(os.path.join(dirname, filename))
            # Aggiungo al zip la cartella annotations
            for dirname, subdirs, files in os.walk(f"annotations"):
                zf.write(dirname)
                for filename in files:
                    zf.write(os.path.join(dirname, filename))
            # Rimuovo il file img_list.txt dalla directory principale.
            os.remove(file_name)
            shutil.rmtree(f"WIDER_train", ignore_errors=True)
            shutil.rmtree(f"annotations", ignore_errors=True)
            zf.close()
           # Passo alla generazione del prossimo client
                 
    def findAnnotations(self, fileName):
        for annotation in self.annotations:
            if os.path.splitext(os.path.basename(fileName))[0] == os.path.splitext(os.path.basename(annotation))[0]:
                return annotation

if __name__ == "__main__":
    if len(sys.argv) > 1:
            nClients = int(sys.argv[1])
    else:
        nClients = int(input("nClients = "))
    
    start_time = time.time()
    splitter = SplitData("data/annotations","data/WIDER_train/images",nClients)
    splitter.split("data/WIDER_train/images")
    end_time = time.time()
    duration = end_time - start_time
    print(f"Tempo di completamento dell'operazione: {duration} secondi")
