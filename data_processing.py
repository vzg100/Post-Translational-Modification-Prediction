import urllib.request
from time import sleep
class scrape:
    def __init__(self, file, output):
        f = open(file)
        output = open(output, "w+")
        for i in f:
            sleep(2)
            i = i.split()
            id = i[1]
            aa = i[7]
            pos = i[2]
            try:
                page = urllib.request.urlopen("http://www.uniprot.org/uniprot/" + id + ".fasta")
                sequence = str(page.read())
                t = 0
                for j in sequence:
                    if j == "\\":
                        break
                    t += 1
                sequence = sequence[t:]
                temp = ''
                for i2 in sequence:
                    if i2 not in "\\n'":
                        temp += i2
                sequence = temp
                output.write(str(id)+","+str(sequence)+","+str(pos)+","+str(aa)+"\n")
                print(id)
            except:
                print("meh")
qq = [["Phosphorylation/Phosphorylation_T.txt", "t_site.txt"],["Phosphorylation/Phosphorylation_Y.txt", "y_site.txt"]]
gg = [["N-linked/N-linked_N.txt","n_site.txt"],["Acetylation/Acetylation_K.txt","k_site.txt"]]
for i in qq:
    scrape(i[0], i[1])


