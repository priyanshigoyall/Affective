import nltk
import io
import calendar
import time
import md5

file = io.open('WhatsApp Chat with Fam-bam.txt', encoding = 'utf-8')

m = md5.new()
for line in file.readlines():
    line = line.split('-')
    m = md5.new()
    name_message = line[len(line)-1].split(":")
    print(list(name_message))
    name = name_message[0]
    m.update(name)
    name = m.digest()
    print name

    #print(name.encode('utf-8'))

    #print(line[len(line)-2])