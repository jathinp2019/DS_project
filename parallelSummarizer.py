import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from transformers import pipeline
from string import punctuation
from heapq import nlargest
from transformers import PegasusForConditionalGeneration, AutoTokenizer
import torch
import re
from mpi4py import MPI
import time
punctuation = punctuation + '\n'
def sum(text):
    nlp = spacy.load('en_core_web_sm')
    stopwords = list(STOP_WORDS)
    doc = nlp(text)
    tokens = [token.text for token in doc]
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in stopwords:
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1               
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word]/max_frequency
    sentence_tokens = [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]
    select_length = int(len(sentence_tokens)*0.3)
    summary = nlargest(select_length, sentence_scores, key = sentence_scores.get)
    final_summary = [word.text for word in summary]
    summary = ' '.join(final_summary)
    return(summary)
    
def pegtor(text):
    model_name = 'google/pegasus-cnn_dailymail'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    batch = tokenizer(text, truncation=True, padding='longest', return_tensors="pt").to(device)
    translated = model.generate(**batch)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return(tgt_text[0])


def piptrans(text):
    summarizer = pipeline("summarization")
    t3=summarizer(text, max_length=130, min_length=100, do_sample=False)
    return t3[0]['summary_text']


txt="""A computer is a machine that can be programmed to carry out sequences of arithmetic or logical operations (computation) automatically. Modern digital electronic computers can perform generic sets of operations known as programs. These programs enable computers to perform a wide range of tasks. A computer system is a nominally complete computer that includes the hardware, operating system (main software), and peripheral equipment needed and used for full operation. This term may also refer to a group of computers that are linked and function together, such as a computer network or computer cluster.

A broad range of industrial and consumer products use computers as control systems. Simple special-purpose devices like microwave ovens and remote controls are included, as are factory devices like industrial robots and computer-aided design, as well as general-purpose devices like personal computers and mobile devices like smartphones. Computers power the Internet, which links billions of other computers and users.

Early computers were meant to be used only for calculations. Simple manual instruments like the abacus have aided people in doing calculations since ancient times. Early in the Industrial Revolution, some mechanical devices were built to automate long, tedious tasks, such as guiding patterns for looms. More sophisticated electrical machines did specialized analog calculations in the early 20th century. The first digital electronic calculating machines were developed during World War II. The first semiconductor transistors in the late 1940s were followed by the silicon-based MOSFET (MOS transistor) and monolithic integrated circuit chip technologies in the late 1950s, leading to the microprocessor and the microcomputer revolution in the 1970s. The speed, power and versatility of computers have been increasing dramatically ever since then, with transistor counts increasing at a rapid pace (as predicted by Moore's law), leading to the Digital Revolution during the late 20th to early 21st centuries.

Conventionally, a modern computer consists of at least one processing element, typically a central processing unit (CPU) in the form of a microprocessor, along with some type of computer memory, typically semiconductor memory chips. The processing element carries out arithmetic and logical operations, and a sequencing and control unit can change the order of operations in response to stored information. Peripheral devices include input devices (keyboards, mice, joystick, etc.), output devices (monitor screens, printers, etc.), and input/output devices that perform both functions (e.g., the 2000s-era touchscreen). Peripheral devices allow information to be retrieved from an external source and they enable the result of operations to be saved and retrieved."""

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
start=time.time()
txt=txt+"\n\n"

if rank == 0:
    l=[]
    x=0
    for i in range(len(txt)-1):
        if txt[i] == '\n':
            if txt[i+1] == '\n':
                l.append(txt[x:i-1])
            x=i+2
    for i in l:
        if i == "":
            l.pop(l.index(i))

    c=len(l)
    comm.send(l[0:c//3],dest=1)
    comm.send(l[(c//3) :2*c//3],dest=2)
    comm.send(l[2*c//3:c],dest=3)

    p=comm.recv(source=1)
    q=comm.recv(source=2)
    r=comm.recv(source=3)
    fin=(p+"\n"+q+"\n"+r)
    print("paraphrased text: ")
    print (fin.replace("<n>", ""))
    print("time taken using parallel processing: ", time.time()-start," seconds")
if rank == 1:
    a=comm.recv(source=0)
    res=""
    for i in a:
        res=res+sum(i)
    comm.send(res, dest=0)
elif rank == 2:
    a=comm.recv(source=0)
    res=""
    for i in a:
        res=res+pegtor(i)
    comm.send(res, dest=0)
elif rank == 3:
    a=comm.recv(source=0)
    res=""
    for i in a:
        res=res+piptrans(i)
    comm.send(res, dest=0)

