from mpi4py import MPI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from difflib import SequenceMatcher
import os
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def plag(text):
    stop_words = set(stopwords.words("english"))
    remtext = [word for word in text.split() if word not in stop_words]
    textex = ' '.join(remtext)

    folder_name = "plag_files"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    input_file_path = os.path.join(folder_name, "_______input.txt")
    with open(input_file_path, "w") as text_file:
        text_file.write(textex)

    if rank == 0:
        student_files = [os.path.join(folder_name, doc) for doc in os.listdir(folder_name) if doc.endswith(".txt")]
        student_files = student_files[::-1]
    else:
        student_files = None

    student_files = comm.bcast(student_files, root=0)

    if rank == 0:
        file_content = [open(_file, encoding='utf8').read() for _file in student_files]
        merge = []
        file_content_updated = []
        for i in file_content:
            for j in i.split(' '):
                if j not in stop_words:
                    merge.append(j)
            txt = ' '.join(merge)
            file_content_updated.append(txt)
            merge = []
        vectors = vectorize(file_content_updated)
    else:
        vectors = None

    vectors = comm.bcast(vectors, root=0)

    s_vectors = list(zip(student_files, vectors))

    highest_score = 0.0
    highest_score_file = ""
    plagiarism_res = []

    for i, (student_a, text_vector_a) in enumerate(s_vectors):
        if rank != i % size:
            continue

        for j, (student_b, text_vector_b) in enumerate(s_vectors[i + 1:]):
            plagiarism_score = similarity(text_vector_a, text_vector_b)
            if plagiarism_score > highest_score:
                highest_score = plagiarism_score
                highest_score_file = student_b
            student_pair = sorted((student_a, student_b))
            score = (student_pair[0], student_pair[1], plagiarism_score)
            plagiarism_res.append(score)

    highest_score_file = comm.reduce(highest_score_file, op=MPI.MAX, root=0)
    plag_score = comm.reduce(highest_score, op=MPI.MAX, root=0)

    plagiarized_content = ""
    plagiarized_phrases = []
    if highest_score_file:
        if rank == 0:
            with open(highest_score_file, encoding='utf8') as file:
                plagiarized_content = file.read()

        matcher = SequenceMatcher(None, plagiarized_content, text)
        for match in matcher.get_matching_blocks():
            if match.size > 0:
                plagiarized_phrases.append(plagiarized_content[match.a: match.a + match.size])

    res_plag = [phrase for phrase in plagiarized_phrases]
    res_plag_str = " ".join(res_plag)

    if rank == 0:
        return plag_score, highest_score_file, res_plag_str
    else:
        return None


def vectorize(text):
    return TfidfVectorizer().fit_transform(text).toarray()


def similarity(doc1, doc2):
    return cosine_similarity([doc1], [doc2])[0][0]


if rank == 0:
    # Read the input text
    txt = """Bag of Words:
    The application of vector space retrieval, a traditional IR concept, to the field of content similarity detection is represented by bag of words analysis. For pairwise similarity calculations, documents are represented as one or more vectors, e.g., for various document portions. The standard cosine similarity measure or more advanced similarity measures may then be used in similarity computation.
    """
    start = time.time()
else:
    txt = None
    start = None

# Broadcast the start time and text to all processes
start = comm.bcast(start, root=0)
txt = comm.bcast(txt, root=0)

# Split the text into chunks for parallel processing
chunks = [txt[i:i + len(txt) // size] for i in range(0, len(txt), len(txt) // size)]

# Execute plagiarism function on each process with their respective text chunks
plag_result = plag(chunks[rank])

# Gather the plagiarism results on rank 0
plag_results = comm.gather(plag_result, root=0)

if rank == 0:
    all_results = [res for res in plag_results if res is not None]
    # Print the plagiarism results
    print("Plagiarized:\n", all_results)
    print("Time taken with parallel processing:", time.time() - start, "seconds")
