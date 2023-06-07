import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from difflib import SequenceMatcher
import sys
import time

def plag(text):
    print("Plag file ",text)
    stop_words = set(stopwords.words("english"))
    remtext = []
    for i in text.split(" "):
        if i not in stop_words:
            remtext.append(i)
    textex = ' '.join(remtext)

    folder_name = "plag_files"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    input_file_path = os.path.join(folder_name, "_______input.txt")
    with open(input_file_path, "w") as text_file:
        text_file.write(textex)

    student_files = [os.path.join(folder_name, doc) for doc in os.listdir(folder_name) if doc.endswith(".txt")]
    student_files = student_files[::-1]
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
    def vectorize(text):
        return TfidfVectorizer().fit_transform(text).toarray()

    def similarity(doc1, doc2):
        return cosine_similarity([doc1], [doc2])[0][0]

    vectors = vectorize(file_content_updated)
    s_vectors = list(zip(student_files, vectors))

    plagiarism_res = []

    def checkplagiarism():
        highest_score = 0.0
        highest_score_file = ""
        for student_a, text_vector_a in s_vectors:
            new_vectors = s_vectors.copy()
            current_index = new_vectors.index((student_a, text_vector_a))
            del new_vectors[current_index]
            for student_b, text_vector_b in new_vectors:
                plagiarism_score = similarity(text_vector_a, text_vector_b)
                if plagiarism_score > highest_score:
                    highest_score = plagiarism_score
                    highest_score_file = student_b
                student_pair = sorted((student_a, student_b))
                score = (student_pair[0], student_pair[1], plagiarism_score)
                plagiarism_res.append(score)
        return highest_score_file, plagiarism_res

    highest_score_file, plagiarism_res_list = checkplagiarism()
    result_list = []
    for i in plagiarism_res_list:
        if input_file_path in i:
            result_list.append(i[2])

    plagiarized_content = ""
    plagiarized_phrases = []
    if highest_score_file:
        with open(highest_score_file, encoding='utf8') as file:
            plagiarized_content = file.read()

        matcher = SequenceMatcher(None, plagiarized_content, text)
        for match in matcher.get_matching_blocks():
            if match.size > 0:
                plagiarized_phrases.append(plagiarized_content[match.a: match.a + match.size])
    res_plag = []

    for phrase in plagiarized_phrases:
        res_plag.append(phrase)
    res_plag_str = " ".join(res_plag)
    plag_score = max(result_list)
    rtrnval = []
    
    rtrnval.append(plag_score)
    rtrnval.append(highest_score_file)
    rtrnval.append(res_plag_str)
    return rtrnval

txt = """Bag of Words:
The application of vector space retrieval, a traditional IR concept, to the field of content similarity detection is represented by bag of words analysis. For pairwise similarity calculations, documents are represented as one or more vectors, e.g., for various document portions. The standard cosine similarity measure or more advanced similarity measures may then be used in similarity computation.
"""
start=time.time()
print("Plagiarized: \n",plag(txt))
print("time taken without parallel processing: ", time.time()-start," seconds")




