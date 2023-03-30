from flask import Flask, jsonify, request
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.classify.textcat import TextCat
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx


def read_article(text):
    
    article = text.split(". ")
    sentences = []

    for sentence in article:
        print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop() 
    
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def generate_summary(text, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text anc split it
    sentences =  read_article()

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    print("Indexes of top ranked_sentence order are ", ranked_sentence)    

    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[i][1]))

    # Step 5 - Offcourse, output the summarize texr
    return (summarize_text)


app = Flask(__name__)


@app.route('/api', methods = ['GET'])
def summ():
    #Query = str(request.args['Query'])
    text = """‘Love the art in yourself’, with these golden words, Alankar Theatre, Chandigarh aims to bring social awareness to the masses. With drama as its medium, Alankar Theatre has been dynamically active both on stage as well as on the streets.
It is an organization that furnishes a platform for individuals who are passionate to work toward social reform. Using theatre as a tool, the fundamental aim of the Alankar Theatre is to fan the flames of social rectification. The group comprises of budding actors and keen directors who are passionate about their work. The associated youth get the scope to enhance their knowledge and hone the abilities that further give them the pedestal to cater to the civil needs of the numerous communities present. As a consequence, a healthy relationship among people worldwide is established through the bodywork of educational theatre.
Founded by Mr. Chakresh Kumar in the year 2005, Alankar Theatre acquired registration in Chandigarh in 2012. Alankar theatre has fabricated several plays. These include ‘MautKyuNahiAati Raat Bhar’, ‘Macbeth’, ‘One or The Part of Two’, ‘Thought’, ‘Ji AayaSahab’, ‘AndhaYug’, ‘Ram SajeevankiPrem Katha’, ‘Parsai Ki Duniya’, ‘Natak Ka Naam Kyu Rakhein’, ‘Mandir’, ‘EkLadki Paanch Deewane’, ‘PhateGulabJamun’, ‘The First Teacher’ and award-winning play ‘Poster’, ‘Bolti Deewaren’, A Mid-Summer Night’s Dream, ‘Challa Ho Gaya’, ‘Inna Ki Awaj, Thiru Nangai, Bharatdurdasha, Meera Bai, Azaadi, Romeo and Juliet…etc. Alankar Theatre has staged these plays in diverse places of India, those being- Patna, Lucknow, Patiala, Chandigarh, Amritsar, Bhatinda, Hissar, Delhi, Ambala, Uttrakhand, Kerala, Jabalpur, Allahabad, Indore, etc. Alankar Theatre is also involved in cinema by constantly producing various short films and providing its actors to grow in the field of the camera as well."""
    return generate_summary(text, 4)


if __name__=="__main__":
    app.run()
 

