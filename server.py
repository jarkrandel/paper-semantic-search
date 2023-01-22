from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import sys

app = Flask(__name__)

# If 'debug' is passed as the first command line argument, use the smaller 'debug' datasets for testing
prefix = "data/total/"
if len(sys.argv) > 1 and sys.argv[1] == 'debug':
    prefix = "data/debug/debug-"

# load arxiv dataset, sentence transformer model for sentence embeddings,
# and stored vector embeddings of paper titles. We use the assymmetric
# msmarco model for abstracts because we expect the query to be much shorter...
title_embs = np.load(prefix + 'title-embeddings.npy')
abstract_embs = np.load(prefix + 'abstract-embeddings-msmarco.npy')
df = pd.read_json(prefix + 'arxiv-metadata.json', lines=True)
title_model = SentenceTransformer('all-MiniLM-L6-v2')
abstract_model = SentenceTransformer('msmarco-MiniLM-L-12-v3')

# all-MiniLM-L6-v2 and msmarco... embeddings are 384 dimensional.
title_index = faiss.IndexFlatL2(384)
title_index.add(title_embs)
abstract_index = faiss.IndexFlatIP(384)
abstract_index.add(abstract_embs)

# Define how many neighbors to search for.
k = 20

@app.route('/', methods=['POST'])
def handle_post_request():
    # Upon form submission, read the string in the search box into query.
    query = request.form['q']
    # Read the search type given in the drop-down box
    search_type = request.form['search-type']
    
    # Embed query using model then search and retrieve indices: [[i1, i2, ...]]
    # and the distances to those indices: [[d1, d2, ...]].
    if search_type == 'title':
        query_embedding = title_model.encode(query).reshape(1, -1)
        D, I = title_index.search(query_embedding, k)
    elif search_type == 'abstract':
        query_embedding = abstract_model.encode(query).reshape(1, -1)
        D, I = abstract_index.search(query_embedding, k)
    I = I[0]

    # Extract relevant entries from total arxiv data.
    neighbors = df.iloc[I]
    dates = neighbors['update_date'].tolist()
    sorted(I, key=lambda x: -int(df.iloc[x]['update_date'][0:4]))
    neighbors = df.iloc[I]

    ids = neighbors['id'].tolist()
    authors = neighbors['authors'].tolist()
    titles = neighbors['title'].tolist()
    abstracts = neighbors['abstract'].tolist()
    categories = neighbors['categories'].tolist()

    return render_template('index.html', 
                            search_type=search_type,
                            indices=I,
                            ids=ids,
                            authors=authors, 
                            titles=titles, 
                            categories=categories,
                            abstracts=abstracts,)

@app.route('/', methods=['GET'])
def handle_get_request():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
