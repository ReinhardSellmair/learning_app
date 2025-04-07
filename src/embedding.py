
# embed the wiki docs
def embed_wiki_docs(wiki_docs, embedding_model):
    wiki_embeddings = {}
    for section, text in wiki_docs.items():
        wiki_embeddings[section] = embedding_model.encode(section + ': ' + text)
    return wiki_embeddings