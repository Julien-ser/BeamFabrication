from SPARQLWrapper import SPARQLWrapper, JSON
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Initialize SPARQL endpoint and embedding model once
SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def sparql_query(query: str):
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    sparql.setReturnFormat(JSON)
    sparql.setQuery(query)
    return sparql.query().convert()

def get_abstract(uri: str) -> str:
    """Fetch the English dbo:abstract for a resource (or empty string)."""
    q = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    SELECT ?abs WHERE {{
      <{uri}> dbo:abstract ?abs .
      FILTER(lang(?abs)="en")
    }} LIMIT 1
    """
    res = sparql_query(q)
    bindings = res["results"]["bindings"]
    return bindings[0]["abs"]["value"] if bindings else ""

def get_wikilinks(uri: str, limit: int = 100) -> list:
    """Fetch up to `limit` dbo:wikiPageWikiLink targets for `uri`."""
    q = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    SELECT DISTINCT ?linked WHERE {{
      <{uri}> dbo:wikiPageWikiLink ?linked .
    }} LIMIT {limit}
    """
    res = sparql_query(q)
    return [b["linked"]["value"] for b in res["results"]["bindings"]]

def most_similar_neighbor(entity_uri: str, link_limit: int = 100) -> str:
    # 1. Get abstract of the entity
    base_abs = get_abstract(entity_uri)
    if not base_abs:
        raise ValueError(f"No English abstract found for {entity_uri}")

    # 2. Get linked entities and their abstracts
    neighbors = get_wikilinks(entity_uri, limit=link_limit)
    abstracts = []
    uris = []
    for u in neighbors:
        if u == entity_uri:
            continue
        abs_text = get_abstract(u)
        if abs_text:
            uris.append(u)
            abstracts.append(abs_text)

    if not abstracts:
        raise ValueError("No linked entities with abstracts found.")

    # 3. Embed all abstracts in one go
    #    First item is the base entity
    all_texts = [base_abs] + abstracts
    embeddings = EMBED_MODEL.encode(all_texts, convert_to_tensor=True)

    base_emb = embeddings[0]
    neighbor_embs = embeddings[1:]

    # 4. Compute cosine similarities
    sims = util.cos_sim(base_emb, neighbor_embs)[0]  # vector of shape [len(neighbors)]

    # 5. Identify the highest-scoring neighbor
    best_idx = int(np.argmax(sims.cpu().numpy()))
    #best_idx = int(np.argmax(sims))
    best_uri = uris[best_idx]
    best_score = float(sims[best_idx])

    #print(f"Most similar neighbor to {entity_uri}:")
    #print(f"  → {best_uri}   (cosine similarity = {best_score:.4f})")
    return best_uri

if __name__ == "__main__":
    entity = "http://dbpedia.org/resource/Heonjong_of_Joseon"
    most_similar = most_similar_neighbor(entity, link_limit=100)