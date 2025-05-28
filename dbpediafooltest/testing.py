from SPARQLWrapper import SPARQLWrapper, JSON

def fetch_triples(entity_uri, endpoint="https://dbpedia.org/sparql", limit=100):
    sparql = SPARQLWrapper(endpoint)
    sparql.setReturnFormat(JSON)

    query = f"""
    SELECT ?s ?p ?o
    WHERE {{
      {{  VALUES ?s {{ <{entity_uri}> }}
         ?s ?p ?o
      }}
      UNION
      {{  VALUES ?o {{ <{entity_uri}> }}
         ?s ?p ?o
      }}
    }}
    LIMIT {limit}
    """
    sparql.setQuery(query)

    results = sparql.query().convert()
    triples = []
    for row in results["results"]["bindings"]:
        triples.append({
            's': row['s']['value'],
            'p': row['p']['value'],
            'o': row['o']['value']
        })
    return triples

def text_to_dbpedia_entity(label, endpoint="https://dbpedia.org/sparql"):
    # 1. Build the "page"/resource URI from the label
    underscored = label.replace(" ", "_")
    resource_uri = f"http://dbpedia.org/resource/{underscored}"

    # 2. SPARQL: ask if this resource redirects elsewhere
    sparql = SPARQLWrapper(endpoint)
    sparql.setReturnFormat(JSON)
    sparql.setQuery(f"""
      PREFIX dbo: <http://dbpedia.org/ontology/>
      SELECT ?redirect WHERE {{
        <{resource_uri}> dbo:wikiPageRedirects ?redirect .
      }}
      LIMIT 1
    """)

    try:
        results = sparql.query().convert()
        bindings = results["results"]["bindings"]
        if bindings:
            # return the first redirect target
            return bindings[0]["redirect"]["value"]
    except Exception as e:
        # could be endpoint errors, timeouts, etc.
        print(f"SPARQL query failed: {e}")

    # no redirect found (or SPARQL error) → return the original constructed URI
    return resource_uri

if __name__ == "__main__":
    #entity = "http://dbpedia.org/resource/Crown_Prince_Hyomyeong"
    #triples = fetch_triples(entity, limit=50)
    #for t in triples:
    #    print(f"{t['s']}  -- {t['p']}  -->  {t['o']}")
    #samples = ["Queen Hyojeong", "Heonjong of Joseon", "Crown Prince Hyomyeong"]
    samples = ["The Offspring", "The Beatles", "Queen Hyojeong", "Heonjong of Joseon", "Crown Prince Hyomyeong"]
    for name in samples:
        uri = text_to_dbpedia_entity(name)
        triples = fetch_triples(uri, limit=50)
        for t in triples:
            print(f"{t['s']}  -- {t['p']}  -->  {t['o']}")
        print(f"{name!r}  →  {uri}")