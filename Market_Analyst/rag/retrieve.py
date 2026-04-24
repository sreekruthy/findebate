def retrieve_filtered(query, company, data_type=None, k=3):
    query_embedding = model.encode(query).tolist()

    if data_type:
        where_filter = {
            "$and": [
                {"company": company},
                {"type": data_type}
            ]
        }
    else:
        where_filter = {"company": company}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where_filter
    )

    return results