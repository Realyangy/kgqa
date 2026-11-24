"""Bi-encoder pair scoring for online KG candidate reranking."""


class BiEncoderPairScorer:
    """Expose SentenceTransformer dot-product scores through ``predict``."""

    def __init__(self, model_path, model=None):
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_path)
        self.model = model

    @staticmethod
    def _dot(left, right):
        return float(sum(float(x) * float(y) for x, y in zip(left, right)))

    def predict(self, pairs):
        pairs = list(pairs or [])
        if not pairs:
            return []

        queries = []
        query_indices = {}
        documents = []
        pair_query_indices = []
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("Each reranking pair must contain one query and one document.")
            query, document = str(pair[0]), str(pair[1])
            if query not in query_indices:
                query_indices[query] = len(queries)
                queries.append(query)
            pair_query_indices.append(query_indices[query])
            documents.append(document)

        encode_kwargs = {
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }
        query_embeddings = self.model.encode(queries, **encode_kwargs)
        document_embeddings = self.model.encode(documents, **encode_kwargs)
        return [
            self._dot(query_embeddings[query_index], document_embedding)
            for query_index, document_embedding in zip(
                pair_query_indices,
                document_embeddings,
            )
        ]
