from string import Template


extract_relation_prompt = Template('''Please retrieve up to $relation_num relations that contribute to the question. Rate each selected relation from 0 to 1; higher means more relevant. Choose only from the numbered candidates and consider their direction. Return a JSON object with a "scores" list containing the candidate "id" and "score".

Example:
Question: In which country is Hokkaido located?
Topic Entity: Hokkaido
Relations:
[1] [outgoing] location.administrative_division.country
[2] [incoming] location.location.contains
Output: {"scores": [{"id": 1, "score": 0.8}]}

Question: $question
Topic Entity: $entity_name
${path_context}\
Relations:
$relations
Output:''')


entity_p_prompt = Template('''Based on the question and path history, select up to $top_k relevant entities from the numbered candidates. Different numbers identify different entities, even when their names are identical. Return a JSON object with a "selected_ids" list in descending order of relevance.

Example:
Question: Which movie featured Miley Cyrus and was produced by Tobin Armbrust?
Current Entity: Tobin Armbrust
Current Relation: film.producer.film
Candidate Entities:
[1] The Resident
[2] So Undercover
Output: {"selected_ids": [2]}

Question: $question
Current Entity: $current_entities
Current Relation: $current_relation
Path History: $path_history
Candidate Entities:
$candidates
Output:''')


EVALUATE_STATE_PROMPT = Template('''Evaluate how useful the knowledge graph reasoning path is for answering the question. Consider relevance, whether it narrows down the answer, and whether the facts support the required reasoning. Assign a score from 0 to 1: 0.0-0.3 is irrelevant, 0.4-0.6 is partly relevant, 0.7-0.8 contributes substantially, and 0.9-1.0 directly resolves the question or provides a critical fact. Return a JSON object with numeric "score" and string "reason".

Example:
Question: Who wrote Pride and Prejudice?
Path: Pride and Prejudice, published.year, 1813
Output: {"score": 0.4, "reason": "The publication year does not establish authorship."}

Question: $question
Path: $triple
Output:''')


answer_directly_prompt = Template('''Answer the question using your knowledge. Return a JSON object with string "reason" and "answer" as a string or null when no answer can be determined.

Question: $question
Output:''')


answer_prompt = Template('''Answer the question using the retrieved knowledge graph triplets and your knowledge. Return a JSON object with string "reason" and "answer" as a string or null when no answer can be determined.

Example:
Question: The artist nominated for The Long Winter lived where?
Knowledge Triplets: The Long Winter, book.written_work.author, Laura Ingalls Wilder; Laura Ingalls Wilder, people.person.places_lived, Unknown; Unknown, people.place_lived.location, De Smet
Output: {"reason": "The path links the author to De Smet.", "answer": "De Smet"}

Question: $question
Knowledge Triplets: $knowledge_triplets
Output:''')
