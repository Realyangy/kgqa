from string import Template

relation_extract_prompt = Template('''Please retrieve $relation_num relations that contribute to the question and rate their contribution on a scale from 0 to 1. Choose only from the numbered relations (including their direction). Return valid JSON with a "scores" list; each item must contain "id" and "score" and may contain a "reason".
Q: Name the president of the country whose main spoken language was Brahui in 1980?
Topic Entity: Brahui Language
Relations:
[1] language.human_language.main_country
[2] language.human_language.language_family
[3] language.human_language.iso_639_3_code
[4] base.rosetta.languoid.parent
[5] language.human_language.writing_system
[6] base.rosetta.languoid.languoid_class
[7] language.human_language.countries_spoken_in
[8] kg.object_profile.prominent_type
[9] base.rosetta.languoid.document
[10] base.ontologies.ontology_instance.equivalent_instances
[11] base.rosetta.languoid.local_name
[12] language.human_language.region
A: {"scores":[{"id":1,"score":0.4,"reason":"This relation is highly relevant as it directly relates to the country whose president is being asked for, and the main country where Brahui language is spoken in 1980."},{"id":7,"score":0.3,"reason":"This relation is also relevant as it provides information on the countries where Brahui language is spoken, which could help narrow down the search for the president."},{"id":4,"score":0.2,"reason":"This relation is less relevant but still provides some context on the language family to which Brahui belongs, which could be useful in understanding the linguistic and cultural background of the country in question."}]}

Q: $question
Topic Entity: $entity_name
${path_context}Relations:
$relations
A: ''')


entity_prune_prompt = Template('''Based on the question and path history, filter the most relevant $top_k entities from the candidate entities.
Question: $question
Current Entity: $current_entities
Current Relation: $current_relation
Path History: $path_history
Candidate Entities: $candidates

Please output a valid JSON object with a "selected_ids" list of the numbered candidate IDs, in descending order of relevance. Different IDs remain distinct even when names are identical:''')


evaluate_prompt = Template('''Your task is to rigorously evaluate whether the selected triplet from the knowledge graph is useful for reasoning toward answering the given question. Follow these steps carefully:

EVALUATION CRITERIA:
1. Does the triplet directly or indirectly mention entities or relationships from the question? If indirect, is the connection clear and meaningful?
2. Does the triplet provide specific information that helps narrow down or answer the question? Is the information sufficient to make progress toward the answer, or is it too vague or tangential?
3. Do the triplets logically necessary or strongly supportive for constructing a reasoning path to the answer?Does it fill a critical gap in the reasoning process?

SCORING GUIDELINES:
0.0-0.3: The triplet is irrelevant or provides no meaningful contribution to answering the question.
0.4-0.6: The triplet is somewhat relevant but only loosely connected or provides minimal information.
0.7-0.8: The triplet is relevant and contributes to the reasoning process but is not decisive or critical.
0.9-1.0: The triplet provids a clear, unambiguous, and direct answer to the question. It must fully resolve the question or provide a critical piece of information that leaves no room for ambiguity or further reasoning.

Important: The presence of entities from the question alone does not guarantee relevance. The triplet must actively help resolve the question, not just include related triplets' entities.

OUTPUT FORMAT:
Provide a valid JSON object with a numeric "score" between 0.0 and 1.0 with one decimal place and a string "reason" containing a concise explanation justifying the score based on the evaluation criteria.

Q: The artist nominated for The Long Winter lived where?
T: Laura Ingalls Wilder -> people.person.places_lived -> Unknown
A: {"score":0.5,"reason":"The triplet connects the author (key entity) to places_lived, but the Unknown value makes it incomplete. While it indicates the KG has author residency data, the lack of concrete location requires supplementing with external knowledge."}

Q: What is the capital of France?
T: France -> has.city -> Lyon,France -> has.city -> Paris
A: {"score":0.4,"reason":"Although both triplets mention French cities, only one (Paris) is relevant to the question. The inclusion of Lyon as a distractor reduces the overall relevance, as it requires additional filtering to identify the correct answer."}

Q: Rift Valley Province is located in a nation that uses which form of currency?
T: Rift Valley Province -> location.administrative_division.country -> Kenya
A: {"score":1,"reason":"This triplet directly resolves the question's core requirement by establishing the country relationship. Combined with common knowledge that Kenya uses the Kenyan shilling, it provides the critical link for answering the currency question."}

Q: Who wrote Pride and Prejudice?
T: Pride and Prejudice -> published.year -> 1813
A: {"score":0.4,"reason":"While the publication year is factual, it doesn't directly address authorship. The temporal context could potentially support era-based reasoning, but this requires substantial external knowledge bridging."}

Q: Which German philosopher wrote The Critique of Pure Reason?
T: Immanuel Kant -> people.person.profession -> Philosopher,The Critique of Pure Reason -> published.author -> Immanuel Kant
A: {"score":0.8,"reason":"The first triplet confirms Kant's profession as a philosopher, and the second directly links the work to Kant. Together, they provide strong evidence to answer the question, though the nationality aspect is still missing."}

Q: What is the capital of France?
T: France -> has.city -> Lyon,France -> has.city -> Marseille
A: {"score":0.2,"reason":"Both triplets mention French cities, but neither is the capital. The inclusion of multiple distractors further reduces relevance, as it requires additional filtering to identify the correct answer."}

Q: Which German philosopher wrote The Critique of Pure Reason?
T: Immanuel Kant -> people.person.nationality -> German
A: {"score":0.7,"reason":"While not directly stating authorship, confirming Kant's German nationality helps narrow down candidates when combined with the common knowledge premise that he authored the mentioned work."}

Q: $question
T: $triple
A: ''')


directly_answer_prompt = Template('''Answer the question using your knowledge. Return a JSON object with string "reason" and "answer" as a string or null when no answer can be determined.

Question: $question
Output:''')

answer_prompt = Template('''
Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer the question using these triplets and your own knowledge. Some triplets may be irrelevant or insufficient, in which case you should rely on your internal knowledge to reason logically.

Answer Format:
Return a valid JSON object with a string "reason" and an "answer" field containing the final answer as a string, or null if it cannot be determined.

Combine the provided triplets with your knowledge to perform comprehensive reasoning
End your answer with the final answer in the JSON "answer" field.
Reasoning Guidance:
First examine if the triplets provide direct relevant information
If triplets are insufficient or irrelevant, clearly explain how you use your internal knowledge
Make logical connections based on both given information and commonly known facts
Provide clear reasoning steps when bridging information gaps

Q: What is the capital of France?
Knowledge Triplets: France -> has.city -> Lyon, France -> exports.wine -> Italy
A: {"reason":"While the triplets mention France but don't provide information about its capital, from common knowledge we know that Paris is the capital of France. The given triplets about Lyon and wine exports are not relevant to this question.","answer":"Paris"}

Q: The artist nominated for The Long Winter lived where?
Knowledge Triplets: The Long Winter -> book.written_work.author -> Laura Ingalls Wilder, Laura Ingalls Wilder -> people.person.places_lived -> Unknown, Unknown -> people.place_lived.location -> De Smet
A: {"reason":"Based on the given knowledge triplets, the author of The Long Winter, Laura Ingalls Wilder, lived in De Smet.","answer":"De Smet"}

Q: Rift Valley Province is located in a nation that uses which form of currency?
Knowledge Triplets: Rift Valley Province -> location.administrative_division.country -> Kenya, Rift Valley Province -> location.location.geolocation -> Unknown, Rift Valley Province -> location.mailing_address.state_province_region -> Unknown
Kenya, location.country.currency_used, Kenyan shilling
A: {"reason":"Based on the given knowledge triplets, Rift Valley Province is located in Kenya, which uses the Kenyan shilling as its currency.","answer":"Kenyan shilling"}

Q: Who wrote Pride and Prejudice?
Knowledge Triplets: Pride and Prejudice -> has.character -> Elizabeth_Bennet, Pride and Prejudice -> published.year -> 1813
A: {"reason":"While the triplets tell us about a character and publication year of Pride and Prejudice, they don't directly state the author. However, it is well known that Jane Austen wrote this classic novel.","answer":"Jane Austen"}

Q: $question
Knowledge Triplets: $knowledge_triplets
A: ''')
