from string import Template

speculate_prompt = Template("""Task: Predict the logical relation path (Skeleton) for the new question. Please output a JSON object with the predicted relation path as a string in the "skeleton" field.

[Instructions]
1. Use the provided examples ONLY as a format reference.
2. CRITICAL: Your prediction must be based on the SEMANTICS of the [New Question], NOT the topic of the examples.
3. If the examples are irrelevant to the new question, IGNORE them and use your own knowledge.

[Example 1]
Question: $q1
Output: {"skeleton": "$s1"}

[Example 2]
Question: $q2
Output: {"skeleton": "$s2"}

[New Question]
Question: $question
Output: """)


subobjective_prompt = Template("""Please break down the process of answering the question into as few subobjectives as possible based on semantic analysis.
Here is an example:
Question: Which of the countries in the Caribbean has the smallest country calling code?
Output: {
    "subobjectives": [
        "Search the countries in the Caribbean",
        "Search the country calling code for each Caribbean country",
        "Compare the country calling codes to find the smallest one"
    ]
}

Now you need to directly output subobjectives of the following question in JSON format using the "subobjectives" field without other information or notes.
Question: $question
Output: """)


relation_extract_prompt = Template("""Please provide as few highly relevant relations as possible to the question and its subobjectives from the following numbered candidate relations (with their directions).
Here is an example:
Question: Name the president of the country whose main spoken language was Brahui in 1980?
Subobjectives: ['Identify the countries where the main spoken language is Brahui', 'Find the president of each country', 'Determine the president from 1980']
Topic Entity: Brahui Language
Relations: [1] [outgoing]language.human_language.main_country [2] [outgoing]language.human_language.language_family [3] [outgoing]language.human_language.iso_639_3_code [4] [incoming]base.rosetta.languoid.parent [5] [outgoing]language.human_language.writing_system [6] [outgoing]base.rosetta.languoid.languoid_class [7] [outgoing]language.human_language.countries_spoken_in [8] [outgoing]kg.object_profile.prominent_type [9] [outgoing]base.rosetta.languoid.document [10] [outgoing]base.ontologies.ontology_instance.equivalent_instances [11] [outgoing]base.rosetta.languoid.local_name [12] [outgoing]language.human_language.region
Output: {"selected_ids": [1, 7, 4]}

Now you need to directly output the numbers of relations highly related to the following question and its subobjectives in JSON format using the "selected_ids" field without other information or notes.
Question: $question
Subobjectives: $subobjectives
Topic Entity: $entity_name
Relations: $relations
Output: """)


entity_prune_prompt = Template("""Which entities in the following numbered candidates in Triplets can be used to answer question? Please provide the minimum possible number of entities, strictly following the constraints in the question.
Critical Instructions for Pruning:
1. Remove when clearly impossible:
   Only REMOVE an entity if it is clearly incompatible with the question based on the text itself
   (e.g., wrong answer type, wrong time/number, or obviously unrelated meaning).
2. When unsure, keep:
   If you are not sure whether an entity satisfies the constraints, do NOT delete it just based on a guess.
   It is better to keep uncertain but possibly relevant entities.
3. Efficiency:
   You MUST select at most 5 entities in total (including names and IDs).
   If more than 5 entities look possibly relevant or uncertain, choose up to 5 that best match the question
   (prefer earlier entities in the list and those sharing important words with the question).
4. All useless case:
   Only if every entity clearly cannot answer the question, you may output an empty selection.

Here are three examples:
Question: Which movies directed by Chris Renaud have featured Taylor Swift?
Triplets: Chris Renaud film.director.film [1] The Secret Life of Pets [2] No Time for Nuts [3] Despicable Me [4] The Lorax [5] Despicable Me 2
Output: {"selected_ids": [4, 3, 5]}

Question: The movie featured Miley Cyrus and was produced by Tobin Armbrust?
Triplets: Tobin Armbrust film.producer.film [1] The Resident [2] So Undercover [3] Let Me In [4] Begin Again [5] The Quiet Ones [6] A Walk Among the Tombstones
Output: {"selected_ids": [2]}

Question: Who served in the 21st Illinois Volunteer Infantry Regiment that also fought in the battle of Vicksburg?
Triplets: 21st Illinois Volunteer Infantry Regiment military.military_person.participated_in_conflicts [1] James B. McPherson [2] William F. Draper [3] Ulysses S. Grant [4] Henry Casey [5] Henry Dow
Output: {"selected_ids": [3]}

Now you need to directly output the entity numbers from the numbered candidates in Triplets for the following question in JSON format using the "selected_ids" field without other information or notes.
Question: $question
Triplets: $entity_name $relation $entity_candidates
Output: """)


update_memory_prompt = Template("""Based on the provided information (which may have missing parts and require further retrieval) and your own knowledge, output the currently known information required to achieve the subobjectives.
Here is an example:
Question: Find the person who said "Taste cannot be controlled by law", what did this person die from?
Subobjectives: ['Search the person who said "Taste cannot be controlled by law"', 'Search the cause of death for that person']
Memory:
Knowledge Triplets: Taste cannot be controlled by law. media_common.quotation.author [Thomas Jefferson]
Output: {
    "1": "Thomas Jefferson said 'Taste cannot be controlled by law'.",
    "2": "It is not mentioned, and I also don't know."
}

Now you need to directly output the results of the following question in JSON format without other information or notes.
Question: $question
Subobjectives: $subobjectives
Memory: $memory
Knowledge Triplets: $knowledge_triplets
Output: """)


evaluate_prompt = Template("""Please answer the question based on the memory, related knowledge triplets and your knowledge. Return your response in JSON format with "sufficient", "reason", and "answer".
Here are eight examples:
Question: Find the person who said "Taste cannot be controlled by law", what did this person die from?
Memory: {
    "1": "The triplet provides the information that Thomas Jefferson said this sentence.",
    "2": "No triplet provides this information."
}
Knowledge Triplets: Taste cannot be controlled by law., media_common.quotation.author, [Thomas Jefferson]
Output: {
    "sufficient": false,
    "reason": "The person who said \\"Taste cannot be controlled by law,\\" is Thomas Jefferson. It is still uncertain to answer the entire question",
    "answer": null
}

Question: The artist nominated for The Long Winter lived where?
Memory: {
    "1": "The triplets provide the information that the author of The Long Winter is Laura Ingalls Wilder.",
    "2": "The triplets provide this information that Laura Ingalls Wilder lived in De Smet."
}
Knowledge Triplets: The Long Winter, book.written_work.author, [Laura Ingalls Wilder]
Laura Ingalls Wilder, people.person.places_lived, [Unknown-Entity]
Unknown-Entity, people.place_lived.location, [De Smet]
Output: {
    "sufficient": true,
    "reason": "The author of The Long Winter is Laura Ingalls Wilder, and Laura Ingalls Wilder lived in De Smet.",
    "answer": "De Smet"
}

Question: Who is the coach of the team owned by Steve Bisciotti?
Memory: {
    "1": "The triplets provide the information that Steve Bisciotti owns Baltimore Ravens.",
    "2": "No triplets provide the information."
}
Knowledge Triplets: Steve Bisciotti, sports.professional_sports_team.owner_s, [Baltimore Ravens]
Steve Bisciotti, sports.sports_team_owner.teams_owned, [Baltimore Ravens]
Steve Bisciotti, organization.organization_founder.organizations_founded, [Allegis Group]
Output: {
    "sufficient": false,
    "reason": "The team owned by Steve Bisciotti is Baltimore Ravens based on knowledge triplets. The coach of the team owned by Steve Bisciotti is not explicitly mentioned.",
    "answer": null
}

Question: Rift Valley Province is located in a nation that uses which form of currency?
Memory: {
    "1": "The triplets provide the information that Rift Valley Province is located in Kenya.",
    "2": "The triplets provide the information that Kenya uses the Kenyan shilling as its currency."
}
Knowledge Triplets: Rift Valley Province, location.administrative_division.country, Kenya
Rift Valley Province, location.location.geolocation, UnName_Entity
Rift Valley Province, location.mailing_address.state_province_region, UnName_Entity
Kenya, location.country.currency_used, Kenyan shilling
Output: {
    "sufficient": true,
    "reason": "Based on knowledge triplets, Rift Valley Province is located in Kenya, which uses the Kenyan shilling as its currency.",
    "answer": "Kenyan shilling"
}

Question: The country with the National Anthem of Bolivia borders which nations?
Memory: {
    "1": "The triplets provide the information that the National Anthem of Bolivia is the anthem of Bolivia.",
    "2": "No triplets provide the information."
}
Knowledge Triplets: National Anthem of Bolivia, government.national_anthem_of_a_country.anthem, UnName_Entity
National Anthem of Bolivia, music.composition.composer, Leopoldo Benedetto Vincenti
National Anthem of Bolivia, music.composition.lyricist, José Ignacio de Sanjinés
UnName_Entity, government.national_anthem_of_a_country.country, Bolivia
Bolivia, location.country.national_anthem, UnName_Entity
Output: {
    "sufficient": false,
    "reason": "Based on knowledge triplets, the National Anthem of Bolivia is the anthem of Bolivia. Therefore, the country with the National Anthem of Bolivia is Bolivia. However, the given knowledge triplets do not provide information about which nations border Bolivia.",
    "answer": null
}

Question: Who served in the 21st Illinois Volunteer Infantry Regiment that also fought in the Battle of Vicksburg?
Memory: {
    "1": "The triplets show that the entity 'm.04h_h95' served in the 21st Illinois Volunteer Infantry Regiment.",
    "2": "The triplets show that 'm.04h_h95' is linked to Ulysses S. Grant.",
    "3": "The triplets show that Ulysses S. Grant participated in the Siege of Vicksburg."
}
Knowledge Triplets: 21st Illinois Volunteer Infantry Regiment, military.military_unit.servicemembers, [m.04h_h95]
m.04h_h95, military.military_person.service, Ulysses S. Grant
Ulysses S. Grant, military.military_person.participated_in_conflicts, Siege of Vicksburg
Output: {
    "sufficient": true,
    "reason": "The entity 'm.04h_h95' is a servicemember of the 21st Illinois Volunteer Infantry Regiment and is linked to Ulysses S. Grant. Grant is explicitly recorded as participating in the Siege of Vicksburg, so he satisfies both conditions in the question.",
    "answer": "Ulysses S. Grant"
}

Question: Which film directed by Nora Ephron has a runtime of 126 minutes?
Memory: {
    "1": "The triplets show several films directed by Nora Ephron.",
    "2": "One film has runtime '126.0'."
}
Knowledge Triplets: FilmA, film.film.directed_by, Nora Ephron
FilmA, film.film.runtime, "126.0"
Output: {
    "sufficient": true,
    "reason": "Runtime 126.0 matches 126, so FilmA satisfies the condition.",
    "answer": "FilmA"
}

Question: Which actor appeared in a film that won the Best Picture award?
Memory: {
    "1": "Triplets show ActorX appeared in FilmY.",
    "2": "Triplets show FilmY won Best Picture."
}
Knowledge Triplets: ActorX, film.actor.film, FilmY
FilmY, award.award_winning_work.awards_won, Best Picture
Output: {
    "sufficient": true,
    "reason": "ActorX appeared in a film that won Best Picture.",
    "answer": "ActorX"
}

Now you need to directly output the results of the following question in JSON format with "sufficient", "reason", and "answer" without other information or notes. If the triplets explicitly contains the answer to the question, prioritize the fact of the triplet over memory.
Question: $question
Memory: $memory
Knowledge Triplets: $knowledge_triplets
Output: """)


judge_reflection_prompt = Template("""Based on the current set of entities to be retrieved and the known information including memory and triplets, is it necessary to add additional entities for answering question?
Here are two examples:
Question: Which of the countries in the Caribbean has the smallest country calling code?
Entities set to be retrieved: ['Anguilla', 'Saint Lucia']
Memory: Caribbean contains Antilles and Saint Lucia
Knowledge Triplets: Caribbean, location.location.contains, ['Antilles', 'Saint Lucia']
Output: {
    "add": true,
    "reason": "The entities set ignores other countries in Caribbean."
}

Question: The artist nominated for The Long Winter lived where?
Entities set to be retrieved: ['Laura Ingalls Wilder']
Memory: "The author of The Long Winter is Laura Ingalls Wilder."
Knowledge Triplets: The Long Winter, book.written_work.author, [Laura Ingalls Wilder]
Output: {
    "add": false,
    "reason": "Now you need to search where Laura Ingalls Wilder lived."
}

Now you need to directly output the results of the following question in JSON format with "add" and "reason" without other information or notes.
Question: $question
Entities set to be retrieved: $entities
Memory: $memory
Knowledge Triplets: $knowledge_triplets
Output: """)


backtrack_entity_prompt = Template("""Please select the fewest necessary entities to be retrieved for answering the Q from Candidate Entities, based on the current known information (Memory), the reason for additional retrieval, and your own knowledge.
Here is an example:
Question: Which of the countries in the Caribbean has the smallest country calling code?
Reason: The entities set ignores other countries in the Caribbean.
Candidate Entities: [1] Saint Marie [2] Saint Martin (Island) [3] Viceroy Anguilla [4] Lesser Antilles [5] Barbados [6] British Virgin Islands [7] Leeward Islands [8] British West Indies [9] Caribbean [10] Saint Thomas [11] Bronte International University [12] Collectivity of Saint Martin [13] Southern Caribbean [14] University of Medicine and Health Sciences [15] Soufrière Hills [16] Lucayan Archipelago [17] University of the West Indies [18] Aureus University School of Medicine [19] North America [20] Netherlands Antilles [21] Puerto Rico [22] Chances Peak [23] Clarendon Parish [24] Saint Kitts and Nevis [25] Saint Lucia [26] Americas [27] Caribbean special municipalities of the Netherlands [28] Sandy Hill [29] School of Business and Computer Science, Trincity [30] School of Business and Computer Science, San Fernando [31] School of Business and Computer Science, Champs Fleurs [32] School of Business and Computer Science, Port of Spain [33] Bridgetown [34] St. Martinus University School of Medicine, main campus [35] Higher Institute of Medical Sciences. main campus [36] Grace University, main campus [37] Anguilla
Memory: {
    "1": "The countries in the Caribbean are Antilles and Latin America.",
    "2": "The country calling codes for Antilles and Latin America are not mentioned.",
    "3": "It is not mentioned which country has the smallest country calling code."
}
Output: {"selected_ids": [5, 25, 37]}

Now you need to directly output the numbers of selected candidate entities for the following question in JSON format using the "selected_ids" field without other information or notes.
Question: $question
Reason: $reason
Candidate Entities: $entity_candidates
Memory: $memory
Output: """)


answer_prompt = """Question: "{question}"
The Knowledge Graph search was incomplete. 
[Graph Evidence]: {path_str[:3500]} 

Based on the [Graph Evidence] and your INTERNAL KNOWLEDGE, provide the final answer.
Constraints:
1. If the answer appears in [Graph Evidence] as an ID (e.g., m.01234), Do NOT output Freebase IDs (starting with 'm.' or 'g.'), you MUST translate it to its real Name using your internal knowledge. Never output raw IDs.
2. Even if the relationship doesn't perfectly match (e.g. "directed" vs "acted"), if there is a strong candidate entity in the path, OUTPUT IT.
3. Do NOT output refusal phrases such as "did not", "no record", "unknown", "no information", "null", "none", or any similar negative expressions. Do NOT output full sentences explaining why the answer is missing. 
4. You MUST provide a best-guess answer. Prioritize entities found in the [Graph Evidence], but if the evidence is missing or incomplete, YOU MUST USE YOUR INTERNAL KNOWLEDGE to answer the question directly.
5. [Format & Logic] Output the final answer string. It could be an Entity Name, a Date, or a Number.Do NOT output JSON. Do NOT explain. Just output the answer string.
6. Strictly follow any constraints in the question.

Answer:"""


answer_prompt = Template("""The Knowledge Graph search was incomplete. Based on the Graph Evidence and your INTERNAL KNOWLEDGE, provide the final answer.
Question: "$question"
Graph Evidence: $graph_evidence

Constraints:
1. If the answer appears in [Graph Evidence] as an ID (e.g., m.01234), Do NOT output Freebase IDs (starting with 'm.' or 'g.'), you MUST translate it to its real Name using your internal knowledge. Never output raw IDs.
2. Even if the relationship doesn't perfectly match (e.g. "directed" vs "acted"), if there is a strong candidate entity in the path, OUTPUT IT.
3. Do NOT output refusal phrases such as "did not", "no record", "unknown", "no information", "null", "none", or any similar negative expressions. Do NOT output full sentences explaining why the answer is missing.
4. You MUST provide a best-guess answer. Prioritize entities found in the [Graph Evidence], but if the evidence is missing or incomplete, YOU MUST USE YOUR INTERNAL KNOWLEDGE to answer the question directly.
5. [Format & Logic] Output the final answer in JSON format using the "answer" field. Its value must be a string, which could be an Entity Name, a Date, or a Number. Do NOT explain.
6. Strictly follow any constraints in the question.

Output: """)




