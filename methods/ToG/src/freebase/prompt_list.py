from string import Template

extract_relation_prompt = Template("""Please retrieve $relation_num relations that contribute to the question from the numbered candidate relations (with their directions) and rate their contribution on a scale from 0 to 1. Return the selected relation numbers and their scores in JSON format.
Q: Name the president of the country whose main spoken language was Brahui in 1980?
Topic Entity: Brahui Language
Relations: [1] [outgoing] language.human_language.main_country [2] [outgoing] language.human_language.language_family [3] [outgoing] language.human_language.iso_639_3_code [4] [incoming] base.rosetta.languoid.parent [5] [outgoing] language.human_language.countries_spoken_in
Output: {
  "scores": [
    {"id": 1, "score": 0.4},
    {"id": 5, "score": 0.3},
    {"id": 4, "score": 0.2}
  ]
}

Q: $question
Topic Entity: $entity_name
Relations: $relations
Output:""")


select_entity_candidates_prompt = Template("""Please select the entities that contribute to answering the question. Return their numbers in JSON format using the "selected_ids" field, or an empty "selected_ids" list if none are useful.
Q: The movie featured Miley Cyrus and was produced by Tobin Armbrust?
Relation: film.producer.film
Entities: [1] The Resident [2] So Undercover [3] Let Me In [4] Begin Again [5] The Quiet Ones [6] A Walk Among the Tombstones
Output: {"selected_ids": [2]}

Q: $question
Current Entity: $entity_name
Relation: $relation
Entities: $entity_candidates
Output:""")


answer_prompt = Template("""Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer the question with these triplets and your knowledge. Return your response in JSON format with "reason" and "answer".
Q: Find the person who said "Taste cannot be controlled by law", what did this person die from?
Knowledge Triplets: Taste cannot be controlled by law., media_common.quotation.author, Thomas Jefferson
A: {
  "reason": "Based on the given knowledge triplets, it's not sufficient to answer the entire question. The triplets only provide information about the person who said \\"Taste cannot be controlled by law\\" which is Thomas Jefferson. To answer the second part of the question, it's necessary to have additional knowledge about where Thomas Jefferson's dead.",
  "answer": null
}

Q: The artist nominated for The Long Winter lived where?
Knowledge Triplets: The Long Winter, book.written_work.author, Laura Ingalls Wilder
Laura Ingalls Wilder, people.person.places_lived, Unknown-Entity
Unknown-Entity, people.place_lived.location, De Smet
A: {
  "reason": "Based on the given knowledge triplets, the author of The Long Winter, Laura Ingalls Wilder, lived in De Smet.",
  "answer": "De Smet"
}

Q: Who is the coach of the team owned by Steve Bisciotti?
Knowledge Triplets: Steve Bisciotti, sports.professional_sports_team.owner_s, Baltimore Ravens
Steve Bisciotti, sports.sports_team_owner.teams_owned, Baltimore Ravens
Steve Bisciotti, organization.organization_founder.organizations_founded, Allegis Group
A: {
  "reason": "Based on the given knowledge triplets, the coach of the team owned by Steve Bisciotti is not explicitly mentioned. However, it can be inferred that the team owned by Steve Bisciotti is the Baltimore Ravens, a professional sports team. Therefore, additional knowledge about the current coach of the Baltimore Ravens can be used to answer the question.",
  "answer": null
}


Q: Rift Valley Province is located in a nation that uses which form of currency?
Knowledge Triplets: Rift Valley Province, location.administrative_division.country, Kenya
Rift Valley Province, location.location.geolocation, UnName_Entity
Rift Valley Province, location.mailing_address.state_province_region, UnName_Entity
Kenya, location.country.currency_used, Kenyan shilling
A: {
  "reason": "Based on the given knowledge triplets, Rift Valley Province is located in Kenya, which uses the Kenyan shilling as its currency.",
  "answer": "Kenyan shilling"
}

Q: The country with the National Anthem of Bolivia borders which nations?
Knowledge Triplets: National Anthem of Bolivia, government.national_anthem_of_a_country.anthem, UnName_Entity
National Anthem of Bolivia, music.composition.composer, Leopoldo Benedetto Vincenti
National Anthem of Bolivia, music.composition.lyricist, José Ignacio de Sanjinés
UnName_Entity, government.national_anthem_of_a_country.country, Bolivia
Bolivia, location.country.national_anthem, UnName_Entity
A: {
  "reason": "Based on the given knowledge triplets, we can infer that the National Anthem of Bolivia is the anthem of Bolivia. Therefore, the country with the National Anthem of Bolivia is Bolivia itself. However, the given knowledge triplets do not provide information about which nations border Bolivia. To answer this question, we need additional knowledge about the geography of Bolivia and its neighboring countries.",
  "answer": null
}

Q: $question
Knowledge Triplets: $knowledge_triplets
A: """)


evaluate_prompt = Template("""Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer whether it's sufficient for you to answer the question with these triplets and your knowledge (Yes or No). Return your response in JSON format with "sufficient", "reason", and "answer".
Q: Find the person who said "Taste cannot be controlled by law", what did this person die from?
Knowledge Triplets: Taste cannot be controlled by law., media_common.quotation.author, Thomas Jefferson
A: {
  "sufficient": false,
  "reason": "Based on the given knowledge triplets, it's not sufficient to answer the entire question. The triplets only provide information about the person who said \\"Taste cannot be controlled by law\\" which is Thomas Jefferson. To answer the second part of the question, it's necessary to have additional knowledge about where Thomas Jefferson's dead.",
  "answer": null
}

Q: The artist nominated for The Long Winter lived where?
Knowledge Triplets: The Long Winter, book.written_work.author, Laura Ingalls Wilder
Laura Ingalls Wilder, people.person.places_lived, Unknown-Entity
Unknown-Entity, people.place_lived.location, De Smet
A: {
  "sufficient": true,
  "reason": "Based on the given knowledge triplets, the author of The Long Winter, Laura Ingalls Wilder, lived in De Smet.",
  "answer": "De Smet"
}

Q: Who is the coach of the team owned by Steve Bisciotti?
Knowledge Triplets: Steve Bisciotti, sports.professional_sports_team.owner_s, Baltimore Ravens
Steve Bisciotti, sports.sports_team_owner.teams_owned, Baltimore Ravens
Steve Bisciotti, organization.organization_founder.organizations_founded, Allegis Group
A: {
  "sufficient": false,
  "reason": "Based on the given knowledge triplets, the coach of the team owned by Steve Bisciotti is not explicitly mentioned. However, it can be inferred that the team owned by Steve Bisciotti is the Baltimore Ravens, a professional sports team. Therefore, additional knowledge about the current coach of the Baltimore Ravens can be used to answer the question.",
  "answer": null
}

Q: Rift Valley Province is located in a nation that uses which form of currency?
Knowledge Triplets: Rift Valley Province, location.administrative_division.country, Kenya
Rift Valley Province, location.location.geolocation, UnName_Entity
Rift Valley Province, location.mailing_address.state_province_region, UnName_Entity
Kenya, location.country.currency_used, Kenyan shilling
A: {
  "sufficient": true,
  "reason": "Based on the given knowledge triplets, Rift Valley Province is located in Kenya, which uses the Kenyan shilling as its currency.",
  "answer": "Kenyan shilling"
}

Q: The country with the National Anthem of Bolivia borders which nations?
Knowledge Triplets: National Anthem of Bolivia, government.national_anthem_of_a_country.anthem, UnName_Entity
National Anthem of Bolivia, music.composition.composer, Leopoldo Benedetto Vincenti
National Anthem of Bolivia, music.composition.lyricist, José Ignacio de Sanjinés
UnName_Entity, government.national_anthem_of_a_country.country, Bolivia
Bolivia, location.country.national_anthem, UnName_Entity
A: {
  "sufficient": false,
  "reason": "Based on the given knowledge triplets, we can infer that the National Anthem of Bolivia is the anthem of Bolivia. Therefore, the country with the National Anthem of Bolivia is Bolivia itself. However, the given knowledge triplets do not provide information about which nations border Bolivia. To answer this question, we need additional knowledge about the geography of Bolivia and its neighboring countries.",
  "answer": null
}

Q: $question
Knowledge Triplets: $knowledge_triplets
A: """)


answer_directly_prompt = Template("""Answer the question in JSON format with "reason" and "answer".
Q: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
A: {
  "reason": "First, the education institution has a sports team named George Washington Colonials men's basketball in is George Washington University , Second, George Washington University is in Washington D.C.",
  "answer": "Washington, D.C."
}

Q: Who lists Pramatha Chaudhuri as an influence and wrote Jana Gana Mana?
A: {
  "reason": "First, Bharoto Bhagyo Bidhata wrote Jana Gana Mana. Second, Bharoto Bhagyo Bidhata lists Pramatha Chaudhuri as an influence.",
  "answer": "Bharoto Bhagyo Bidhata"
}

Q: Who was the artist nominated for an award for You Drive Me Crazy?
A: {
  "reason": "First, the artist nominated for an award for You Drive Me Crazy is Britney Spears.",
  "answer": "Jason Allen Alexander"
}

Q: What person born in Siegen influenced the work of Vincent Van Gogh?
A: {
  "reason": "First, Peter Paul Rubens, Claude Monet and etc. influenced the work of Vincent Van Gogh. Second, Peter Paul Rubens born in Siegen.",
  "answer": "Peter Paul Rubens"
}

Q: What is the country close to Russia where Mikheil Saakashvii holds a government position?
A: {
  "reason": "First, China, Norway, Finland, Estonia and Georgia is close to Russia. Second, Mikheil Saakashvii holds a government position at Georgia.",
  "answer": "Georgia"
}

Q: What drug did the actor who portrayed the character Urethane Wheels Guy overdosed on?
A: {
  "reason": "First, Mitchell Lee Hedberg portrayed character Urethane Wheels Guy. Second, Mitchell Lee Hedberg overdose Heroin.",
  "answer": "Heroin"
}

Q: $question
A: """)
