EXTRACT_RELATION_PROMPT = """Please provide as few highly relevant relations as possible to the question and suggested relation from the following relations (separated by semicolons). But you need to output at least one.
Here is an example:
Q: Name the president of the country whose main spoken language was Brahui in 1980?
Suggested Relation:main_country,spoken_in
Topic Entity: Brahui Language
Relations: language.human_language.main_country; language.human_language.language_family; language.human_language.iso_639_3_code; base.rosetta.languoid.parent; language.human_language.writing_system; base.rosetta.languoid.languoid_class; language.human_language.countries_spoken_in; kg.object_profile.prominent_type; base.rosetta.languoid.document; base.ontologies.ontology_instance.equivalent_instances; base.rosetta.languoid.local_name; language.human_language.region
The output is: 
['language.human_language.main_country','language.human_language.countries_spoken_in','base.rosetta.languoid.parent']
Q: What country with a GDP Growth Rate Percentage of -0.008502 once do people speak Portuguese?
Suggested Relation:GDP growth rates 
Topic Entity: Angola
Relations: location.statistical_region.agriculture_as_percent_of_gdp; location.statistical_region.population_growth_rate; topic_server.population_number; location.statistical_region.gdp_growth_rate
The output is: 
['location.statistical_region.gdp_growth_rate']
Now you need to directly output relations highly related to the following question and its subobjectives in list format without other information or notes. Do not output any additional explanations. You must include the relations within [] without any other label.
Q: """

PRUNE_ENTITY_PROMPT = """
Which entities in the following list ([] in Triples) can be used to answer question? Please provide the minimum possible number of entities, and strictly adhering to the constraints mentioned in the question. The entity might have ID form such as 'm.xxxx', you can also choose them.
Here is an example:
Q: The movie featured Miley Cyrus and was produced by Tobin Armbrust?
Suggested Entity:(Miley Cyrus,film.producer.film, So Undercover)
Triplets: Miley Cyrus, film.producer.film, [The Resident; So Undercover; Let Me In; Begin Again; The Quiet Ones; A Walk Among the Tombstones]
Output: ['So Undercover']
Now you need to directly output the entities from [] for the following question in list format without other information or notes.
Do not output empty list or entities not in [].Refer to the question if not suggested entity is provided.Do not output any additional explanations.
Q: {}"""

VERIFY_PROMPT = """Based on the knowledge triplets, analyze whether it is necessary to revise LLM's observation or add additional triplets in LLM's observation. The entity might have ID form such as m.xxxx. If all entities starts with 'm.', just ignore them and no need to revise. 
You are only allowed to use the information from knowledge triplets. DO NOT use your own knowledge.

Here are few examples:
LLM output:   
Thought 2: Now that I have a list of countries where Portuguese is spoken, I need to find out which of these countries had a child labor percentage of 1.8%.  
Action 2: Search[child labor percentage in Portugal, Brazil, Angola, Mozambique, Guinea-Bissau, East Timor, Equatorial Guinea, Cape Verde, São Tomé and Príncipe]  
Observation 2: (Brazil, unit.dated_percentage.rate, 1.8%), (Angola, unit.dated_percentage.rate, 1.5%), (Mozambique, unit.dated_percentage.rate, 2.0%), (Guinea-Bissau, unit.dated_percentage.rate, 2.3%), (East Timor, unit.dated_percentage.rate, 1.0%), (Equatorial Guinea, unit.dated_percentage.rate, 1.2%), (Cape Verde, unit.dated_percentage.rate, 1.4%), (São Tomé and Príncipe, unit.dated_percentage.rate, 1.6%), (Portugal, unit.dated_percentage.rate, 0.5%)  
Knowledge Triplets: (Mozambique, unit.dated_percentage.rate, 1.8%),(Brazil, unit.dated_percentage.rate, 1.8%)
Output: 
{
    "Revise": "Yes",
    "Reason": "Based on the triplets, the observations have factual errors that might lead to wrong answer.",
    "Revised Observation": (Brazil, unit.dated_percentage.rate, 2.0%), (Angola, unit.dated_percentage.rate, 1.5%), (Mozambique, unit.dated_percentage.rate, 1.8%), (Guinea-Bissau, unit.dated_percentage.rate, 2.3%), (East Timor, unit.dated_percentage.rate, 1.0%), (Equatorial Guinea, unit.dated_percentage.rate, 1.2%), (Cape Verde, unit.dated_percentage.rate, 1.4%), (São Tomé and Príncipe, unit.dated_percentage.rate, 1.6%), (Portugal, unit.dated_percentage.rate, 0.5%)  
}
LLM output:  
Thought 2: Now that I have identified that "Noble patria, tu hermosa bandera" is the national anthem of Venezuela, I need to find out what currency is used in that country.
Action 2: Search[currency of Venezuela]  
Observation 2: ('Venezuela', currency is, 'Venezuelan bolívar')
Knowledge Triplets: ("m.0h_1h3x","location.country.national_anthem","Costa Rica")
Output: 
{
    "Revise": "Yes",
    "Reason": "Based on the triplet, the observations have factual errors that might lead to wrong answer.",
    "Revised Observation":(Noble patria, tu hermosa bandera', national anthem of, Costa Rica)
}
LLM output:   
Thought 1: I need to know the artist of the "Country Nation World Tour"
Action 1: Search[artist of "Country Nation World Tour"]
Observation 1: （Country Nation World Tour, artist, Brad Paisley)

Knowledge Triplets: ('Country Nation World Tour','music.concert_tour.artist','Brad Paisley')
Output: 
{
    "Revise": "No",
    "Reason": "Based on the triplets, current observation is factually correct."
}

Now you need to directly output the results of the following question in the JSON format (must include "Revise" and "Reason", if 'yes' in revise, include "Revised Observation") without other information or notes.
Q: """

GENERATE_PLAN_PROMPT = """
    You are an intelligent assistant tasked with answering the following question. Your job is to understand the question and plan all the necessary steps to solve it. Do not judge the question or give an unknown answer. Do NOT make too long (>10 steps) plans.
    You can only use the following two actions:
    (1) Search[Keyword]: To retrieve relative information based on the given question.  
    (2) Finish[Answer]: When the observations are sufficient to answer the question, return the final answer and finish the task.  
    
    Your response should only include the Thought, Action and Observation steps required to solve the question until the final answer is reached. Each step must strictly follow the format:
    Thought i:
    Action i:
    Observation i:
    The observation should be in the format of knowledge triplets. More than one triplet is allowed in observation.
    ### Here are the examples:
    Question:In which countries do the people speak Portuguese, where the child labor percentage was once 1.8?
    Thought 1: I need to identify the countries where Portuguese is spoken.  
    Action 1: Search[countries where Portuguese is spoken]  
    Observation 1: (Portuguese, spoken in, Portugal), (Portuguese, spoken in, Brazil), (Portuguese, spoken in, Angola), (Portuguese, spoken in, Mozambique), (Portuguese, spoken in, Guinea-Bissau), (Portuguese, spoken in, East Timor), (Portuguese, spoken in, Equatorial Guinea), (Portuguese, spoken in, Cape Verde), (Portuguese, spoken in, São Tomé and Príncipe)  
    
    Thought 2: Now that I have a list of countries where Portuguese is spoken, I need to find out which of these countries had a child labor percentage of 1.8%.  
    Action 2: Search[child labor percentage in Portugal, Brazil, Angola, Mozambique, Guinea-Bissau, East Timor, Equatorial Guinea, Cape Verde, São Tomé and Príncipe]  
    Observation 2: (Brazil, unit.dated_percentage.rate, 2.0%), (Angola, unit.dated_percentage.rate, 1.5%), (Mozambique, unit.dated_percentage.rate, 1.8%), (Guinea-Bissau, unit.dated_percentage.rate, 2.3%), (East Timor, unit.dated_percentage.rate, 1.0%), (Equatorial Guinea, unit.dated_percentage.rate, 1.2%), (Cape Verde, unit.dated_percentage.rate, 1.4%), (São Tomé and Príncipe, unit.dated_percentage.rate, 1.6%), (Portugal, unit.dated_percentage.rate, 0.5%)  
    
    Thought 3: The only country where Portuguese is spoken and had a child labor percentage of 1.8% is Mozambique.  
    Action 3: Finish[Mozambique]

    Question:Who held his governmental position from before Janaury 6, 2003 and was the 2009 Governor of Arizona?
    Thought 1: I need to find out who was the Governor of Arizona in 2009.  
    Action 1: Search[2009 Governor of Arizona]  
    Observation 1: (Janet Napolitano, served as, Governor of Arizona), (Janet Napolitano, served from, January 2003 to January 2009)  
    
    Thought 2: Now that I know Janet Napolitano was the Governor of Arizona in 2009, I need to check if she held her governmental position before January 6, 2003.  
    Action 2: Search[Janet Napolitano governmental positions before January 6, 2003]  
    Observation 2: (Janet Napolitano, served as, Attorney General of Arizona), (Janet Napolitano, served as, U.S. Attorney for Arizona)  
    
    Thought 3: Janet Napolitano held governmental positions as Attorney General of Arizona and U.S. Attorney for Arizona before January 6, 2003.  
    Action 3: Finish[Janet Napolitano]
    Follow these example and answer Question:{question}. Do not output other additional information. You can only end with Finish[] actions.
    
    """

UPDATE_PLAN_PROMPT = """
Complete the plan for the given question with the following steps:
Here is two examples:
Question: What movie with a prequel called "Escape to Witch Mountain" was Kim Richards in?
Thought 1: I need to identify the movie that has a prequel called "Escape to Witch Mountain" and features Kim Richards.  
Action 1: Search[Kim Richards movie "Escape to Witch Mountain"]  
Observation 1: (Kim Richards, starred in, "Escape to Witch Mountain"), ("Escape to Witch Mountain", has prequel, "Return from Witch Mountain")  
Thought 2: Now that I have identified that Kim Richards starred in "Escape to Witch Mountain," I need to confirm if there is another movie related to it that features her.  
Action 2: Search[Kim Richards filmography]  
Observation 2: (Kim Richards, starred in, "Return from Witch Mountain"), (Kim Richards, starred in, "Escape to Witch Mountain")  
Output:Thought 3: I have confirmed that Kim Richards starred in both "Escape to Witch Mountain" and its sequel "Return from Witch Mountain." Since the question asks for the movie with a prequel called "Escape to Witch Mountain," I can conclude that the answer is "Return from Witch Mountain."  
Action 3: Finish[Return from Witch Mountain]
Example 2:
Question:What currency is used in the country with Nobel Patria, tu hermosa as its national anthem?
Thought 1: I need to identify the country that has "Noble patria, tu hermosa bandera" as its national anthem.
Action 1: Search["Noble patria, tu hermosa bandera" national anthem country]
Observation 1: '("Noble patria, tu hermosa bandera", national anthem of, country associated with m.0h_1h3x)'
Thought 2: Now that I have identified the country associated with the national anthem "Noble patria, tu hermosa bandera," I need to find out what currency is used in that country.
Action 2: Search[currency of country associated with "Noble patria, tu hermosa bandera"]
Observation 2: ("Noble patria, tu hermosa bandera", national anthem of, "Costa Rica"), ("Costa Rica", currency is, "Costa Rican colón")
Output:Thought 3: I have confirmed that the national anthem "Noble patria, tu hermosa bandera" belongs to Costa Rica. Now I need to state the currency used in Costa Rica.  
Action 3: Finish[Costa Rican colón]
Follow this example and continue current steps to answer initial question.  Use the same word in current observation and ensure you answer the question.
Question:
"""

UPDATE_PLAN_LOOKAHEAD_PROMPT = """
    You are provided with a reasoning plan for the following question and future relations as reference.Based on the given context, revise the plan as needed to correctly answer the question. 
    Your job is to understand the question and plan all the necessary steps to solve it. 
    You can use the following two actions:
    (1) Search[Keyword]: To retrieve relative information based on the given question.  
    (2) Finish[Answer]: When the observations are sufficient to answer the question, return the final answer extracted from the observation.  
    
    Your response should include all the Thought, Action and Observation steps required to solve the question until the final answer is reached. Each step must follow the format:
    Thought i:
    Action i:
    Observation i:
    The observation should be in the format of knowledge triplets. More than one triplet is allowed in observation.
    ### Example 1:
    Question: What movie with a prequel called "Escape to Witch Mountain" was Kim Richards in?
    Thought 1: I need to identify the movie that has a prequel called "Escape to Witch Mountain" and features Kim Richards.  
    Action 1: Search[Kim Richards movie "Escape to Witch Mountain"]  
    Observation 1: (Kim Richards, starred in, "Escape to Witch Mountain"), ("Escape to Witch Mountain", has prequel, "Return from Witch Mountain")  
    Thought 2: Now that I have identified that Kim Richards starred in "Escape to Witch Mountain," I need to confirm if there is another movie related to it that features her.  
    Action 2: Search[Kim Richards filmography]  
    Observation 2: (Kim Richards, starred in, "Return from Witch Mountain"), (Kim Richards, starred in, "Escape to Witch Mountain")  
    Thought 3: I have confirmed that Kim Richards starred in both "Escape to Witch Mountain" and its sequel "Return from Witch Mountain." Since the question asks for the movie with a prequel called "Escape to Witch Mountain," I can conclude that the answer is "Return from Witch Mountain."  
    Action 3: Finish[Return from Witch Mountain]
    ### Example 2:
    Question:What currency is used in the country with Nobel Patria, tu hermosa as its national anthem?
    Thought 1: I need to identify the country that has "Noble patria, tu hermosa bandera" as its national anthem.
    Action 1: Search["Noble patria, tu hermosa bandera" national anthem country]
    Observation 1: '("Noble patria, tu hermosa bandera", national anthem of, country associated with m.0h_1h3x)'
    Now you can observe:("Noble patria, tu hermosa bandera", national anthem of, "Costa Rica"), ("Costa Rica", currency is, "Costa Rican colón")
    Please use this current observation to revise given steps and complete the whole plan.
    Output:Thought 2: Now that I have identified the country associated with the national anthem "Noble patria, tu hermosa bandera," I need to find out what currency is used in that country.
    Action 2: Search[currency of country associated with "Noble patria, tu hermosa bandera"]
    Observation 2: ("Noble patria, tu hermosa bandera", national anthem of, "Costa Rica"), ("Costa Rica", currency is, "Costa Rican colón")
    Thought 3: I have confirmed that the national anthem "Noble patria, tu hermosa bandera" belongs to Costa Rica. So the answer to the question is Costa Rica.  
    Action 3: Finish[Costa Rican colón] 
    ###
    Here is the next step's information from KG for your to continue above steps.{info}
    Strictly follow the format in example and use above information to continue current steps below and answer initial question. Try to use exact words from observation and ensure you answer the question.
    Question:{question}
    {steps}
    """

UPDATE_PLAN_LOCAL_PROMPT = """
    You are provided with a reasoning plan for the following question. Based on the given current observation, revise the plan as needed to correctly answer the question. You can also adjust previous steps based on how the observation aligns with or contradicts existing steps.The observations must strictly come from the KG triplets.
    
    Available actions:
    (1) Search[Keyword]: To retrieve relevant information based on the given question.  
    (2) Finish[Answer]: When the observations are sufficient to answer the question, return the final answer extracted from the observation.
    
    Your response should include all Thought, Action, and Observation steps required to solve the question until the final answer is reached. Each step must follow the format:
    Thought i:
    Action i:
    Observation i:
    
    The observation should be structured in knowledge triplets. Multiple triplets per observation are allowed.
    
    ### Example 1:
    Question: What movie with a prequel called "Escape to Witch Mountain" was Kim Richards in?
    Thought 1: I need to identify the movie with a prequel called "Escape to Witch Mountain" that features Kim Richards.  
    Action 1: Search[Kim Richards movie "Escape to Witch Mountain"]  
    Observation 1: (Kim Richards, starred in, "Escape to Witch Mountain"), ("Escape to Witch Mountain", has prequel, "Return from Witch Mountain")  
    Thought 2: Now that I have identified that Kim Richards starred in "Escape to Witch Mountain," I should verify if she starred in the related film "Return from Witch Mountain."  
    Action 2: Search[Kim Richards filmography]  
    Observation 2: (Kim Richards, starred in, "Return from Witch Mountain"), (Kim Richards, starred in, "Escape to Witch Mountain")  
    Thought 3: Kim Richards starred in both films. Since the question asks for the movie with a prequel called "Escape to Witch Mountain," the answer is "Return from Witch Mountain."  
    Action 3: Finish[Return from Witch Mountain]
    ### Example 2:
    Question:What currency is used in the country with Nobel Patria, tu hermosa as its national anthem?
    Thought 1: I need to identify the country that has "Noble patria, tu hermosa bandera" as its national anthem.
    Action 1: Search["Noble patria, tu hermosa bandera" national anthem country]
    Observation 1: '("Noble patria, tu hermosa bandera", national anthem of, country associated with m.0h_1h3x)'
    Now you can observe:("Noble patria, tu hermosa bandera", national anthem of, "Costa Rica"), ("Costa Rica", currency is, "Costa Rican colón")
    Please use this current observation to revise given steps and complete the whole plan.
    Output:Thought 2: Now that I have identified the country associated with the national anthem "Noble patria, tu hermosa bandera," I need to find out what currency is used in that country.
    Action 2: Search[currency of country associated with "Noble patria, tu hermosa bandera"]
    Observation 2: ("Noble patria, tu hermosa bandera", national anthem of, "Costa Rica"), ("Costa Rica", currency is, "Costa Rican colón")
    Thought 3: I have confirmed that the national anthem "Noble patria, tu hermosa bandera" belongs to Costa Rica. So the answer to the question is Costa Rica.  
    Action 3: Finish[Costa Rican colón] 
     
    Follow this example to revise or continue the current steps using exactly the current observation provided below. Ensure your final plan answers the original question correctly.
    
    Question: {question}
    Current steps: {steps}
    Current observation: {observation}
    
    Output:"""

UPDATE_PLAN_GLOBAL_PROMPT = """
    You are provided with a reasoning plan for the following question and a set of knowledge graph (KG) triplets. The existing reasoning plan is severely incorrect or unreliable. Please revise the reasoning process completely from Thought 1.
    If the problem is conjunction, you are encouraged to utilize the triplets that have appeared in the Knowledge Graph (KG) to infer the earlier steps. The observations must strictly come from the KG triplets.
    
    (1) Search[Keyword]: To retrieve relative information based on the given question.  
    (2) Finish[Answer]: When the observations are sufficient to answer the question, return the final answer extracted from the observation.  
    
    Your response should include all the Thought, Action and Observation steps required to solve the question until the final answer is reached. Each step must follow the format:
    Thought i:
    Action i:
    Observation i:
    The observation should be in the format of knowledge triplets. More than one triplet is allowed in observation.
    Here is two examples:
    ### Example 1
    Question: What movie with a prequel called "Escape to Witch Mountain" was Kim Richards in?
    Thought 1: I need to identify the movie that has a prequel called "Escape to Witch Mountain" and features Kim Richards.  
    Action 1: Search[Kim Richards movie "Escape to Witch Mountain"]  
    Observation 1: (Kim Richards, starred in, "Escape to Witch Mountain"), ("Escape to Witch Mountain", has prequel, "Return from Witch Mountain")  
    Now you can observe:(Kim Richards, starred in, "Return from Witch Mountain"), (Kim Richards, starred in, "Escape to Witch Mountain") 
    Please use this current observation to revise given steps and complete the whole plan.
    Output:Thought 2: Now that I have identified that Kim Richards starred in "Escape to Witch Mountain," I need to confirm if there is another movie related to it that features her.  
    Action 2: Search[Kim Richards filmography]  
    Observation 2: (Kim Richards, starred in, "Return from Witch Mountain"), (Kim Richards, starred in, "Escape to Witch Mountain")  
    Thought 3: I have confirmed that Kim Richards starred in both "Escape to Witch Mountain" and its sequel "Return from Witch Mountain." Since the question asks for the movie with a prequel called "Escape to Witch Mountain," I can conclude that the answer is "Return from Witch Mountain."  
    Action 3: Finish[Return from Witch Mountain]
    ### Example 2:
    Question:What currency is used in the country with Nobel Patria, tu hermosa as its national anthem?
    Thought 1: I need to identify the country that has "Noble patria, tu hermosa bandera" as its national anthem.
    Action 1: Search["Noble patria, tu hermosa bandera" national anthem country]
    Observation 1: '("Noble patria, tu hermosa bandera", national anthem of, country associated with m.0h_1h3x)'
    Now you can observe:("Noble patria, tu hermosa bandera", national anthem of, "Costa Rica"), ("Costa Rica", currency is, "Costa Rican colón")
    Please use this current observation to revise given steps and complete the whole plan.
    Output:Thought 2: Now that I have identified the country associated with the national anthem "Noble patria, tu hermosa bandera," I need to find out what currency is used in that country.
    Action 2: Search[currency of country associated with "Noble patria, tu hermosa bandera"]
    Observation 2: ("Noble patria, tu hermosa bandera", national anthem of, "Costa Rica"), ("Costa Rica", currency is, "Costa Rican colón")
    Thought 3: I have confirmed that the national anthem "Noble patria, tu hermosa bandera" belongs to Costa Rica. So the answer to the question is Costa Rica.  
    Action 3: Finish[Costa Rican colón] 
    ###
     Strictly follow the format in example without additional notation and use current KG retrieved triplets to revise whole steps below and answer initial question. Try to align each reasoning step with the provided KG triplets below. Ensure the final plan answers the original question correctly.
    Question:{question}
    Current plan:{steps}
    KG triplets:{KG}
    
    Output:"""

ANSWER_PROMPT = """Answer the question using the retrieved knowledge graph triplets and your knowledge. Give a concise answer only.
Question: """
