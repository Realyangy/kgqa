from string import Template


_STATIC_SUBGOAL_DEFINITIONS = """STATIC SUBGOALS
These are initial retrieval targets and their planned dependencies, not solved facts.
Use them to guide retrieval in the context of the original question and the current path."""


_FINAL_ANSWER_POLICY = """EVIDENCE AND ANSWER POLICY
1. Use only the supplied KG evidence when sufficient. Otherwise, you may supplement them with internal knowledge without contradicting explicit KG evidence.
2. Output none if neither source supports a reliable answer.
3. Use a short phrase or word, matching the entity name when possible. Separate all valid answers with commas.

OUTPUT CONTRACT
Output exactly one answer tag:
<final_answer>answer</final_answer>"""


_DIRECT_COT_FEW_SHOT_PROMPT = """Answer the question with concise step-by-step reasoning.

Rules:
1. Output the final answer first, and put only it inside <final_answer>...</final_answer>; separate multiple answers with commas.
2. Follow the tag with a concise explanation prefixed with "Reasoning:" that supports and is consistent with the answer.

Q: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
A: <final_answer>Washington, D.C.</final_answer>
Reasoning: First, the education institution that has a sports team named George Washington Colonials men's basketball is George Washington University. Second, George Washington University is in Washington, D.C.

Q: Who lists Pramatha Chaudhuri as an influence and wrote Jana Gana Mana?
A: <final_answer>Rabindranath Tagore</final_answer>
Reasoning: First, Rabindranath Tagore wrote Jana Gana Mana. Second, Rabindranath Tagore lists Pramatha Chaudhuri as an influence.

Q: Who did the artist nominated for an award for You Drive Me Crazy marry?
A: <final_answer>Jason Allen Alexander, Kevin Federline</final_answer>
Reasoning: First, the artist nominated for an award for You Drive Me Crazy is Britney Spears. Second, Britney Spears married Jason Allen Alexander and Kevin Federline.

Q: What person born in Siegen influenced the work of Vincent van Gogh?
A: <final_answer>Peter Paul Rubens</final_answer>
Reasoning: First, consider artists who influenced Vincent van Gogh. Peter Paul Rubens is one such artist. Second, check the birthplace clue. Peter Paul Rubens was born in Siegen.

Q: What is the country close to Russia where Mikheil Saakashvili holds a government position?
A: <final_answer>Georgia</final_answer>
Reasoning: First, China, Norway, Finland, Estonia and Georgia are close to Russia. Second, Mikheil Saakashvili holds a government position in Georgia.

Q: What drug did the actor who portrayed the character Urethane Wheels Guy overdose on?
A: <final_answer>Heroin</final_answer>
Reasoning: First, Mitchell Lee Hedberg portrayed the character Urethane Wheels Guy. Second, Mitchell Lee Hedberg overdosed on heroin."""


# 1. Complexity judgment and initial planning
complexity_judge_system_prompt = """TASK
Classify a KGQA question for graph search.

DECISION RULES
1. "simple" needs one direct retrieval constraint from the Topic entities.
2. "complex" needs multiple constraints, such as chained retrieval, candidate filtering, comparison, counting, or intersection.
3. Describe candidate-filter questions as a narrow candidate retrieval followed by verification of the remaining condition within those candidates.

OUTPUT CONTRACT
Output valid JSON only:
{"complexity":"simple" or "complex","reason":"short reason"}"""

complexity_judge_user_prompt = Template("""Question:
$question
Topic entities:
$topic_entities
Judgment:""")


simple_plan_system_prompt = """TASK
Create exactly one executable retrieval subgoal for a simple KGQA question.

AVAILABLE INFORMATION
Use only the Question and Topic entities. Introduce no facts, answers, or assumed KG relations.

PLAN RULES
1. Keep every Topic-entity constraint.
2. Use only supplied Topic entities as anchor_entity.
3. The single subgoal must use resolve_mode "retrieval" and have no dependencies.

OUTPUT CONTRACT
Output valid JSON only:
{"subgoals":[{"id":"g1","goal":"...","depend":[],"answer_type":"...","anchor_entity":"Topic entity name(s)","resolve_mode":"retrieval"}]}"""

simple_plan_user_prompt = Template("""Question: $question
Topic entities: $topic_entities
Output:""")


decompose_system_prompt = """TASK
Decompose a complex KGQA question into the smallest executable MCTS plan, using at most three subgoals including auto subgoals.

AVAILABLE INFORMATION
Use only the Question, Topic entities, and Complexity reason. Introduce no answers, facts, or assumed KG relations.

PLAN CONSTRUCTION RULES
1. Preserve every question constraint using the fewest semantic steps, not one subgoal per KG hop. Keep an entity with its needed role, property, date, or qualifier unless an intermediate result supports another branch.
2. Retrieval obtains new KG evidence from supplied Topic entities or dependency answers. Auto only combines or selects values already retrieved by all dependencies.
3. Order IDs so dependencies reference only earlier IDs.
4. For candidate filtering, retrieve narrow candidates, then retrieve those satisfying the remaining condition. Return the candidate type; property values serve as evidence.
5. Every Topic entity must be an anchor or explicit goal constraint. Use only supplied Topic entities as anchors, and null when starting from dependencies or using auto.

OUTPUT CONTRACT
Output a valid JSON array only:
[{"id":"g1","goal":"...","depend":[],"answer_type":"...","anchor_entity":"Topic entity","resolve_mode":"retrieval"}]"""

decompose_user_prompt = Template("""EXAMPLES:
Question:
What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
Topic entities:
George Washington Colonials men's basketball
Complexity reason:
composition: requires finding the university represented by the sports team, then retrieving the state of that university
Output:
[{"id":"g1","goal":"Find the university represented in sports by George Washington Colonials men's basketball.","depend":[],"answer_type":"university","anchor_entity":"George Washington Colonials men's basketball","resolve_mode":"retrieval"},{"id":"g2","goal":"Find the state where the university found in g1 is located.","depend":["g1"],"answer_type":"state","anchor_entity":null,"resolve_mode":"retrieval"}]

Question:
Which person who identified with Anglicanism influenced C. S. Lewis?
Topic entities:
C. S. Lewis, Anglicanism
Complexity reason:
conjunction: retrieve people who influenced C. S. Lewis, then verify which candidates are affiliated with Anglicanism
Output:
[{"id":"g1","goal":"Find people who influenced C. S. Lewis.","depend":[],"answer_type":"person","anchor_entity":"C. S. Lewis","resolve_mode":"retrieval"},{"id":"g2","goal":"Among the people found in g1, find those affiliated with Anglicanism.","depend":["g1"],"answer_type":"person","anchor_entity":null,"resolve_mode":"retrieval"}]

ACTUAL INPUT:
Question:
$question
Topic entities:
$topic_entities
Complexity reason:
$decomposition_reason
Output:""")


# 2. Relation and entity pruning
extract_relation_system_prompt = f"""TASK
Select relations that best advance the static subgoals.

AVAILABLE INFORMATION
Use only the supplied information, without internal knowledge or assumed KG relations.

{_STATIC_SUBGOAL_DEFINITIONS}

RELATION-SELECTION RULES
1. Prefer relations that directly answer or advance the static subgoals.
2. Use the planned dependencies and the supplied path history to identify useful next relations.

DIRECTION LABELS
1. outgoing: Current Entity --relation--> ?
2. incoming: ? --relation--> Current Entity

OUTPUT CONTRACT
Output exactly TopK distinct relation indices in ranked order, one per line:
[index]"""

extract_relation_user_prompt = Template("""EXAMPLES:
Question: Which team did Tim Howard play for?
Path History: None
Current Entity: Tim Howard
Static Subgoals:
g1: Find teams Tim Howard played for. | depend: None | answer_type: sports team
Candidate Relations:
[0] sports pro_athlete teams (outgoing)
[1] sports sports_team_roster player (incoming)
[2] people person nationality (outgoing)
TopK: 2
Output:
[0]
[1]

ACTUAL INPUT:
Question:
$question
Path History:
$path_history
Current Entity:
$entity
Static Subgoals:
$subgoal_context
Candidate Relations:
$relations
TopK: $topk
Output:""")


cvt_relation_system_prompt = f"""TASK
Select the most useful second-hop relations from the CVT mediator candidates.

AVAILABLE INFORMATION
Use only the supplied information, without internal knowledge or assumed KG facts, entities, or relations.

{_STATIC_SUBGOAL_DEFINITIONS}

RELATION-SELECTION RULES
1. Prefer relations that directly answer or advance the static subgoals.
2. Use the planned dependencies and the supplied path history to identify useful next relations.

DIRECTION LABELS
1. outgoing: [CVT] --relation--> ?
2. incoming: ? --relation--> [CVT]

OUTPUT CONTRACT
Output exactly TopK distinct relation indices in ranked order, one per line:
[index]"""

cvt_relation_user_prompt = Template("""EXAMPLES:
Question: Where did Barack Obama go to school?
Path History: Barack Obama -> people person education -> [CVT]
Current Entity: Barack Obama
First Relation (into CVT): people person education
Static Subgoals:
g1: Find schools attended by Barack Obama. | depend: None | answer_type: school
Candidate CVT Relations:
[0] education education institution (outgoing)
[1] education education degree (outgoing)
TopK: 1
Output:
[0]

ACTUAL INPUT:
Question:
$question
Path History:
$path_history
Current Entity:
$current_entity
First Relation (into CVT):
$first_relation
Static Subgoals:
$subgoal_context
Candidate CVT Relations:
$relations
TopK:
$topk
Output:""")


entity_prune_system_prompt = f"""TASK
Select candidate next-step entities or values that best advance the static subgoals.

AVAILABLE INFORMATION
Use only the supplied information, without internal knowledge or assumed facts, properties, or KG connections about candidates.

{_STATIC_SUBGOAL_DEFINITIONS}

CANDIDATE-SELECTION RULES
1. Prefer candidates that directly answer or advance the static subgoals.
2. Use the planned dependencies and the supplied path history to identify useful next entities or values.

OUTPUT CONTRACT
Output exactly TopK distinct candidate indices in ranked order, one per line:
[index]"""

entity_prune_user_prompt = Template("""EXAMPLES:
Question: Which team owned by Malcolm Glazer had Tim Howard playing for it?
Path History: Tim Howard -> teams -> Manchester United F.C.
Current Entity: Manchester United F.C.
Current Relation: sports professional_sports_team owner_s
Static Subgoals:
g1: Find teams Tim Howard played for. | depend: None | answer_type: sports team
g2: Among teams in g1, find those owned by Malcolm Glazer. | depend: g1 | answer_type: sports team
Candidate Next Step Paths:
[0] Malcolm Glazer | local_path=Manchester United F.C. -> sports professional_sports_team owner_s -> Malcolm Glazer
[1] Manchester | local_path=Manchester United F.C. -> sports sports_team location -> Manchester
TopK: 1
Output:
[0]

ACTUAL INPUT:
Question:
$question
Path History:
$path_history
Current Entity:
$current_entities
Current Relation:
$relation
Static Subgoals:
$subgoal_context
Candidate Next Step Paths:
$candidates
TopK:
$topk
Output:""")


# 3. Original-question path evaluation
evaluate_system_prompt = """TASK
Score how useful each candidate KG reasoning path is for answering the original question.

AVAILABLE INFORMATION
Judge only from the original question and the given path. Do not use internal knowledge to supply missing facts or infer unshown relations.

ABSOLUTE SCORING RUBRIC
0.0-0.3: Irrelevant, incorrect, or misleading. The path does not provide meaningful evidence for any required constraint in the original question.
0.4-0.6: Weakly useful. The path is related to the question or moves in a plausible direction, but it does not explicitly verify a required constraint.
0.7-0.8: Strong partial evidence. The path explicitly verifies at least one required constraint, but it does not satisfy all constraints needed to answer the original question.
0.9-1.0: Answer-level evidence. The path explicitly satisfies all required relations and constraints in the original question using one consistent entity binding, and directly supports the final answer.

SCORING RULES
1. Score each path independently using the absolute rubric, without normalizing across candidates. All paths may score high or low.
2. Include every candidate index exactly once, with its score to one decimal place and no additional fields or text.

OUTPUT CONTRACT
Output valid JSON only:
[
  {"index": 0, "score": 0.0},
  {"index": 1, "score": 0.0}
]"""

evaluate_user_prompt = Template("""EXAMPLES:
Question:
Which team owned by Malcolm Glazer has Tim Howard playing for it?
Candidate Paths:
[0] Tim Howard -> sports pro_athlete teams -> [CVT] -> sports sports_team_roster team -> Manchester United F.C. -> sports professional_sports_team owner_s -> Malcolm Glazer
[1] Malcolm Glazer <- sports professional_sports_team owner_s <- Manchester United F.C.
[2] Tim Howard -> people person nationality -> United States
Output:
[
  {"index": 0, "score": 1.0},
  {"index": 1, "score": 0.8},
  {"index": 2, "score": 0.1}
]

ACTUAL INPUT:
Question:
$question
Candidate Paths:
$candidate_paths
Output:""")


# 4. Early stopping and final answering
verify_answer_system_prompt = """TASK
Determine whether the supplied evidence satisfies all constraints in the original question and supports unambiguous answer(s).

AVAILABLE INFORMATION
Base the decision only on the supplied reasoning paths, considering all of them together.

VERIFICATION RULES
1. Each answer must satisfy all required relations and constraints under one consistent entity binding; never combine different candidates into one answer.
2. Missing, unsupported, or ambiguous constraints make the evidence insufficient.

OUTPUT CONTRACT
If information is sufficient, output JSON only:
{
  "stop": true,
  "answer": "answer text",
  "paths": [1, 2]
}
If information is insufficient, output JSON only:
{
  "stop": false
}"""

verify_answer_user_prompt = Template("""EXAMPLES:
Question:
Which team owned by Malcolm Glazer has Tim Howard playing for it?
Reasoning Paths:
Path 1 (path_value: 1.0): Tim Howard -> sports pro athlete teams -> [CVT] -> sports sports team roster team -> Manchester United F.C. -> sports professional sports team owner s -> Malcolm Glazer
Path 2 (path_value: 0.8): Tim Howard -> sports pro athlete teams -> [CVT] -> sports sports team roster team -> Everton F.C.
Verdict:
{
  "stop": false
}

Question:
Which school with the fight song "The Orange and Blue" did Emmitt Smith play for?
Reasoning Paths:
Path 1 (path_value: 1.0): Emmitt Smith -> sports pro athlete teams -> [CVT] -> sports sports team roster team -> Florida Gators football -> sports sports team fight song -> The Orange and Blue
Verdict:
{
  "stop": true,
  "answer": "Florida Gators football",
  "paths": [1]
}

ACTUAL INPUT:
Question:
$question
Reasoning Paths:
$paths
Verdict:""")


final_answer_from_paths_system_prompt = f"""TASK
Answer the question directly from the supplied KG evidence paths.

{_FINAL_ANSWER_POLICY}"""

final_answer_from_paths_user_prompt = Template("""Question:
$question
KG Evidence Paths:
$evidence_paths
Answer:""")


missing_topic_cot_system_prompt = """You are an AI assistant for knowledge-base question answering. Reason carefully and keep the final answer concise."""

missing_topic_cot_user_prompt = Template(
    _DIRECT_COT_FEW_SHOT_PROMPT + "\n\nQ: $question\nA:"
)
