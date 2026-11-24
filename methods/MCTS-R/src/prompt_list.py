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


# 2. Relation pruning
extract_relation_system_prompt = f"""TASK
Select next relations that best advance the static subgoals and the original question.

AVAILABLE INFORMATION
Use only the supplied topic entity, existing relation path and candidate relations.
The variables in the relation path are unknown entities or values, not observed answers.

{_STATIC_SUBGOAL_DEFINITIONS}

RULES
1. Prefer relations that directly advance a required constraint or connect a planned dependency.
2. Respect each candidate's direction relative to the current path endpoint.
3. Select exactly TopK distinct indices in descending usefulness, one per line:
[index]"""

extract_relation_user_prompt = Template("""EXAMPLE:
Question: Which teams did Tim Howard play for?
Topic Entity: Tim Howard
Relation Path: Tim Howard -> sports.pro_athlete.teams -> ?e1
Static Subgoals:
g1: Find teams Tim Howard played for. | depend: None | answer_type: sports team
Candidate Relations:
[0] sports.sports_team_roster.team (outgoing)
[1] sports.sports_team_roster.from (outgoing)
[2] people.person.nationality (outgoing)
TopK: 1
Output:
[0]

ACTUAL INPUT:
Question:
$question
Topic Entity:
$topic_entity
Relation Path:
$relation_path
Static Subgoals:
$subgoal_context
Candidate Relations:
$relations
TopK: $topk
Output:""")


# 3. Batch relation-path evaluation
evaluate_system_prompt = """TASK
Score each directed relation path, anchored at its topic entity, for answering the original question.

AVAILABLE INFORMATION
The path is a connected KG relation pattern with unknown entity/value bindings.
Judge semantic relevance, direction and structural coverage of the question, not actual answer correctness.
Do not invent intermediate or final entities, facts or unshown relations.

ABSOLUTE SCORING RUBRIC
0.0-0.3: Irrelevant, misleading, or incorrect relation semantics/direction.
0.4-0.6: Plausible intermediate path, but not yet covering a key requested relation.
0.7-0.8: Covers a key relation or part of the required structure; important constraints still need a path or grounding.
0.9-1.0: Structurally suitable to retrieve or verify the answer, covering the question's required relation pattern. A high score is not a guarantee that bindings satisfy the constraints.

SCORING RULES
1. Score independently on the absolute scale, without normalization.
2. Include each candidate index exactly once, using a score to one decimal place.

OUTPUT CONTRACT
Output valid JSON only:
[
  {"index": 0, "score": 0.0},
  {"index": 1, "score": 0.0}
]"""

evaluate_user_prompt = Template("""EXAMPLE:
Question:
Which teams did Tim Howard play for?
Candidate Relation Paths:
[0] Tim Howard -> sports.pro_athlete.teams -> ?e1 -> sports.sports_team_roster.team -> ?e2
[1] Tim Howard -> sports.pro_athlete.teams -> ?e1
[2] Tim Howard -> people.person.nationality -> ?e1
Output:
[
  {"index": 0, "score": 1.0},
  {"index": 1, "score": 0.7},
  {"index": 2, "score": 0.1}
]

ACTUAL INPUT:
Question:
$question
Candidate Relation Paths:
$candidate_paths
Output:""")


# 4. Structural sufficiency and grounded final answering
verify_answer_system_prompt = """TASK
Determine whether the supplied relation paths jointly cover the relation structure required by the original question, so relation exploration may stop and grounding may begin.

RULES
1. Inspect topic anchors, relation direction, variable roles and all requested constraints.
2. Paths are patterns, not concrete evidence; do not assume a variable equals a named entity or that different paths have matching bindings.
3. A complete pattern may still require entity grounding, joining compatible bindings, or filtering values. Stop only if the selected structures could support these operations without retrieving a missing relation.
4. Do not output an answer.

OUTPUT CONTRACT
If structurally sufficient, output JSON only: {"stop": true, "paths": [1, 2]}
Otherwise output JSON only: {"stop": false}
Path indices are one-based."""

verify_answer_user_prompt = Template("""EXAMPLES:
Question: Which teams did Tim Howard play for that were owned by Malcolm Glazer?
Relation Paths:
Path 1: Tim Howard -> sports.pro_athlete.teams -> ?e1 -> sports.sports_team_roster.team -> ?e2
Verdict:
{"stop": false}

Question: Which teams did Tim Howard play for?
Relation Paths:
Path 1: Tim Howard -> sports.pro_athlete.teams -> ?e1 -> sports.sports_team_roster.team -> ?e2
Verdict:
{"stop": true, "paths": [1]}

ACTUAL INPUT:
Question:
$question
Relation Paths:
$paths
Verdict:""")

final_answer_from_paths_system_prompt = f"""TASK
Answer the question from the grounded KG paths. Each binding row is one connected path instance.
Do not combine incompatible entity bindings into an answer. Use the intermediate entities when needed.
An unnamed entity ID is not a readable answer. A limited list of bindings is not proof of exhaustive coverage.

{_FINAL_ANSWER_POLICY}"""

final_answer_from_paths_user_prompt = Template("""Question:
$question
Grounded KG Evidence Paths:
$evidence_paths
Answer:""")


missing_topic_cot_system_prompt = "You are an AI assistant for knowledge-base question answering. Reason carefully and keep the final answer concise."

missing_topic_cot_user_prompt = Template(
    _DIRECT_COT_FEW_SHOT_PROMPT + "\n\nQ: $question\nA:"
)
