from string import Template

io_prompt = Template("""Answer the question directly and return the answer in JSON format using the "answer" field.
Here are six examples:
Q: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
A: {"answer": "Washington, D.C."}

Q: Who lists Pramatha Chaudhuri as an influence and wrote Jana Gana Mana?
A: {"answer": "Bharoto Bhagyo Bidhata"}

Q: Who was the artist nominated for an award for You Drive Me Crazy?
A: {"answer": "Britney Spears"}

Q: What person born in Siegen influenced the work of Vincent Van Gogh?
A: {"answer": "Peter Paul Rubens"}

Q: What is the country close to Russia where Mikheil Saakashvii holds a government position?
A: {"answer": "Georgia"}

Q: What drug did the actor who portrayed the character Urethane Wheels Guy overdosed on?
A: {"answer": "Heroin"}

Now directly output the answer to the following question in JSON format using the "answer" field without other information or notes.
Q: $question
A: """)


cot_prompt = Template("""Answer the question step by step and return the results in JSON format using the "reason" and "answer" fields.
Here are six examples:
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
    "reason": "First, \\"You Drive Me Crazy\\" is performed by Britney Spears. Therefore, the artist referred to in the question is Britney Spears.",
    "answer": "Britney Spears"
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

Now directly output the results of the following question in JSON format using the "reason" and "answer" fields without other information or notes.
Q: $question
A: """)


