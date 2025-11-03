# Copyright 2022 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import json
import os

from vllm import LLM, SamplingParams

with open("needle_set.json", "r") as f:
    needle_set = json.load(f)

# Gather all question from the needle set
question_set = []
question_config_map = {}
for exp_config in needle_set:
    for question in exp_config["questions"].values():
        for test_id, test in exp_config["tests"].items():
            args = test["input_args"]
            for arg_no, arg in enumerate(args):
                arg_placeholder = "{" + str(arg_no + 1) + "}"
                if arg_placeholder in question:
                    full_question = question.replace(arg_placeholder, arg)
                    text_code = (
                        exp_config["id"] + "_" + test_id + "_Arg" + str(arg_no + 1)
                    )
                    if full_question not in question_set:
                        question_set.append(full_question)
                        question_config_map[full_question] = [text_code]
                    elif text_code not in question_config_map[full_question]:
                        question_config_map[full_question].append(text_code)

print("Gathered", len(question_set), "questions")

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
CHUNK_SIZE = 1000
STRIDE = 800
MAX_GEN_LEN = 256
TEMP = 0.0
N_GPUs = 8

# Load the model through vLLM
llm_model = LLM(MODEL_NAME, tensor_parallel_size=N_GPUs, enable_prefix_caching=True)
sampling_params = SamplingParams(temperature=TEMP, max_tokens=MAX_GEN_LEN)
print("Model loaded")

BOOKPATH = "./books/II-clean/wo_dist/"
booknames = os.listdir(BOOKPATH)

CONTROL_RESULTS_DIR = "filtering_results/round_1/"
os.makedirs(CONTROL_RESULTS_DIR, exist_ok=True)

for bookname in booknames:
    # Load books one by one and then prompt using the questions
    if not bookname.endswith(".txt"):
        continue

    FILENAME = bookname

    with open(BOOKPATH + FILENAME, "r", encoding="utf-8") as f:
        booktext = f.read()
    print("Read book with", len(booktext), "characters")

    full_results = {}
    full_results["questions"] = question_set
    full_results["questions_wid"] = {
        question: i for i, question in enumerate(question_set)
    }
    full_results["question_config_map"] = question_config_map
    full_results["filename"] = FILENAME
    full_results["CHUNK_SIZE"] = CHUNK_SIZE
    full_results["STRIDE"] = STRIDE
    full_results["TEMP"] = TEMP
    full_results["MODEL_NAME"] = MODEL_NAME
    full_results["summarized_results"] = {
        "total": 0,
        "yes": 0,
        "no": 0,
        "per_question": {
            question: {"total": 0, "yes": 0, "no": 0} for question in question_set
        },
    }
    full_results["results"] = []

    prompts = []
    chunk_start = 0

    # Chunk the book into smaller parts
    while chunk_start < len(booktext):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(booktext))
        chunk_text = booktext[chunk_start:chunk_end]

        for q_id, question in enumerate(question_set):
            conversation = [
                {
                    "role": "system",
                    "content": "You'll be given a text snippet and a question afterward. You must answer the question based on the information in the text snippet. The answer should either be based on a *direct mention* or a *strong inference*. IMPORTANT: The response should include an explanation leading to the final answer or, if there is no answer, write N/A.",
                },
                {
                    "role": "user",
                    "content": """Story: .... Rebecca stood perfectly still in the centre of the floor and looked about her. There was a square of oilcloth in front of each article of furniture and a drawn-in rug beside the single four poster, which was covered with a fringed white dimity counterpane. 
 Everything was as neat as wax, but the ceilings were much higher than Rebecca was accustomed to. It was a north room, and the window, which was long and narrow, looked out on the back buildings and the barn. Jack is a vegetarian.
 It was not the room, which was far more comfortable than Rebecca's own at the farm, ...

Question: Which character should not eat an scrambled eggs?""",
                },
                {"role": "assistant", "content": "N/A"},
                {
                    "role": "user",
                    "content": """Story: Imagine what it would do to you if at mile 20 of a
marathon, someone ran up beside you and said "You must feel really
tired.  Would you like to stop and take a rest?"  Conversations
with corp dev are like that but worse, because the suggestion of
stopping gets combined in your mind with the imaginary high price
you think they'll offer.And then you're really in trouble.  If they can, corp dev people
like to turn the tables on you. They like to get you to the point
where you're trying to convince them to buy instead of them trying
to convince you to sell.  And surprisingly often they succeed. John had this experience in University of Sao Paulo. This is a very slippery slope, greased with some of the most powerful
forces that can work on founders' minds, and attended by an experienced
professional whose full time job is to push you down it.Their tactics in pushing you down that slope are usually fairly...

Question: Which character has been to Brazil?""",
                },
                {
                    "role": "assistant",
                    "content": "It is mentioned that John had some experience in University of Sao Paulo which is in Brazil. -- John",
                },
                {
                    "role": "user",
                    "content": """Story: ... "Nothing wrong with that," Foeren said. "Didn't you hear what the man said? This is our planet!" 
 "With an average life expectancy of three Earth years," Barrent reminded him. 
 "...
...er what a credit  thief  is. But perhaps it'll come back to me." 
 "Maybe the authorities have some sort of memory retraining system," Foeren said. 
 "Authorities?" Joe said indignantly. "What do you mean, authorities? This is  our  planet. We're all equal here. By definition, there can't be any authorities. No, friends, we left all that nonsense behind on Earth. Here we—" 
 He stopped abruptly. The barracks' door had opened and a man walked in. He was evidently an older resident of Omega since he lacked the gray prison uniform. He was fat, and dressed in garish yellow and blue clothing. On a belt around his ample waist he carried a holstered pistol and a knife. He stood just inside the doorway, his hands on his hips, glaring at the new arrivals. 
 "Well?" he said. "Don't you new men recognize a Quaestor? Stand up!" 
 None of the men moved. ...

Question: Which character has been to China?""",
                },
                {"role": "assistant", "content": "N/A"},
                {
                    "role": "user",
                    "content": """Story: ... He  remembered  this town, and the monotonous houses had individuality and meaning for him. He had been born and raised in this town. A message came in saying, \"I'm a pescatarian,\" from Carol.
 There was Grothmeir's store, and across the street was the home of Havening, the local interior decorating champion. Here was Billy Havelock's house. Billy had been his best friend. They had planned on being starmen together, and had remained good friends after school—until Barrent had been sentenced to Omega. 
 Here was Andrew Therkaler's house. And down the block was the school he had attended. He could remember the classes. He could remember how, every day, they had gone through the door that led to the closed class. But he still could not remember what he had learned there. 
 Right here, near two huge elms, the murder had taken place....

Question: Which character should not eat hotdogs?""",
                },
                {
                    "role": "assistant",
                    "content": "The text mentions that Carol is a pescatarian. Therefore, she shouldn't eat hotdogs. -- Carol",
                },
                {
                    "role": "user",
                    "content": "Story: "
                    + ("..." if chunk_start > 0 else "")
                    + chunk_text
                    + "..."
                    + "\n\nQuestion: "
                    + question,
                },
            ]
            prompts.append(conversation)
            full_results["results"].append(
                {"q_id": q_id, "chunk_start": chunk_start, "chunk_end": chunk_end}
            )

        chunk_start += STRIDE

    # Generate outputs
    outputs = llm_model.chat(
        messages=prompts, sampling_params=sampling_params, use_tqdm=True
    )

    # Store results
    for i, output in enumerate(outputs):
        full_results["results"][i]["response"] = output.outputs[0].text

        # Sometimes response is N/A, but it has some extra description which is still a flagged response
        full_results["results"][i]["metric"] = (
            0
            if "n/a" in full_results["results"][i]["response"].lower()
            and len(full_results["results"][i]["response"]) < 128
            else 1
        )
        if full_results["results"][i]["metric"] > 0:
            full_results["results"][i]["text_snippet"] = booktext[
                full_results["results"][i]["chunk_start"] : full_results["results"][i][
                    "chunk_end"
                ]
            ]

        full_results["summarized_results"]["total"] += 1
        full_results["summarized_results"]["yes"] += (
            full_results["results"][i]["metric"] == 1
        )
        full_results["summarized_results"]["no"] += (
            full_results["results"][i]["metric"] == 0
        )

        full_results["summarized_results"]["per_question"][
            full_results["questions"][full_results["results"][i]["q_id"]]
        ]["total"] += 1
        full_results["summarized_results"]["per_question"][
            full_results["questions"][full_results["results"][i]["q_id"]]
        ]["yes"] += full_results["results"][i]["metric"] == 1
        full_results["summarized_results"]["per_question"][
            full_results["questions"][full_results["results"][i]["q_id"]]
        ]["no"] += full_results["results"][i]["metric"] == 0

    print("Storing results...")
    output_name = "filtering_results_" + FILENAME.replace(".txt", "") + ".json"
    with open(
        os.path.join(CONTROL_RESULTS_DIR, output_name), "w", encoding="utf-8"
    ) as f:
        json.dump(full_results, f, indent=2)
