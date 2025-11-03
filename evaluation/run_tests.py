# Copyright 2022 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import json
import os
from copy import copy

from async_evaluate import NoLiMa_Tester
from jsonargparse import ActionConfigFile, ArgumentParser

DEFAULT_TASK_TEMPLATE = "You will answer a question based on the following book snippet:\n\n{haystack}\n\nUse the information provided in the book snippet to answer the question. Your answer should be short and based on either explicitly stated facts or strong, logical inferences.\n\nQuestion: {question}\n\n Return only the final answer with no additional explanation or reasoning."

if __name__ == "__main__":
    parser = ArgumentParser(description="NoLiMa Multi-setup Tester")
    parser.add_argument(
        "--config", action=ActionConfigFile, help="Path to a configuration YAML file."
    )
    parser.add_argument("--model_name", type=str, help="Name of the model to test")
    parser.add_argument(
        "--model_configs_dir",
        type=str,
        help="Directory containing model configurations",
    )
    parser.add_argument(
        "--needle_set_path",
        type=str,
        help="Path to a file containing the needle tests configuration",
    )
    parser.add_argument(
        "--haystack_dir", type=str, help="Directory containing the haystack files"
    )
    parser.add_argument("--task_template", type=str, help="Name of the model to test")
    parser.add_argument(
        "--use_default_system_prompt", type=bool, help="Use default system prompt"
    )
    parser.add_argument(
        "--parent_results_dir", type=str, help="Parent directory to save results"
    )
    parser.add_argument(
        "--context_length", type=int, help="Context length for the needle placement"
    )
    parser.add_argument(
        "--document_depth_percent_min",
        type=float,
        default=0,
        help="Minimum document depth percentage",
    )
    parser.add_argument(
        "--document_depth_percent_max",
        type=float,
        default=100,
        help="Maximum document depth percentage",
    )
    parser.add_argument(
        "--document_depth_percent_intervals",
        type=int,
        default=35,
        help="Number of intervals between min and max depth",
    )
    parser.add_argument(
        "--shift",
        type=int,
        default=0,
        help="Shift applied to the beginning of the haystack",
    )
    parser.add_argument(
        "--static_depth",
        type=float,
        default=-1,
        help="Static depth for needle placement",
    )
    parser.add_argument("--metric", type=str, default="EM", help="Evaluation metric")
    parser.add_argument(
        "--log_placements_dir",
        type=str,
        default="",
        help="Directory to save needle placements for debugging",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Base seed for random character selection",
    )
    parser.add_argument(
        "--prevent_duplicate_tests",
        type=bool,
        default=True,
        help="Prevent duplicate tests",
    )

    args = parser.parse_args()

    with open(args.needle_set_path, "r") as file:
        needle_set = json.load(file)

    tests = []

    for exp_config in needle_set:
        system_prompt = exp_config["system_prompt"]
        exp_id = exp_config["id"]
        for question_type, question in exp_config["questions"].items():
            for test_id, test in exp_config["tests"].items():
                full_needle = "" + exp_config["needle"]
                input_args = test["input_args"]
                tests.append(
                    {
                        "test_name": exp_id + "_" + test_id + "_" + question_type,
                        "system_prompt": system_prompt,
                        "task_template": DEFAULT_TASK_TEMPLATE
                        if "task_template" not in exp_config
                        else exp_config["task_template"],
                        "gold_answers": test["gold_answers"]
                        if "gold_answers" in test
                        else "",
                        "seed": args.base_seed + int(exp_id[:4]),
                    }
                )
                if "character_set" in exp_config:
                    tests[-1]["character_set"] = exp_config["character_set"]
                else:
                    tests[-1]["character_set"] = ""
                full_question = copy(question)
                full_distractor = None
                for arg_no, arg in enumerate(input_args):
                    arg_placeholder = "{" + str(arg_no + 1) + "}"
                    if arg_placeholder in question:
                        full_question = question.replace(arg_placeholder, arg)
                    if arg_placeholder in full_needle:
                        full_needle = full_needle.replace(arg_placeholder, arg)
                    if (
                        "distractors" in exp_config
                        and arg_placeholder in exp_config["distractors"][question_type]
                    ):
                        full_distractor = exp_config["distractors"][
                            question_type
                        ].replace(arg_placeholder, arg)
                tests[-1]["needle"] = full_needle
                tests[-1]["retrieval_question"] = full_question
                tests[-1]["distractor"] = full_distractor

    haystacks = os.listdir(args.haystack_dir)
    for haystack_no, haystack in enumerate(haystacks):
        haystack_path = os.path.join(args.haystack_dir, haystack)
        haystack_name = haystack.split(".")[0]

        for test in tests:
            tester = NoLiMa_Tester(
                model_name=args.model_name,
                model_configs_dir=args.model_configs_dir,
                needle=test["needle"],
                haystack_path=haystack_path,
                results_dir=os.path.join(args.parent_results_dir, test["test_name"]),
                retrieval_question=test["retrieval_question"],
                gold_answers=json.dumps(test["gold_answers"])
                if test["gold_answers"] != ""
                else "",
                character_set=json.dumps(test["character_set"])
                if test["character_set"] != ""
                else "",
                system_prompt=test["system_prompt"],
                use_default_system_prompt=args.use_default_system_prompt,
                task_template=test["task_template"],
                context_length=args.context_length,
                document_depth_percent_min=args.document_depth_percent_min,
                document_depth_percent_max=args.document_depth_percent_max,
                document_depth_percent_intervals=args.document_depth_percent_intervals,
                shift=args.shift,
                static_depth=args.static_depth,
                metric=args.metric,
                log_placements_dir=args.log_placements_dir,
                test_name=test["test_name"],
                seed=test["seed"] + haystack_no,
                prevent_duplicate=args.prevent_duplicate_tests,
                distractor=test["distractor"],
            )

            tester.evaluate()
