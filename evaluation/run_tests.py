# Copyright 2022 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import json
import os
from copy import copy

from args import DataArgs, ExpArgs, ModelArgs, NeedleTestItem
from async_evaluate import NoLiMa_Tester
from jsonargparse import ArgumentParser


def prepare_tests(exp_config: NeedleTestItem):
    system_prompt = exp_config["system_prompt"]
    exp_id = exp_config["id"]
    for question_type, question in exp_config["questions"].items():
        for test_id, test in exp_config["tests"].items():
            full_needle = "" + exp_config["needle"]
            input_args = test["input_args"]
            new_item = {
                "test_name": f"{exp_id}_{test_id}_{question_type}",
                "system_prompt": system_prompt,
                "task_template": data_args.task_template,
                "gold_answers": test["gold_answers"] if "gold_answers" in test else "",
                "seed": exp_args.base_seed + int(exp_id[:4]),
                "character_set": exp_config.get("character_set", ""),
            }

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
                    full_distractor = exp_config["distractors"][question_type].replace(
                        arg_placeholder, arg
                    )
            new_item["needle"] = full_needle
            new_item["retrieval_question"] = full_question
            new_item["distractor"] = full_distractor

            yield new_item


def run_test(
    model_args: ModelArgs,
    data_args: DataArgs,
    exp_args: ExpArgs,
):
    with open(data_args.needle_set_path, "r") as file:
        needle_set: list[NeedleTestItem] = json.load(file)

    tests = [test for exp_config in needle_set for test in prepare_tests(exp_config)]

    haystacks = os.listdir(data_args.haystack_dir)
    for haystack_no, haystack in enumerate(haystacks):
        haystack_path = os.path.join(data_args.haystack_dir, haystack)
        haystack_name = haystack.split(".")[0]

        for test in tests:
            tester = NoLiMa_Tester(
                model_args=model_args,
                needle=test["needle"],
                haystack_path=haystack_path,
                results_dir=os.path.join(
                    exp_args.parent_results_dir, test["test_name"]
                ),
                retrieval_question=test["retrieval_question"],
                gold_answers=json.dumps(test["gold_answers"])
                if test["gold_answers"] != ""
                else "",
                character_set=json.dumps(test["character_set"])
                if test["character_set"] != ""
                else "",
                system_prompt=test["system_prompt"],
                use_default_system_prompt=data_args.use_default_system_prompt,
                task_template=test["task_template"],
                context_length=data_args.context_length,
                document_depth_percent_min=data_args.document_depth_percent_min,
                document_depth_percent_max=data_args.document_depth_percent_max,
                document_depth_percent_intervals=data_args.document_depth_percent_intervals,
                shift=data_args.shift,
                static_depth=data_args.static_depth,
                metric=exp_args.metric,
                log_placements_dir=exp_args.log_placements_dir,
                test_name=test["test_name"],
                seed=test["seed"] + haystack_no,
                prevent_duplicate=exp_args.prevent_duplicate_tests,
                distractor=test["distractor"],
            )

            tester.evaluate()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_dataclass_arguments(ModelArgs, "model")
    parser.add_dataclass_arguments(DataArgs, "data")
    parser.add_dataclass_arguments(ExpArgs, "exp")

    args = parser.parse_args()

    model_args: ModelArgs = args.model
    data_args: DataArgs = args.data
    exp_args: ExpArgs = args.exp

    run_test(
        model_args=model_args,
        data_args=data_args,
        exp_args=exp_args,
    )
