# unit_test_rl
Project code for training LLMs to write better unit tests + code

Warning: This is experiment code and isn't necessarily production ready or easy to use (E.g. There may be hardcoded paths to local data files). I'm just sharing some experiments that I thought were fun.

You may need this fork of OpenRLHF, I made some modifications to support guided decoding and pass arguments to the reward model: https://github.com/rosmineb/OpenRLHF branch custom_changes_main

To run:

1. Start the reward server, `python unit_tests_server/unit_test_reward_server.py --workers 8 --port 5432`
2. Call the run.sh experiment `source experiments/u6_7B_form/run.sh`
