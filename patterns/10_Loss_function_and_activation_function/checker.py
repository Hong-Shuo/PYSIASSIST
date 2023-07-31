import core.core as core
import os
test_scripts = [r"10_test_code.py"]
Loss_function_and_activation_function_subcheckers=[filename for filename in os.listdir('.') if filename.startswith("10_subchecker") and filename.endswith('.json')]
for subchecker_json in Loss_function_and_activation_function_subcheckers:
    isadd,isfind = core.main(subchecker_json,test_scripts)
    if isfind:
        break
