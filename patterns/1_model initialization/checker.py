import core.core as core

test_scripts = [r"1_test_code.py"]

isadd, isfind = core.main(r"1_subchecker.json",test_scripts)

print(isadd, isfind)