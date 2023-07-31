import core.core as core

test_scripts = [r"5_test_code.py"]

isadd, isfind = core.main(r"5_subchecker.json",test_scripts)

print(isadd, isfind)