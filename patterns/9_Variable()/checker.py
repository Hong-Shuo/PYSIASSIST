import core.core as core

test_scripts = [r"9_test_code.py"]

isadd, isfind = core.main(r"9_subchecker.json",test_scripts)

print(isadd, isfind)