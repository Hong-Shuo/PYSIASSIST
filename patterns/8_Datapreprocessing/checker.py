import core.core as core

test_scripts = [r"8_test_code.py"]

isadd, isfind = core.main(r"8_subchecker.json",test_scripts)

print(isadd, isfind)