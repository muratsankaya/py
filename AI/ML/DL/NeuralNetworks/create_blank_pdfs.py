import argparse
import re
import shutil

n_free_response = 9


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("netid", type=str)
    args = parser.parse_args()

    netid = args.netid.lower()
    assert re.search(r"^[a-z]{3}[0-9]{3,4}$", netid) is not None, "Your NetID looks like xyz0123"

    for i in range(1, n_free_response + 1):
        shutil.copyfile("blank.pdf", f"{netid}_q{i}.pdf")
