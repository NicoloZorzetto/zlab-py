"""
This is a quick test for the zform function
using the penguins seaoborn dataset

zform is part of the zlab library by Nicolò Zorzetto

License
-------
GPL v3
"""
import os
import sys

try:
    # Works when running as a file
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    # Fallback for interactive environments (no __file__)
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


from zlab import zform

def try_zform_penguins():
    try:
        import seaborn as sbn
    except ImportError:
        raise ImportError(
            "You do not have seaborn installed"
        )
    while True:
        which_data = input("Would you like to run the small (input 1) or big (input 2) dataset? ")
        if which_data == "1":
            df = sbn.load_dataset("penguins").dropna()
            break
        if which_data == "2":
            df = sbn.load_dataset("penguins").dropna()
            df = pd.concat([df] * 1000, ignore_index=True)
            break
        else:
            print("Only input 1 for the small dataset, or 2 for the big one.")
    df_out, forms = zform(
        df,
        group_col="species",
        return_results=True,
	    min_obs = 30,
	    apply = True,
	    naming = "standard",
        export_csv="./forms.csv",
        n_jobs=-1
    )


if __name__ == "__main__":
    while True:
        run_test_input = input("Would you like to run the penguins test? (y/n) ")
        if run_test_input.lower()[0] == "y":
            try_zform_penguins()
            break
        if run_test_input.lower()[0] == "n":
            print("Not running the test.")
            break
        else:
            print("Please input y or n")
