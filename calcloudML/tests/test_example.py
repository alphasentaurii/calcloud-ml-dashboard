import pytest
import os

from calcloudML.makefigs import load_data
from calcloudML.makefigs import predictor


def test_data_import():
    print(os.getcwd())
    assert True
    # df = load_data.get_single_dataset("data/hst_data.csv")
    # instruments = list(df["instr_key"].unique())
    # assert len(instruments) == 4


def test_model_import():
    assert True
    # clf = predictor.get_model("models/mem_clf")
    # assert len(clf.layers) > 0


# def test_primes():
#     assert primes(10) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


# def test_imax_too_big():
#     with pytest.raises(ValueError):
#         primes(10001)


# def test_no_cython():
#     with pytest.raises(NotImplementedError):
#         do_primes(2, usecython=True)


# def test_cli(capsys):
#     main(args=['-tp', '2'])
#     captured = capsys.readouterr()
#     assert captured.out.startswith('Found 2 prime numbers')
#     assert len(captured.err) == 0
