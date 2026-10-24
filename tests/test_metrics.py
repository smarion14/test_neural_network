import numpy as np

from nn import accuracy, confidence_table, confusion_matrix, majority_baseline


def test_confidence_table_counts_only_the_games_the_model_is_sure_about():
    p = np.array([[0.95], [0.85], [0.35], [0.55]])           # sure of: 95% blue, 85% blue, 65% red, 55% blue
    y = np.array([[1.0], [0.0], [0.0], [1.0]])               # the 85% call is wrong, the others are right
    rows = confidence_table(y, p, thresholds=(0.6, 0.9, 0.99))
    assert rows == [(0.6, 0.75, 2 / 3), (0.9, 0.25, 1.0)]    # nobody reaches 99%, so that row is left out


def test_accuracy_confusion_and_baseline_on_a_hand_checked_example():
    y = np.array([[1.0], [1.0], [0.0], [0.0], [1.0]])
    p = np.array([[0.9], [0.4], [0.2], [0.7], [0.6]])        # right, wrong, right, wrong, right
    assert accuracy(y, p) == 0.6
    assert confusion_matrix(y, p).tolist() == [[1, 1], [1, 2]]
    assert majority_baseline(y) == 0.6


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
