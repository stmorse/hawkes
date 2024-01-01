import numpy as np

from MHP import MHP

expected_seq = np.array([[6.799319039689095, 0], [6.832330144004188, 0]])
expected_multiple_seq = np.array(
    [
        [3.39965952, 0.0],
        [3.40566154, 0.0],
        [12.48864946, 0.0],
        [13.17625391, 0.0],
        [13.46781712, 1.0],
        [14.65373507, 2.0],
        [14.77459075, 0.0],
    ]
)

m = np.array([0.2, 0.0, 0.0])
a = np.array([[0.1, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 0.9, 0.0]])
w = 3.1


def test_generate_seq():
    P = MHP(seed=0)
    P.generate_seq(10)
    np.testing.assert_array_equal(P.data, expected_seq)


def test_generate_seq_multiple():
    P = MHP(mu=m, alpha=a, omega=w, seed=0)
    P.generate_seq(15)
    np.testing.assert_allclose(P.data, expected_multiple_seq)


def test_EM():
    P = MHP(seed=0)
    P.data = expected_seq
    alpha = np.array([[0.5]])
    mu = np.array([0.1])
    omega = 1.0
    expected = (np.array([[0.28310623]]), np.array([0.20985337]))
    for i, value in enumerate(P.EM(alpha, mu, omega)):
        np.testing.assert_allclose(value, expected[i])


def test_EM_multiple():
    P = MHP(seed=0)
    P.data = expected_multiple_seq
    expected = (
        np.array([[0.1676253, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([0.28169129, 0.0, 0.0]),
    )
    for i, value in enumerate(P.EM(a, m, w)):
        np.testing.assert_allclose(value, expected[i])
