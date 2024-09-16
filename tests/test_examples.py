import pytest
import matplotlib

matplotlib.use('Agg')

@pytest.mark.example
def test_ldc_example():
    from examples.ldc import main

    main(nx=4)

@pytest.mark.example
def test_ldc2_example():
    from examples.ldc2 import main

    main(nx=4)
