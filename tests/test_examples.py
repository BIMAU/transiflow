import pytest
import matplotlib

matplotlib.use('Agg')

@pytest.mark.example
def test_ldc_example():
    from examples.ldc import main

    main(nx=4)
