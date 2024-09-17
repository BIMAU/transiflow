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

@pytest.mark.example
def test_ldc3_example():
    from examples.ldc3 import main

    main(nx=4)

@pytest.mark.example
def test_ldc_3d_example():
    try:
        from examples.ldc_3d import main

        main(nx=4)
    except ImportError:
        pytest.skip("HYMLS not found")

@pytest.mark.example
def test_dhc_example():
    from examples.dhc import main

    main(nx=16)

@pytest.mark.example
def test_qg_example():
    from examples.qg import main

    main()
