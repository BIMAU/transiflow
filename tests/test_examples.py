import matplotlib

matplotlib.use('Agg')

def test_ldc_example():
    from examples.ldc import main

    main(nx=4)
