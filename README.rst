CALCLOUD-AI
======================

.. image:: https://readthedocs.org/projects/stsci-package-template/badge/?version=latest
    :target: https://stsci-package-template.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://github.com/spacetelescope/stsci-package-template/workflows/CI/badge.svg
    :target: https://github.com/spacetelescope/stsci-package-template/actions
    :alt: GitHub Actions CI Status

.. image:: https://codecov.io/gh/spacetelescope/stsci-package-template/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/spacetelescope/stsci-package-template
    :alt: Coverage Status

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge


Artifical Neural Networks for predicting compute resource requirements for Hubble Space Telescope data reprocessing in AWS Lambda (CALCLOUD).

Install
-------

Using setup.py::

    git clone https://grit.stsci.edu/rkein/calcloud-machine-learning

    cd calcloud-machine-learning

    python setup.py install --user


Using virtual env::

    git clone https://grit.stsci.edu/rkein/calcloud-machine-learning

    python virtualenv dash-venv

    source dash-venv/bin/activate

    cd calcloud-machine-learning
    
    pip install -r requirements.txt


Run
-------
.. code-block:: bash
    $ cd calcloudML
    $ python app.py


View the running dashboard in your browser: http://127.0.0.1:8050/


License
-------

See `LICENSE.rst` for more information.
